import json
import os
from typing import Any
import dotenv
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel, OllamaModel
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama
from tqdm import tqdm
from app.app import rephrase_question, search_context, ask_llm
import tenacity

dotenv.load_dotenv()

chat1 = ChatOllama(
    model="llama4:latest",
    base_url=os.environ["OLLAMA_BASE_URL"],
    temperature=0,
)
# Импортируйте класс OllamaModel из DeepEval
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "480"

evaluation_model = OllamaModel(
    model="llama4:latest",
    base_url="https://aicltr.itmo.ru/ollama/",
    temperature=0,  #
)

# evaluation_model = GeminiModel(
#     model="gemini-3-flash-preview",
#     api_key=os.environ["GEMINI_API_KEY"],
# )


def load_local_dataset(filepath: str) -> list[dict[str, Any]]:
    """Загружает датасет из JSON файла."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_rag_response_with_context(results: list[dict[str, str | list[str]]]) -> None:
    with open("rag_ollama_model_answers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def get_rag_response_with_context(query: str) -> dict[str, str | list[str]]:
    """
    Получает ответ И КОНТЕКСТ от вашей RAG-системы.
    ВАЖНО: Эта функция сейчас возвращает заглушку для `retrieval_context`.
    Для корректной оценки метрик ретривера и `Faithfulness` замените заглушку на реальный контекст, извлеченный вашим ретривером.
    """
    history = []
    rephrase = rephrase_question(query, history)
    context: list[str] = [c[1] for c in search_context(rephrase)]
    answer = ask_llm(query, context, history)
    return {
        "answer": answer,
        "retrieval_context": context
    }


contextual_relevancy = ContextualRelevancyMetric(
    model=evaluation_model,  
    threshold=0.7,
    include_reason=True
)

contextual_precision = ContextualPrecisionMetric(
    model=evaluation_model, 
    threshold=0.7,
    include_reason=True
)

contextual_recall = ContextualRecallMetric(
    model=evaluation_model, 
    threshold=0.7,
    include_reason=True
)


answer_relevancy = AnswerRelevancyMetric(
    model=evaluation_model, 
    threshold=0.7,
    include_reason=True
)

faithfulness = FaithfulnessMetric(
    model=evaluation_model,  
    threshold=0.7,
    include_reason=True
)


def create_test_cases(dataset_path: str) -> list[LLMTestCase]:
    """Загружает датасет, получает ответы и контекст, создает тест-кейсы."""
    dataset = load_local_dataset(dataset_path)
    test_cases = []
    results = []
    for idx, item in enumerate(dataset):
        print(f"Обработка {idx+1}/{len(dataset)}...")
        rag_result = get_rag_response_with_context(item)

        test_case = LLMTestCase(
            input=item['query'],
            actual_output=rag_result['answer'],
            expected_output=item.get('ground_truth', ''), 
            retrieval_context=rag_result['retrieval_context']  
        )
        test_cases.append(test_case)
        results.append(rag_result)

    save_rag_response_with_context(results)

    return test_cases


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
)
def evaluate_case(test_case):
    return evaluate(
            test_cases=[test_case],
            metrics=[
                contextual_relevancy,
                contextual_precision,
                contextual_recall,
                answer_relevancy,
                faithfulness
            ],
            async_config=AsyncConfig(
                run_async=False,
                throttle_value=3,
                max_concurrent=3,
            )
        )

def main(force: bool = False):
    dataset_path = "tests/test_dataset.json"
    if not force and os.path.exists("rag_ollama_model_answers.json"):
        print("Результаты уже существуют. Используем их.")
        test_cases_answers = load_local_dataset("rag_ollama_model_answers.json")
        dataset = load_local_dataset(dataset_path)
        test_cases = []
        for test, question in zip(test_cases_answers, dataset):
            test_case = LLMTestCase(
                input=question['query'],
                actual_output=test['answer'],
                expected_output=question.get('ground_truth', ''),
                retrieval_context=test['retrieval_context']
            )
            test_cases.append(test_case)
    else:
        print("1. Загрузка датасета и создание тест-кейсов...")
        test_cases = create_test_cases(dataset_path)

    if not test_cases:
        print("Не удалось создать тест-кейсы.")
        return

    print(f"2. Запуск оценки {len(test_cases)} тест-кейсов через Ollama...")
    
    results = []
    for test_case in tqdm(test_cases):
        result = evaluate_case(test_case)
        dump = result.model_dump()
        results.append(dump)
        print(dump)

    with open("rag_ollama_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("3. Детали сохранены в 'rag_ollama_evaluation.json'")

if __name__ == "__main__":

    main()
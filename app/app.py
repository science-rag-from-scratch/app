import asyncio
from datetime import datetime
import json
import os
import time

import dotenv
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from langchain_ollama import ChatOllama
import psycopg2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

dotenv.load_dotenv()
# ------------------------
# Config
# ------------------------
DB_CONN = (
    f"dbname={os.environ['POSTGRES_DB']} "
    f"user={os.environ['POSTGRES_USER']} "
    f"password={os.environ['POSTGRES_PASSWORD']} "
    f"host={os.environ['POSTGRES_HOST']}"
)
CHAINLIT_CONN = f"postgresql+asyncpg://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
EMB_MODEL = "intfloat/multilingual-e5-large"

TOP_K = 9

inference_semaphore = asyncio.Semaphore(1)

# ------------------------
# Models
# ------------------------
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)
emb_model = AutoModel.from_pretrained(EMB_MODEL).to(device).eval()


llm_model = ChatOllama(
    model="llama4:latest",
    base_url=os.environ["OLLAMA_BASE_URL"],
)

# ------------------------
# DB
# ------------------------
def get_db_connection():
    return psycopg2.connect(DB_CONN)

# ------------------------
# Embedding helpers
# ------------------------
MAX_LENGTH = 512

# ------------------------
# app
# ------------------------

# app.router.routes.insert(0, Mount("/docs", app=StaticFiles(directory=DOCS), name="docs"))


def average_pool(last_hidden_states, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
    sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def embed(text: str):
    inputs = emb_tokenizer("query: "+text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
        emb = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        emb = F.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy()

# ------------------------
# DB search
# ------------------------
def search_context(query, top_k=TOP_K):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
    # Embeddings
        query_emb = embed(query).tolist()
        cur.execute(
            """
            SELECT d.arxiv_id, d.content, 
                   jsonb_build_object(
                       'paper_name', dm.paper_name,
                       'pub_year', dm.pub_year,
                       'main_category', dm.main_category,
                       'subcategory', dm.subcategory
                   ) as metadata,
                   1 - (d.embedding <-> %s) AS vec_score
            FROM documents d
            LEFT JOIN documents_meta dm ON d.arxiv_id = dm.arxiv_id
            ORDER BY d.embedding <-> %s
            LIMIT %s
            """,
            (json.dumps(query_emb), json.dumps(query_emb), top_k*2)
        )
        vec_results = cur.fetchall()

        # BM25
        cur.execute(
            """
            SELECT d.arxiv_id, d.content,
                   jsonb_build_object(
                       'paper_name', dm.paper_name,
                       'pub_year', dm.pub_year,
                       'main_category', dm.main_category,
                       'subcategory', dm.subcategory
                   ) as metadata,
                   ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', %s)) AS bm25_score
            FROM documents d
            LEFT JOIN documents_meta dm ON d.arxiv_id = dm.arxiv_id
            WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', %s)
            ORDER BY bm25_score DESC
            LIMIT %s
            """,
            (query, query, top_k*2)
        )
        bm25_results = cur.fetchall()

        # Ranking and concat
        k = 60
        rrf_scores = {}

        for rank, (arxiv_id, content, metadata, _) in enumerate(vec_results):
            key = content
            # Parse metadata if it's a string (JSONB from PostgreSQL)
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            elif metadata is None:
                metadata = {}
            rrf_scores.setdefault(key, {'arxiv_id': arxiv_id, 'metadata': metadata, 'score':0})
            rrf_scores[key]['score'] += 1.0 / (k + rank + 1)

        for rank, (arxiv_id, content, metadata, _) in enumerate(bm25_results):
            key = content
            # Parse metadata if it's a string (JSONB from PostgreSQL)
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            elif metadata is None:
                metadata = {}
            if key not in rrf_scores:
                rrf_scores[key] = {'arxiv_id': arxiv_id, 'metadata': metadata, 'score':0}
            rrf_scores[key]['score'] += 1.0 / (k + rank + 1)

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True)

        return [(item['arxiv_id'], content, item['metadata']) for content, item in sorted_items][:top_k]
    
    finally:
        cur.close()
        conn.close()

# ------------------------
# LLM helpers
# ------------------------
def ask_llm(question, context, chat_history):
    system_prompt = """You are an assistant for searching in documents. Answer ONLY based on the provided context.

INSTRUCTIONS:
1. Answer ONLY based on the context below
2. If information is not in the context - say "There is no information on this question in the provided documents"
3. Be accurate
4. Do not make up information
5. If needed, specify which document you are using    
"""

    prompt = f"""DOCUMENT CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: 
{question}

ANSWER:"""

    messages = [{"role":"system", "content":system_prompt},{"role":"user", "content":prompt}]
    
    response = llm_model.invoke(messages)
    
    return response.content

def rephrase_question(question, history) -> str:
    history_text = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])

    system_prompt = """You are a helper for rephrasing search queries. 
Rephrase the user's last question taking into account the dialogue context, 
but DO NOT include information from previous answers if you could not find an answer to the question in the new search query."""

    prompt = f"""Chat history (for context only):
{history_text}

Current question: 
{question}

Rephrase the current question as a self-contained search query, 
preserving its original meaning. Do not mention previous irrelevant answers.

Rephrased question:
"""
    messages = [{"role":"system", "content":system_prompt},{"role":"user", "content":prompt}]

    output = llm_model.invoke(
        messages,
    )

    return output.content

# ------------------------
# Chat history
# ------------------------
def save_chat_history(user_id, arxiv_id, user_msg, rephrased_msg, assistant_msg, timestamp, sources_ids, chunks):
    conn = get_db_connection()
    cur = conn.cursor()
    sources_ids = json.dumps(sources_ids)
    try:
        cur.execute(
            """
            INSERT INTO chat_history (user_id, doc_id, user_message, rephrased_message, assistant_message, timestamp, sources_ids, chunks)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (user_id, arxiv_id, user_msg, rephrased_msg, assistant_msg, timestamp, sources_ids, chunks)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def load_chat_history(thread_id: str, max_pairs: int = 20):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT type, output FROM steps WHERE "threadId" = %s AND type IN ('user_message', 'assistant_message') AND output is NOT NULL ORDER BY "createdAt" ASC
            """, (thread_id,))

        rows = cur.fetchall()
        chat_history = []
        user_msg = None

        ignore_prefixes = ("⌛", "♻️", "🔍", "✍️", "❌")

        for msg_type, content in rows:
            if msg_type == 'user_message':
                user_msg = content
            elif msg_type == 'assistant_message' and user_msg and not any(content.startswith(p) for p in ignore_prefixes):
                chat_history.append({"user": user_msg, "assistant": content.split("⌛")[0][:-2:]})
                user_msg = None
        
        return chat_history[-max_pairs:]

    finally:
        cur.close()
        conn.close()


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=CHAINLIT_CONN)


def get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

@cl.on_chat_start
async def start_chat():
    torch.cuda.empty_cache()


@cl.password_auth_callback
async def on_login(username: str, password: str) -> cl.User | None:
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor()
        cur.execute("SELECT id, identifier, metadata FROM users WHERE metadata->>'username' = %s;", (username,))
        row = cur.fetchone()
        if row:
            user_id, identifier, metadata = row
            if metadata.get("password") == password:
                return cl.User(identifier=identifier, display_name=metadata.get("display_name"), metadata={"username":metadata.get("username"), "password":metadata.get("password"), "access":metadata.get("access"), "display_name":metadata.get("display_name")})
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
    return None

# ------------------------
# Chainlit core
# ------------------------
async def run_with_dots(
    message: cl.Message,
    base_text: str,
    task: asyncio.Task,
    dots_interval: float=0.6,
    max_dots: int=3):
    dots = ""
    while not task.done():
        dots = "." * (len(dots)%max_dots+1)
        message.content = f"{base_text}{dots}"
        await message.update()
        await asyncio.sleep(dots_interval)
    return await task


@cl.on_message
async def on_message(message: cl.Message):
    start_time = time.time()

    msg = await cl.Message(content="⌛ Ваш запрос в очереди на исполнение. Пожалуйста, подождите...", author="Assistant").send()
    thread_id = cl.context.session.thread_id
    chat_history = load_chat_history(thread_id, max_pairs = 20)
    print(chat_history)
    loop = asyncio.get_event_loop()

    try:
        # Рефразирование вопроса с учетом истории
        old_q = message.content
        async with inference_semaphore:
            if chat_history:
                rephrase_task = loop.run_in_executor(None, rephrase_question, message.content, chat_history)
                new_question = await run_with_dots(msg, "♻️ Формирую запрос", rephrase_task)
                rephrased_q = new_question
            else:
                msg.content = "⌛ Ваш запрос в очереди на исполнение. Пожалуйста, подождите..."
                await msg.update()
                new_question = message.content
                rephrased_q = None

        # Поиск релевантного контекста
        async with inference_semaphore:
            search_task = loop.run_in_executor(None, search_context, new_question)
            context_chunks = await run_with_dots(msg, "🔍 Ищу релевантные документы", search_task)
        if not context_chunks:
            msg.content = "❌ Информация в документах не найдена."
            await msg.update()

        # Достаём текст чанков
        context = "\n\n".join([c[1] for c in context_chunks])
        doc_id = context_chunks[0][0]
        sources = [c[0] for c in context_chunks]
        
        # Генерация ответа
        async with inference_semaphore:
            llm_task = loop.run_in_executor(None, ask_llm, message.content, context, chat_history)
            answer = await run_with_dots(msg, "✍️ Генерирую ответ", llm_task)
        
        data_layer = get_data_layer()
        
        paths = []
        files = []
        # Предполагаем, что 'sources' — это список arxiv_id
        for arxiv_id in sources:
            if arxiv_id not in paths:
                try:
                    display_name = arxiv_id
                    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                    files.append(f"🔗 [arXiv:{display_name}]({arxiv_url})")
                    paths.append(arxiv_id)
                except Exception as e:
                    print(f"❌ Не удалось сформировать ссылку на arXiv для {arxiv_id}: {e}")

        # Отправка ответа
        sources_text = "\n".join(f"- {link}" for link in files)
        end_time = time.time()
        msg.content=f"{answer}\n\n⌛ Время исполнения запроса: {round(end_time-start_time, 1)} секунд.\n\n📁 Источники:\n{sources_text}"
        await msg.update()

        timestamp = datetime.now()
        
        current_user = cl.user_session.get("user")
        user_id = current_user.identifier

        context = "\n-----------------------------------------------------------\n".join([c[1] for c in context_chunks])
        save_chat_history(user_id, doc_id, old_q, rephrased_q, answer, timestamp, sources, context)

    except Exception as e:
        msg = cl.Message(content="♻️ Обрабатываю запрос...", author="Assistant")
        msg.content=f"☠️ Произошла ошибка: {str(e)}"
        await msg.send()
        
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    pass

@cl.on_chat_end
def on_chat_end():
    torch.cuda.empty_cache()


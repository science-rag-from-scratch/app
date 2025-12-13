import json
import psycopg2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
from typing import Optional


# ------------------------
# Config
# ------------------------
DB_CONN = "dbname=postgres user=postgres password=postgres host=77.234.216.102"
EMB_MODEL_PATH = "/models/multilingual-e5-large-instruct/snapshots/274baa43b0e13e37fafa6428dbc7938e62e5c439"
LLM_MODEL_PATH = "/models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
TOP_K = 9

# ------------------------
# Models
# ------------------------
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_PATH)
emb_model = AutoModel.from_pretrained(EMB_MODEL_PATH).to(device).eval()

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    device_map=device,
    trust_remote_code=True,
    quantization_config=quantization_config,
    #dtype = torch.float16
).eval()
llm_model = torch.compile(llm_model)

# ------------------------
# DB
# ------------------------
def get_db_connection():
    return psycopg2.connect(DB_CONN)

# ------------------------
# Embedding helpers
# ------------------------
MAX_LENGTH = 512

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
    
    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)
    
    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.inference_mode():
        output = llm_model.generate(
            **input_ids,
            max_new_tokens=1024,
            do_sample=False,
            eos_token_id=terminators,
            num_beams=1
        )
        
    generated_tokens = output[0][input_ids["input_ids"].shape[-1]:]    
    answer = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer

def rephrase_question(question, history):
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

    input_ids = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)
    
    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.inference_mode():
        output = llm_model.generate(
            **input_ids,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=terminators,
            num_beams=1
        )
    
    generated_tokens = output[0][input_ids["input_ids"].shape[-1]:]    
    return llm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# ------------------------
# Chat history
# ------------------------
def save_chat_history(user_id, arxiv_id, user_msg, rephrased_msg, assistant_msg, timestamp, sources_ids, chunks):
    conn = get_db_connection()
    cur = conn.cursor()
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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
import logging
from dotenv import load_dotenv
import re

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ---- CONFIG ----
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama3-70b-8192"
TOP_K = 8

# ---- FASTAPI SETUP ----
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
@app.get("/")
async def root():
    return {"message": "FastAPI backend is running."}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class AskRequest(BaseModel):
    question: str

# ---- LOAD DATA ----
df = pd.read_csv("data/cleaned_news_data.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
logger.info(f"Loaded dataframe: {df.shape[0]} rows, {df.shape[1]} columns.")

embedder = SentenceTransformer(EMBED_MODEL)
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection("csv_rows")

if collection.count() == 0:
    logger.info("Building row embeddings for vector DB...")
    texts = [
        f"Row {idx}: " + " | ".join([f"{col}: {str(row[col])}" for col in df.columns])
        for idx, row in df.iterrows()
    ]
    vectors = embedder.encode(texts, show_progress_bar=True)
    ids = [str(i) for i in range(len(df))]
    collection.add(documents=texts, embeddings=vectors, ids=ids)
    logger.info("Row embeddings indexed.")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("Missing GROQ_API_KEY")
llm = ChatGroq(
    api_key=groq_api_key,
    model_name=GROQ_MODEL,
    temperature=0.1,
    max_tokens=1024
)

def prompt_llm_for_code(question, schema):
    column_descriptions = """
    - link: URL to the news article.
    - headline: Title of the news article.
    - category: News category or section (e.g., U.S. NEWS, COMEDY, PARENTING).
    - short_description: Brief summary or excerpt of the article content.
    - authors: Author(s) of the article.
    - date: Publication date of the article (pandas.Timestamp). Use pd.to_datetime() on string literals for filtering or comparison.
    """

    return (
        f"You are a pandas expert. "
        f"The DataFrame is called `df` and has columns: {schema}.\n"
        f"Column descriptions:\n{column_descriptions}\n"
        f"When selecting multiple columns, always use a list (not tuple): df[['col1','col2']], never df['col1','col2'].\n"
        f"Assign the final output to a variable called result.\n"
        f"When filtering or comparing dates, ALWAYS use pd.to_datetime() on any string literal, especially on the 'date' column.\n"
        f"For any string/categorical comparison (such as 'category' or 'authors'), always use .str.lower().str.strip() for the column and compare to the lowercased/stripped value, e.g.: df['category'].str.lower().str.strip() == 'u.s. news'.\n"
        f"Treat the 'link' column as a string URL; filtering by it is uncommon.\n"
        f"Write Python pandas code to answer: '{question}'. "
        f"Return ONLY the code, no explanation."
    )



def get_pandas_code_from_llm(question):
    schema = ", ".join(df.columns)
    prompt = prompt_llm_for_code(question, schema)
    result = llm.invoke([HumanMessage(content=prompt)])
    code = extract_code_from_response(result.content)
    return code

def extract_code_from_response(text):
    # Remove triple backticks if present
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text.strip()

def run_pandas_code(code, df):

    local_vars = {"df": df, "pd": pd}
    try:
        exec(code, {}, local_vars)
        if "result" in local_vars:
            res = local_vars["result"]
            # Pretty print DataFrame/Series
            if isinstance(res, pd.DataFrame):
                return res.to_markdown(index=False)
            elif isinstance(res, pd.Series):
                return res.to_string()
            else:
                return str(res)
        else:
            return "Executed code, but no 'result' variable found."
    except Exception as e:
        return f"Error running code: {e}"

def answer_with_pandas(question: str, df: pd.DataFrame):
    q = question.lower()

    # 1. Direct ID/entity lookup (try to detect any "id"-like column and value)
    id_columns = [col for col in df.columns if 'id' in col.lower()]
    for id_col in id_columns:
        id_match = re.search(r"\b([a-z]\d{1,5})\b", q)
        if id_match:
            target_id = id_match.group(1).upper()
            record = df[df[id_col].astype(str).str.upper() == target_id]
            if not record.empty:
                info = record.to_dict(orient="records")
                return {
                    "explanation": f'Found data for {id_col} == {target_id}.',
                    "short_answer": info
                }
            else:
                return {
                    "explanation": f'No record found for {id_col} == {target_id}.',
                    "short_answer": "Not enough information."
                }

    # 2. How many unique values of a categorical column (count, value_counts)
    match = re.search(r"how many (.+?)(?:s)? (are there|do we have|do i have)?", q)
    if match:
        group = match.group(1).strip()
        for col in df.columns:
            # If column directly matches query group
            if group == col.lower():
                # Is it categorical? Count unique values.
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    count = df[col].nunique()
                    values = df[col].value_counts()
                    details = "; ".join([f"{v} ({c})" for v, c in values.items()])
                    return {
                        "explanation": f"Column '{col}' has {count} unique values.",
                        "short_answer": details
                    }
                else:
                    count = df[col].count()
                    return {
                        "explanation": f"Column '{col}' is numeric with {count} records.",
                        "short_answer": str(count)
                    }
        # If group is a value in any column, count that value
        for col in df.columns:
            matches = df[col].astype(str).str.lower() == group
            if matches.any():
                count = matches.sum()
                return {
                    "explanation": f"Value '{group}' found in column '{col}' {count} times.",
                    "short_answer": str(count)
                }
    # 3. Age or numeric filtering (below/above/between)
    for col in df.select_dtypes(include='number').columns:
        # Below/under/less than
        below = re.search(rf"(?:below|under|less than) (?:the )?{col}(?: of)? (\d+)", q)
        if below:
            val = float(below.group(1))
            count = df[df[col] < val].shape[0]
            return {
                "explanation": f'Counted rows with {col} < {val}.',
                "short_answer": str(count)
            }
        # Above/over/greater than
        above = re.search(rf"(?:above|over|greater than) (?:the )?{col}(?: of)? (\d+)", q)
        if above:
            val = float(above.group(1))
            count = df[df[col] > val].shape[0]
            return {
                "explanation": f'Counted rows with {col} > {val}.',
                "short_answer": str(count)
            }
        # Between
        between = re.search(rf"(?:between|from) (?:the )?{col}(?: of)? (\d+) (?:and|to) (\d+)", q)
        if between:
            low = float(between.group(1))
            high = float(between.group(2))
            count = df[(df[col] >= low) & (df[col] <= high)].shape[0]
            return {
                "explanation": f'Counted rows with {low} <= {col} <= {high}.',
                "short_answer": str(count)
            }
    # 4. Show columns
    if "column" in q and ("what" in q or "list" in q or "show" in q):
        return {
            "explanation": "All columns in the data.",
            "short_answer": ", ".join(df.columns)
        }
    # 5. Generic value_counts for any column
    match = re.search(r"(?:distribution|breakdown|counts|value_counts) of ([a-z_ ]+)", q)
    if match:
        col = match.group(1).strip()
        if col in df.columns:
            vc = df[col].value_counts()
            detail = "; ".join([f"{v} ({c})" for v, c in vc.items()])
            return {
                "explanation": f'Distribution of values for column {col}.',
                "short_answer": detail
            }
    return None

def get_basic_stats(df):
    stats = []
    # Existing stats...

    # Add authors per category count
    authors_per_category = df.groupby('category')['authors'].nunique().reset_index()
    for _, row in authors_per_category.iterrows():
        stats.append({
            "column": "authors_per_category",
            "category": row['category'],
            "unique_authors": row['authors']
        })
    # or prepare it in a nicer format and pass it to the LLM prompt

    return stats

def prompt_llm_for_news_summary(question, stats_text):
    prompt = (
        f"You are a data scientist and news expert. Here are summary statistics for news articles:\n"
        f"{stats_text}\n\n"
        f"Based on the above, answer this user question in 2-4 clear sentences, "
        f"and provide an actionable insight or recommendation if relevant. "
        f"The question is:\n{question}"
    )
    return prompt



def get_relevant_rows(question: str, k=TOP_K):
    q_vec = embedder.encode([question])[0]
    results = collection.query(query_embeddings=[q_vec], n_results=k)
    indices = [int(idx) for idx in results["ids"][0]]
    rows = df.iloc[indices]
    docs = results["documents"][0]
    return rows, docs

@app.post("/ask", response_model=dict)
async def ask(data: AskRequest):
    question = data.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
    logger.info("User question: %s", question)

    q_lower = question.lower()

    # Intent classification (simple keyword matching)
    if any(word in q_lower for word in ['summary', 'summarize', 'insight', 'overview', 'analysis']):
        stats_text = get_basic_stats(df)
        prompt = prompt_llm_for_news_summary(question, stats_text)
        result = llm.invoke([HumanMessage(content=prompt)])
        answer = getattr(result, "content", str(result))
        return {"answer": answer.strip()}

    elif any(word in q_lower for word in ['count', 'number', 'how many', 'distribution', 'value_counts']):
        # Use pandas codegen for numeric/stat queries
        code = get_pandas_code_from_llm(question)
        if not any(line.strip().startswith("result") for line in code.splitlines()):
            code = f"result = {code.strip()}"
        result = run_pandas_code(code, df)
        return {"answer": str(result)}

    else:
        # General fallback: semantic search + context + LLM response
        rows, docs = get_relevant_rows(question)
        context = "\n".join(docs)
        prompt = f"Answer the question based on the following news data:\n{context}\nQuestion: {question}"
        result = llm.invoke([HumanMessage(content=prompt)])
        answer = getattr(result, "content", str(result))
        return {"answer": answer.strip()}



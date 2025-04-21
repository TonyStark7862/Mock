
import streamlit as st
import os
import io
import PyPDF2  # PDF processing
import pandas as pd  # For CSV handling
import numpy as np  # For user's functions (column_des)
import random  # For user's functions (column_des)
import sqlalchemy # For SQL DB connection
from pathlib import Path
import re
import csv
import logging
import datetime
import traceback
from typing import Any, Sequence, List, Generator, AsyncGenerator, Union, Dict, Tuple, Optional, Callable

# LlamaIndex Core Imports
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings,
    QueryBundle,
    PromptTemplate,
)
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import BaseQueryEngine, SubQuestionQueryEngine # Using SubQuestionQueryEngine
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
    ResponseMode
)
from llama_index.core.callbacks import CallbackManager # Import CallbackManager

# LlamaIndex LLM Imports
from llama_index.core.llms import (
    LLM,
    CompletionResponse,
    ChatResponse,
    ChatMessage,
    MessageRole,
    LLMMetadata,
)
# LlamaIndex Response Schema
from llama_index.core.base.response.schema import Response # Import Response

# LlamaIndex Embeddings & Vector Stores
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus

# --- Configuration ---
QDRANT_PERSIST_DIR = "./qdrant_storage_unified_subq_final"
QDRANT_PDF_COLLECTION_PREFIX = "unified_subq_pdf_coll_" # Prefix for unique PDF collections
SQL_DB_DIR = "./sql_database_unified_subq_final"
SQL_DB_FILENAME = "csv_data_unified_subq_final.db"
SQL_DB_PATH = os.path.join(SQL_DB_DIR, SQL_DB_FILENAME)
SQL_DB_URL = f"sqlite:///{SQL_DB_PATH}"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384
LLM_MODEL_NAME = "custom_abc_llm" # Placeholder name

# Create directories if they don't exist
os.makedirs(QDRANT_PERSIST_DIR, exist_ok=True)
os.makedirs(SQL_DB_DIR, exist_ok=True)

# Setup Logging
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# --- Helper Functions (CSV Description & Sanitization) ---
# ==============================================================================
def got_type(list_):
    # (Code from previous version)
    def judge(string):
        s_val = str(string) if string is not None else "";
        if not s_val: return "string"
        try: int(s_val); return "int"
        except ValueError:
            try: float(s_val); return "float"
            except ValueError: return "string"
    return [judge(str(x) if x is not None else "") for x in list_]

def column_des(df):
    # (Code from previous version, including error handling per column)
    def single_des(name,data):
        description = "{\"Column Name\": \"" + name + "\"" + ", "; valid_data = data.dropna().tolist();
        if not valid_data: return ""
        pre_len = len(data); post_len = len(valid_data); types = got_type(valid_data)
        if "string" in types: type_ = "string"; data_proc = [str(x) for x in valid_data]
        elif "float" in types: type_ = "float"; data_proc = np.array([float(x) for x in valid_data])
        else: type_ = "int"; data_proc = np.array([int(x) for x in valid_data])
        description += "\"Type\": \"" + type_ + "\", "
        if type_ in ["int", "float"]:
            if data_proc.size > 0:
                 min_ = data_proc.min(); max_ = data_proc.max();
                 description += "\"MIN\": " + str(min_) + ", \"MAX\": " + str(max_)
            else: description += "\"MIN\": null, \"MAX\": null"
        elif type_ == "string":
            values = list(set(["\"" + str(x).strip().replace('"',"'") + "\"" for x in data_proc]))
            random.shuffle(values);
            if len(values) > 15: values = values[:random.randint(5, 10)]
            numerates = ", ".join(values); description += "\"Sample Values\": [" + numerates + "]"
        description += ", \"Contains NaN\": " + str(post_len != pre_len); return description + "}"
    columns_dec = []
    for c in df.columns:
        try:
            desc = single_des(c, df[c])
            if desc: columns_dec.append(desc)
        except Exception as e:
            logger.warning(f"Could not generate description for column '{c}': {e}")
            columns_dec.append("{\"Column Name\": \"" + str(c) + "\", \"Error\": \"Could not generate description\"}")
    random.shuffle(columns_dec)
    return "\n".join(columns_dec)

def generate_table_description(df: pd.DataFrame, table_name: str, source_csv_name: str) -> str:
    # (Code from previous version)
    try:
        rows_count, columns_count = df.shape
        description = f"Table Name: '{table_name}' (derived from CSV: '{source_csv_name}')\n"
        description += f"Contains {rows_count} rows and {columns_count} columns.\n"
        description += f"SQL Table Columns: {', '.join(df.columns)}\n"
        description += f"--- Column Details and Sample Data ---\n"
        col_descriptions = column_des(df)
        description += col_descriptions if col_descriptions else "No detailed column descriptions generated."
        return description
    except Exception as e:
        logger.error(f"Failed to generate table description for {table_name} from {source_csv_name}: {e}", exc_info=True)
        return f"Error generating description for table '{table_name}'. Columns: {', '.join(df.columns)}. Error: {e}"

def sanitize_for_name(filename: str, max_len: int = 40) -> str:
    # (Code from previous version)
    name = Path(filename).stem
    name = re.sub(r'\W+', '_', name)
    name = name.strip('_')
    if name and name[0].isdigit(): name = '_' + name
    name = name[:max_len].lower()
    if not name or name in ["_", "__"]: name = f"file_{random.randint(1000, 9999)}"
    return name

# ==============================================================================
# --- Custom LLM Implementation (Placeholder) ---
# ==============================================================================
def abc_response(prompt: str) -> str:
    # (Code identical to previous version)
    logger.info(f"MyCustomLLM received prompt (first 100 chars): {prompt[:100]}...")
    response = f"This is a dummy response from MyCustomLLM for the prompt starting with: {prompt[:50]}..."
    logger.info(f"MyCustomLLM generated response (first 100 chars): {response[:100]}...")
    return response

class MyCustomLLM(LLM):
    # (Code identical to previous version)
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        logger.info("MyCustomLLM: chat() called")
        prompt = "\n".join([f"{m.role.value}: {m.content}" for m in messages])
        response_text = abc_response(prompt)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=response_text))
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        logger.info("MyCustomLLM: complete() called")
        response_text = abc_response(prompt)
        return CompletionResponse(text=response_text)
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        logger.info("MyCustomLLM: achat() called - calling sync chat()")
        return self.chat(messages, **kwargs)
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        logger.info("MyCustomLLM: acomplete() called - calling sync complete()")
        return self.complete(prompt, **kwargs)
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        logger.warning("MyCustomLLM: stream_chat() called - Returning single response")
        yield self.chat(messages, **kwargs)
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        logger.warning("MyCustomLLM: stream_complete() called - Returning single response")
        yield self.complete(prompt, **kwargs)
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        logger.warning("MyCustomLLM: astream_chat() called - NotImplementedError")
        raise NotImplementedError("Async streaming chat not supported by MyCustomLLM.")
        yield
    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        logger.warning("MyCustomLLM: astream_complete() called - NotImplementedError")
        raise NotImplementedError("Async streaming complete not supported by MyCustomLLM.")
        yield
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=LLM_MODEL_NAME, is_chat_model=True)

# ==============================================================================
# --- Custom SQL Engine (Includes full prompts) ---
# ==============================================================================
class CustomSQLEngine:
    # Includes full SQL prompts now
    def __init__(
        self, sql_engine: sqlalchemy.engine.Engine, table_name: str,
        llm_callback: Callable[[str], str], table_description: Optional[str] = None,
        verbose: bool = True ):
        self.sql_engine = sql_engine
        self.table_name = table_name
        self.llm_callback = llm_callback
        self.table_description = table_description or ""
        self.verbose = verbose
        # --- START FULL PROMPTS ---
        self.sql_prompt_template = """You are an expert SQL query generator for SQLite.

Based on the user's natural language query: "{query_str}"

Generate a SQL query for the following SQL table:
Table Name: {table_name}

{table_description}

IMPORTANT GUIDELINES FOR SQLITE:
1. Use double quotes for table and column names if they contain spaces or special characters (usually avoided by cleaning). Stick to standard SQL syntax otherwise. Column names provided are already cleaned.
2. For string comparisons, use LIKE with % wildcards for partial matches (e.g., `column LIKE '%value%'`). String comparisons are case-sensitive by default unless specified otherwise.
3. For aggregations (SUM, COUNT, AVG, MIN, MAX), always use GROUP BY for any non-aggregated columns included in the SELECT list.
4. Handle NULL values explicitly using `IS NULL` or `IS NOT NULL`. Use `COALESCE(column, default_value)` to provide defaults if needed.
5. Avoid complex joins if possible, as this engine primarily works with single tables derived from CSVs.
6. For date/time operations, use standard SQLite functions like DATE(), TIME(), DATETIME(), STRFTIME(). Assume date columns are stored in a format SQLite understands (like YYYY-MM-DD HH:MM:SS).
7. Keep queries simple and direct. Only select columns needed to answer the query. Do not add explanations.

Return ONLY the executable SQL query without any explanation, markdown formatting, or comments.
Do not enclose the query in backticks or add the word 'sql'.
Example: SELECT COUNT(*) FROM "{table_name}" WHERE "Some Column" > 10;
"""
        self.sql_fix_prompt = """The following SQL query for SQLite failed:
```sql
{failed_sql}
```
Error message: {error_msg}

Table information:
Table Name: {table_name}
{table_description}

Please fix the SQL query to work with SQLite following these guidelines:
Fix syntax errors (quoting, commas, keywords). Ensure column names match exactly those provided in the description (case-sensitive).
Replace incompatible functions with SQLite equivalents.
Simplify the query logic if it seems overly complex or likely caused the error. Check aggregations and GROUP BY clauses.
Ensure data types in comparisons are appropriate (e.g., don't compare text to numbers directly without casting if necessary).
Double-check table name correctness: {table_name}

Return ONLY the corrected executable SQL query without any explanation or formatting.
"""
        # --- END FULL PROMPTS ---

    def _get_schema_info(self) -> str:
        # (Code identical to previous version)
        try:
            metadata = sqlalchemy.MetaData()
            inspector = sqlalchemy.inspect(self.sql_engine)
            if not inspector.has_table(self.table_name): return f"Error: Table '{self.table_name}' does not exist."
            metadata.reflect(bind=self.sql_engine, only=[self.table_name])
            table = metadata.tables.get(self.table_name)
            if not table: return f"Error: Could not find table '{self.table_name}' after reflection."
            columns = []
            for column in table.columns:
                col_type = str(column.type)
                constraints = []
                if column.primary_key: constraints.append("PRIMARY KEY")
                if not column.nullable: constraints.append("NOT NULL")
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                columns.append(f"  - \"{column.name}\": {col_type}{constraint_str}")
            return f"Actual Schema for table \"{self.table_name}\":\nColumns:\n" + "\n".join(columns)
        except Exception as e:
            logger.error(f"Error getting schema info for {self.table_name}: {e}", exc_info=True)
            return f"Error retrieving schema for table {self.table_name}: {e}"

    def _clean_sql(self, sql: str) -> str:
        # (Code identical to previous version)
        sql = re.sub(r'```sql|```', '', sql)
        sql = re.sub(r'^sql\s+', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        return sql.strip()

    def _is_safe_sql(self, sql: str) -> bool:
        # (Code identical to previous version)
        lower_sql = sql.lower().strip()
        if not lower_sql.startswith('select') and not lower_sql.startswith('with'): return False
        dangerous_keywords = [r'\bdrop\b', r'\bdelete\b', r'\btruncate\b', r'\bupdate\b', r'\binsert\b', r'\balter\b', r'\bcreate\b', r'\breplace\b', r'\bgrant\b', r'\brevoke\b', r'\battach\b', r'\bdetach\b']
        for pattern in dangerous_keywords:
            if re.search(pattern, lower_sql): return False
        return True

    def _format_results(self, result_df: pd.DataFrame) -> str:
         # (Code identical to previous version)
        if result_df.empty: return "The query returned no results."
        max_rows_to_show, max_cols_to_show = 20, 15
        original_shape = result_df.shape
        df_to_show = result_df
        show_cols_truncated = original_shape[1] > max_cols_to_show
        show_rows_truncated = original_shape[0] > max_rows_to_show
        if show_cols_truncated: df_to_show = df_to_show.iloc[:, :max_cols_to_show]
        if show_rows_truncated: df_to_show = df_to_show.head(max_rows_to_show)
        result_str = ""
        if show_rows_truncated or show_cols_truncated:
             result_str += f"Query returned {original_shape[0]} rows, {original_shape[1]} columns. "
             parts = []
             if show_rows_truncated: parts.append(f"first {max_rows_to_show} rows")
             if show_cols_truncated: parts.append(f"first {max_cols_to_show} columns")
             result_str += f"Showing { ' and '.join(parts)}:\n\n"
        else: result_str += "Query Result:\n\n"
        try: markdown_result = df_to_show.to_markdown(index=False)
        except Exception: markdown_result = df_to_show.to_string(index=False)
        return result_str + markdown_result

    def _execute_sql(self, sql: str) -> Tuple[bool, Union[pd.DataFrame, str]]:
        # (Code identical to previous version)
        try:
            if not self._is_safe_sql(sql): return False, "SQL query failed safety check."
            if self.verbose: logger.info(f"Executing safe SQL: {sql}")
            with self.sql_engine.connect() as connection: result_df = pd.read_sql_query(sql, connection)
            return True, result_df
        except sqlalchemy.exc.SQLAlchemyError as db_err: return False, f"Database Error: {db_err}"
        except Exception as e: return False, f"General Error executing SQL: {e}"

    def _execute_with_retry(self, sql: str, max_retries: int = 1) -> Tuple[bool, Union[pd.DataFrame, str], str]:
        # (Code identical to previous version)
        current_sql, original_sql = sql, sql
        for attempt in range(max_retries + 1):
            if self.verbose: logger.info(f"SQL Exec Attempt {attempt+1}/{max_retries+1}")
            success, result = self._execute_sql(current_sql)
            if success:
                if self.verbose: logger.info("SQL success")
                return True, result, current_sql # result is DataFrame
            error_message = str(result)
            logger.warning(f"SQL failed: {error_message}")
            if attempt == max_retries: return False, f"SQL failed: {error_message}", current_sql
            if self.verbose: logger.info("Attempting SQL fix...")
            try:
                schema_info = self._get_schema_info()
                fix_prompt = self.sql_fix_prompt.format( failed_sql=current_sql, error_msg=error_message, table_name=self.table_name, table_description=f"{self.table_description}\n\n{schema_info}")
                fixed_sql_response = self.llm_callback(fix_prompt)
                fixed_sql = self._clean_sql(fixed_sql_response)
                if fixed_sql and fixed_sql.lower() != current_sql.lower():
                    current_sql = fixed_sql
                    if self.verbose: logger.info(f"LLM proposed fix: {current_sql}")
                else:
                    if self.verbose: logger.warning("LLM fix failed.")
                    return False, f"SQL failed: {error_message} (LLM fix failed)", original_sql
            except Exception as fix_error:
                logger.error(f"LLM fix error: {fix_error}", exc_info=True)
                return False, f"SQL failed: {error_message} (LLM fix error: {fix_error})", original_sql
        return False, "Unexpected SQL retry loop error", original_sql

    def query(self, query_text: str) -> str:
        # (Code identical to previous version)
        if self.verbose: logger.info(f"Custom SQL Engine Query: {query_text} on {self.table_name}")
        try:
            schema_info = self._get_schema_info();
            if "Error:" in schema_info: return f"Schema error: {schema_info}"
            generate_prompt = self.sql_prompt_template.format( query_str=query_text, table_name=self.table_name, table_description=f"{self.table_description}\n\n{schema_info}")
            sql_response = self.llm_callback(generate_prompt)
            if sql_response.startswith("LLM Error:"): return f"LLM error: {sql_response}"
            sql_query = self._clean_sql(sql_response)
            if not sql_query: return "Error: LLM failed to generate SQL."
            if self.verbose: logger.info(f"Generated SQL: {sql_query}")
            success, result, final_sql = self._execute_with_retry(sql_query)
            if not success: return f"Error querying table '{self.table_name}':\n{result}\nSQL attempted:\n```sql\n{final_sql}\n```"
            formatted_results = self._format_results(result)
            return f"Query result from table '{self.table_name}':\n\n{formatted_results}"
        except Exception as e:
            logger.error(f"Unexpected error in CustomSQLEngine query: {e}", exc_info=True)
            return f"Unexpected error querying table '{self.table_name}': {e}"

# ==============================================================================
# --- Custom SQL Engine Wrapper (with fixes, commented super init) ---
# ==============================================================================
class CustomSQLQueryEngineWrapper(BaseQueryEngine):
    """Adapter to make CustomSQLEngine compatible with LlamaIndex tools."""
    def __init__(self, engine: CustomSQLEngine, llm: Optional[LLM] = None):
        self._engine = engine
        self._llm = llm or Settings.llm
        if not self._llm: raise ValueError("LLM must be available via argument or Settings")
        # --- SUPER INIT COMMENTED OUT AS REQUESTED ---
        # super().__init__() # Try simplest init first
        # --- END COMMENT ---

    @property
    def callback_manager(self) -> CallbackManager:
         # Explicitly provide callback_manager property to satisfy interface
         return getattr(Settings, "callback_manager", CallbackManager([]))

    @property
    def llm(self) -> LLM: return self._llm

    @property
    def synthesizer(self) -> BaseSynthesizer:
         return get_response_synthesizer(llm=self._llm, response_mode=ResponseMode.NO_TEXT)

    def _get_prompt_modules(self) -> Dict[str, Any]: return {}

    def _query(self, query_bundle: QueryBundle) -> Response: # Return Response object
        logger.info(f"CustomSQLWrapper: _query for table {self._engine.table_name}")
        query_text = query_bundle.query_str
        result_str = self._engine.query(query_text) # Gets formatted string result
        return Response(response=result_str) # Wrap in Response object

    async def _aquery(self, query_bundle: QueryBundle) -> Response: # Return Response object
        logger.info(f"CustomSQLWrapper: _aquery for table {self._engine.table_name} - using sync")
        import asyncio
        loop = asyncio.get_running_loop()
        response_obj = await loop.run_in_executor(None, self._query, query_bundle)
        return response_obj

# ==============================================================================
# --- Global Settings and Initialization ---
# ==============================================================================
@st.cache_resource
def get_llm():
    # (Code identical to previous version)
    logger.info("Initializing Custom LLM (cached)...")
    return MyCustomLLM()

@st.cache_resource
def get_embed_model():
    # (Code identical to previous version)
    logger.info(f"Initializing Embedding Model (cached): {EMBEDDING_MODEL_NAME}")
    try:
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        test_embedding = embed_model.get_query_embedding("test")
        actual_dim = len(test_embedding)
        if actual_dim != EXPECTED_EMBEDDING_DIM: raise ValueError(f"Embed dim mismatch! Expected {EXPECTED_EMBEDDING_DIM}, Got {actual_dim}.")
        logger.info(f"Embedding model loaded. Dim: {actual_dim}")
        return embed_model
    except Exception as e: logger.error(f"FATAL: Failed embed model init: {e}", exc_info=True); st.error(f"Embedding model failed: {e}"); return None

@st.cache_resource
def get_qdrant_client() -> Optional[qdrant_client.QdrantClient]:
    # (Code identical to previous version)
    logger.info(f"Initializing Qdrant client (cached): {QDRANT_PERSIST_DIR}")
    try:
        client = qdrant_client.QdrantClient(path=QDRANT_PERSIST_DIR)
        client.get_collections(); logger.info("Qdrant client initialized.")
        return client
    except Exception as e: logger.error(f"FATAL: Failed Qdrant client init: {e}", exc_info=True); st.error(f"Qdrant vector DB failed: {e}. PDF analysis unavailable."); return None

def configure_global_settings() -> bool:
    # (Code identical to previous version)
    logger.info("Configuring LlamaIndex Global Settings...")
    try:
        llm = get_llm()
        embed_model = get_embed_model()
        if llm is None or embed_model is None: logger.error("LLM/Embed Model failed init."); return False
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=200)
        Settings.num_output = 512
        Settings.context_window = 4096
        # Settings.callback_manager = CallbackManager([]) # Optional: configure if needed
        logger.info("Global Settings configured.")
        return True
    except Exception as e: logger.error(f"Settings config error: {e}", exc_info=True); st.error(f"Core component config failed: {e}"); return False

# ==============================================================================
# --- Data Processing Functions ---
# ==============================================================================
def process_pdf(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[Document]:
    # (Code identical to previous version)
    if not uploaded_file: return None
    file_name = uploaded_file.name; logger.info(f"Processing PDF: {file_name}")
    try:
        with io.BytesIO(uploaded_file.getvalue()) as pdf_stream:
            reader = PyPDF2.PdfReader(pdf_stream); text_content = ""; num_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text: text_content += page_text + "\n\n"
                    else: logger.warning(f"No text page {page_num+1}/{num_pages} in '{file_name}'.")
                except Exception as page_err: logger.error(f"Page extract error {page_num+1} in '{file_name}': {page_err}", exc_info=True); text_content += f"[Error page {page_num+1}]\n\n"
        if not text_content.strip(): st.warning(f"No text extracted from PDF '{file_name}'."); return None
        doc = Document(text=text_content, metadata={"file_name": file_name})
        logger.info(f"Processed PDF '{file_name}'. Length: {len(text_content)}")
        return doc
    except Exception as e: logger.error(f"PDF process failed '{file_name}': {e}", exc_info=True); st.error(f"Error reading PDF '{file_name}': {e}"); return None

def process_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[pd.DataFrame]:
    # (Code identical to previous version)
    if not uploaded_file: return None
    file_name = uploaded_file.name; logger.info(f"Processing CSV: {file_name}")
    try:
        try: df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        except UnicodeDecodeError: logger.warning(f"UTF-8 fail '{file_name}', trying latin1."); df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), encoding='latin1')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        logger.info(f"Loaded CSV '{file_name}'. Shape: {df.shape}")
        if df.empty: st.warning(f"CSV '{file_name}' is empty."); return None
        return df
    except Exception as e: logger.error(f"CSV process failed '{file_name}': {e}", exc_info=True); st.error(f"Error reading CSV '{file_name}': {e}"); return None

# ==============================================================================
# --- Tool Creation Functions (Unique per file) ---
# ==============================================================================
def create_pdf_tool(
    pdf_document: Document, qdrant_client_instance: qdrant_client.QdrantClient
) -> Optional[QueryEngineTool]:
    # (Code identical to previous version)
    if not pdf_document or not pdf_document.text.strip(): return None
    if not qdrant_client_instance: return None
    file_name = pdf_document.metadata.get("file_name", f"unknown_pdf_{random.randint(1000,9999)}.pdf")
    sanitized_name = sanitize_for_name(file_name)
    collection_name = f"{QDRANT_PDF_COLLECTION_PREFIX}{sanitized_name}"
    tool_name = f"pdf_{sanitized_name}_tool"
    logger.info(f"Creating PDF tool: '{file_name}' (Coll: {collection_name}, Tool: {tool_name})")
    try:
        try: qdrant_client_instance.create_collection( collection_name=collection_name, vectors_config=VectorParams(size=EXPECTED_EMBEDDING_DIM, distance=Distance.COSINE)); logger.info(f"Collection '{collection_name}' created.")
        except Exception as create_exc: logger.warning(f"Create failed '{collection_name}' (may exist): {create_exc}. Checking...");
             try: qdrant_client_instance.get_collection(collection_name=collection_name); logger.info(f"Collection '{collection_name}' exists.")
             except Exception as get_exc: logger.error(f"Failed ensure collection '{collection_name}': {get_exc}", exc_info=True); st.error(f"Qdrant error '{file_name}': {get_exc}"); return None
        vector_store = QdrantVectorStore(client=qdrant_client_instance, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info(f"Indexing doc '{file_name}' into '{collection_name}'...")
        index = VectorStoreIndex.from_documents([pdf_document], storage_context=storage_context, show_progress=True)
        logger.info("Indexing complete.")
        pdf_query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
        logger.info("Query engine created.")
        tool_description = f"Provides info from PDF document '{file_name}'. Use for text search, summarization, content understanding in this specific file."
        pdf_tool = QueryEngineTool(query_engine=pdf_query_engine, metadata=ToolMetadata(name=tool_name, description=tool_description))
        logger.info(f"PDF tool '{tool_name}' created.")
        return pdf_tool
    except Exception as e: logger.error(f"PDF tool failed '{file_name}': {e}", exc_info=True); st.error(f"PDF tool setup error '{file_name}': {e}"); return None

def create_csv_tool(
    df: pd.DataFrame, csv_file_name: str, sql_alchemy_engine: sqlalchemy.engine.Engine
) -> Optional[QueryEngineTool]:
    # (Code identical to previous version)
    if df is None or df.empty: return None
    if not sql_alchemy_engine: return None
    sanitized_base_name = sanitize_for_name(csv_file_name)
    table_name = f"csv_tbl_{sanitized_base_name}"
    tool_name = f"csv_{sanitized_base_name}_tool"
    logger.info(f"Creating CSV tool: '{csv_file_name}' (Table: {table_name}, Tool: {tool_name})")
    try:
        original_columns = df.columns.tolist(); cleaned_column_map={}; seen_cleaned_names=set()
        for i, col in enumerate(df.columns):
            cleaned_col = re.sub(r'\W+|^(?=\d)', '_', str(col)).lower().strip('_') or f"column_{i}"
            final_cleaned_col=cleaned_col; suffix=1
            while final_cleaned_col in seen_cleaned_names: final_cleaned_col=f"{cleaned_col}_{suffix}"; suffix+=1
            seen_cleaned_names.add(final_cleaned_col); cleaned_column_map[col] = final_cleaned_col
        df_renamed=df.rename(columns=cleaned_column_map); cleaned_columns=df_renamed.columns.tolist(); logger.info(f"Cleaned cols: {cleaned_columns}")
        dtype_mapping={col: sqlalchemy.types.TEXT for col in df_renamed.select_dtypes(include=['object', 'string']).columns}
        logger.info(f"Saving DF {df_renamed.shape} to SQL table '{table_name}'..."); df_renamed.to_sql(name=table_name, con=sql_alchemy_engine, index=False, if_exists='replace', chunksize=1000, dtype=dtype_mapping); logger.info("Save complete.")
        logger.info("Generating table description..."); table_desc = generate_table_description(df_renamed, table_name, csv_file_name); logger.info("Description generated.")
        logger.info("Instantiating CustomSQLEngine...")
        def llm_callback(prompt: str) -> str:
            try:
                if not Settings.llm: raise ValueError("Global LLM not configured.")
                return Settings.llm.complete(prompt).text
            except Exception as e: logger.error(f"LLM callback error: {e}", exc_info=True); return f"LLM Error: {e}"
        custom_sql_engine_instance = CustomSQLEngine(sql_engine=sql_alchemy_engine, table_name=table_name, llm_callback=llm_callback, table_description=table_desc, verbose=True)
        wrapped_engine = CustomSQLQueryEngineWrapper(custom_sql_engine_instance); logger.info("Custom engine wrapped.")
        tool_description = f"Queries SQL table '{table_name}' (from CSV '{csv_file_name}'). Use for structured data lookup, filtering, calculations (SUM, COUNT, AVG), aggregation. SQL columns: {', '.join(cleaned_columns)}."
        csv_tool = QueryEngineTool(query_engine=wrapped_engine, metadata=ToolMetadata(name=tool_name, description=tool_description))
        logger.info(f"CSV tool '{tool_name}' created.")
        return csv_tool
    except sqlalchemy.exc.SQLAlchemyError as db_err: logger.error(f"DB error CSV '{csv_file_name}': {db_err}", exc_info=True); st.error(f"DB error CSV '{csv_file_name}': {db_err}"); return None
    except Exception as e: logger.error(f"CSV tool failed '{csv_file_name}': {e}", exc_info=True); st.error(f"CSV tool setup error '{csv_file_name}': {e}"); return None

# ==============================================================================
# --- Main Engine Setup Function (Uses SubQuestionQueryEngine) ---
# ==============================================================================
def setup_engine(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    qdrant_client_instance: qdrant_client.QdrantClient
) -> Tuple[Optional[SubQuestionQueryEngine], List[str]]:
    # (Code identical to previous version)
    st.info("🚀 Starting engine setup...")
    start_time = datetime.datetime.now(); agent_tools = []; processed_filenames = []
    if not configure_global_settings(): st.error("Engine setup failed: Core component config error."); return None, []
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
    csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]
    logger.info(f"Found {len(pdf_files)} PDF(s) and {len(csv_files)} CSV(s).")
    # Cleanup Resources
    if os.path.exists(SQL_DB_PATH):
        try: logger.warning(f"Removing DB: {SQL_DB_PATH}"); os.remove(SQL_DB_PATH); logger.info("Old DB removed.")
        except OSError as e: logger.error(f"DB remove error: {e}", exc_info=True); st.error(f"DB cleanup error: {e}")
    if qdrant_client_instance:
        logger.warning(f"Cleaning Qdrant (prefix '{QDRANT_PDF_COLLECTION_PREFIX}')..."); cleaned_count = 0
        try:
            collections = qdrant_client_instance.get_collections().collections
            for collection in collections:
                if collection.name.startswith(QDRANT_PDF_COLLECTION_PREFIX):
                    logger.info(f"Deleting old Qdrant: {collection.name}")
                    try: qdrant_client_instance.delete_collection(collection_name=collection.name, timeout=60); cleaned_count += 1
                    except Exception as del_exc: logger.error(f"Failed delete Qdrant '{collection.name}': {del_exc}"); st.warning(f"Could not delete '{collection.name}'.")
            logger.info(f"Qdrant cleanup finished. Deleted {cleaned_count}.")
        except Exception as list_exc: logger.error(f"Qdrant list fail: {list_exc}", exc_info=True); st.warning("Qdrant cleanup failed.")
    else: logger.warning("Qdrant client unavailable, skipping cleanup.")
    # Process PDFs
    pdf_success_count = 0
    if pdf_files:
        st.write(f"Processing {len(pdf_files)} PDF(s)..."); pdf_progress = st.progress(0.0)
        if not qdrant_client_instance: st.error("Qdrant unavailable.")
        else:
            for i, uploaded_pdf in enumerate(pdf_files):
                progress_text_pdf = f"PDF: {uploaded_pdf.name} ({i+1}/{len(pdf_files)})..."
                try:
                    pdf_progress.progress((i / len(pdf_files)) + (0.1 / len(pdf_files)), text=progress_text_pdf)
                    pdf_doc = process_pdf(uploaded_pdf)
                    if pdf_doc:
                        pdf_progress.progress((i / len(pdf_files)) + (0.5 / len(pdf_files)), text=f"Indexing {uploaded_pdf.name}...")
                        pdf_tool = create_pdf_tool(pdf_doc, qdrant_client_instance)
                        if pdf_tool: agent_tools.append(pdf_tool); processed_filenames.append(uploaded_pdf.name); pdf_success_count += 1
                except Exception as pdf_loop_err: logger.error(f"PDF loop error {uploaded_pdf.name}: {pdf_loop_err}", exc_info=True); st.error(f"Processing {uploaded_pdf.name} failed.")
            pdf_progress.progress(1.0, text="PDF Processing Complete.")
    # Process CSVs
    csv_success_count = 0; sql_alchemy_engine = None
    if csv_files:
        st.write(f"Processing {len(csv_files)} CSV(s)..."); csv_progress = st.progress(0.0)
        try:
            logger.info(f"Creating SQLAlchemy engine: {SQL_DB_URL}"); sql_alchemy_engine = sqlalchemy.create_engine(SQL_DB_URL)
            with sql_alchemy_engine.connect() as conn: logger.info("DB Engine connected.")
            for i, uploaded_csv in enumerate(csv_files):
                 progress_text_csv = f"CSV: {uploaded_csv.name} ({i+1}/{len(csv_files)})..."
                 try:
                     csv_progress.progress((i / len(csv_files)) + (0.1 / len(csv_files)), text=progress_text_csv)
                     csv_df = process_csv(uploaded_csv)
                     if csv_df is not None:
                         csv_progress.progress((i / len(csv_files)) + (0.5 / len(csv_files)), text=f"Creating DB for {uploaded_csv.name}...")
                         csv_tool = create_csv_tool(csv_df, uploaded_csv.name, sql_alchemy_engine)
                         if csv_tool: agent_tools.append(csv_tool); processed_filenames.append(uploaded_csv.name); csv_success_count += 1
                 except Exception as csv_loop_err: logger.error(f"CSV loop error {uploaded_csv.name}: {csv_loop_err}", exc_info=True); st.error(f"Processing {uploaded_csv.name} failed.")
            csv_progress.progress(1.0, text="CSV Processing Complete.")
        except Exception as db_eng_err: logger.error(f"SQLAlchemy engine fail: {db_eng_err}", exc_info=True); st.error(f"DB engine error: {db_eng_err}."); sql_alchemy_engine = None
    st.write(f"✅ Processing finished: {pdf_success_count} PDF tool(s), {csv_success_count} CSV tool(s).")
    # Build Engine
    if not agent_tools: st.error("Engine setup failed: No tools created."); return None, []
    st.success(f"Created {len(agent_tools)} tools for: {', '.join(processed_filenames) or 'None'}"); logger.info(f"Tools created: {len(agent_tools)}. Files: {processed_filenames}")
    st.info("Building the main query engine..."); logger.info("Creating SubQuestionQueryEngine...")
    try:
        if not Settings.llm: raise ValueError("LLM not in Settings.")
        final_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=agent_tools, verbose=True, use_async=False )
        logger.info("SubQuestionQueryEngine created."); st.success("🎉 Query Engine is ready!")
        end_time = datetime.datetime.now(); st.caption(f"Setup took {(end_time - start_time).total_seconds():.2f}s.")
        return final_engine, processed_filenames
    except Exception as e: logger.error(f"Failed SubQuestionQueryEngine init: {e}", exc_info=True); st.error(f"Query Engine creation failed: {e}"); return None, processed_filenames

# ==============================================================================
# --- Streamlit App UI (Unified - Uses SubQuestionQueryEngine) ---
# ==============================================================================
st.set_page_config(page_title="Unified SubQuery Engine - Final", layout="wide")
st.title("📄📊 Unified PDF & CSV Analysis Engine (SubQuery)")
st.markdown("""
Upload PDF and/or CSV files. The engine uses vector search for PDFs (one index per PDF)
and a custom SQL engine for CSVs (one table per CSV), coordinated by a Sub-Question Query Engine.
**Note:** Uses a dummy LLM - actual query analysis requires replacing `MyCustomLLM`.
""")

# --- Initialize Core Components & Session State ---
qdrant_client_instance = get_qdrant_client()
if 'query_engine' not in st.session_state: st.session_state.query_engine = None
if 'chat_messages' not in st.session_state: st.session_state.chat_messages = []
if 'processed_filenames' not in st.session_state: st.session_state.processed_filenames = []
if 'processing_complete' not in st.session_state: st.session_state.processing_complete = False
if 'initial_message_shown' not in st.session_state: st.session_state.initial_message_shown = False # Flag for initial message

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Files")
    uploaded_files = st.file_uploader("Upload PDF and/or CSV files", type=['pdf', 'csv'], accept_multiple_files=True, key="unified_file_uploader")
    st.header("2. Process Files")
    process_button_disabled = not uploaded_files or qdrant_client_instance is None
    if qdrant_client_instance is None: st.error("Qdrant init failed. Cannot process PDFs.")

    if st.button("Process Files & Build Engine", type="primary", disabled=process_button_disabled, key="unified_process_button"):
        # Clear state for reprocessing
        st.session_state.query_engine = None
        st.session_state.chat_messages = []
        st.session_state.processed_filenames = []
        st.session_state.processing_complete = False
        st.session_state.initial_message_shown = False # Reset flag
        if uploaded_files:
            with st.spinner("Processing files and building engine... Please wait."): # Add spinner for processing
                engine_instance, processed_names = setup_engine(uploaded_files, qdrant_client_instance)
            st.session_state.query_engine = engine_instance
            st.session_state.processed_filenames = processed_names # Store filenames from setup
            st.session_state.processing_complete = True # Mark as done (attempted at least)
            if engine_instance is None: logger.error("Engine setup returned None.")
            else: logger.info("Engine created and stored.")
        else: st.warning("Please upload files.")

    # --- Config Info ---
    st.sidebar.divider()
    st.sidebar.header("Configuration Info")
    st.sidebar.info(f"LLM: {LLM_MODEL_NAME} (Custom Dummy)")
    st.sidebar.info(f"Embedding: {EMBEDDING_MODEL_NAME}")
    if qdrant_client_instance:
        st.sidebar.info(f"Vector Store: Qdrant (Prefix: {QDRANT_PDF_COLLECTION_PREFIX})")
        st.sidebar.caption(f"Qdrant Path: {os.path.abspath(QDRANT_PERSIST_DIR)}")
    else: st.sidebar.warning("Vector Store: Qdrant Client Failed!")
    st.sidebar.info(f"CSV DB: SQLite ({SQL_DB_FILENAME})")
    st.sidebar.caption(f"DB Path: {os.path.abspath(SQL_DB_PATH)}")

# --- Main Chat Area ---
st.header("💬 Chat with the Engine")

final_engine = st.session_state.get('query_engine', None)
processed_files_list = st.session_state.get('processed_filenames', [])
processing_done = st.session_state.get('processing_complete', False)
initial_message_shown_flag = st.session_state.get('initial_message_shown', False)

# Display initial guidance message only ONCE after successful processing
if processing_done and final_engine and not initial_message_shown_flag:
    with st.chat_message("assistant"):
        greeting_msg = f"Hello! I've processed the following {len(processed_files_list)} file(s): `{', '.join(processed_files_list)}`.\n\n"
        if len(processed_files_list) > 2:
            greeting_msg += "**Tip:** Since multiple files are loaded, asking questions that mention specific filenames helps me answer faster and more accurately. For example:\n"
            example_pdf = next((f for f in processed_files_list if f.lower().endswith('.pdf')), '[filename.pdf]') # Placeholder if none
            example_csv = next((f for f in processed_files_list if f.lower().endswith('.csv')), '[data.csv]') # Placeholder if none
            file1_example = processed_files_list[0] if processed_files_list else '[file1.pdf]'
            file2_example = processed_files_list[1] if len(processed_files_list) > 1 else '[file2.csv]'
            # Add examples
            greeting_msg += f"- 'Summarize `{example_pdf}`'\n"
            greeting_msg += f"- 'What is the total revenue in `{example_csv}`?'\n"
            greeting_msg += f"- 'Compare findings in `{file1_example}` with data in `{file2_example}` using `[common topic/column]`'\n" # Comparison example
            greeting_msg += "\n"
        greeting_msg += "How can I help you analyze these documents?"
        st.markdown(greeting_msg)
        st.session_state.chat_messages.append({"role": "assistant", "content": greeting_msg})
        st.session_state.initial_message_shown = True # Set flag

# Display chat history
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Logic
chat_input_disabled = final_engine is None
if not uploaded_files and not final_engine: st.info("👈 Upload files & Process to start.")
elif final_engine is None and processing_done: st.warning("Engine init failed. Check logs. Chat disabled.")
elif final_engine is None and not processing_done: st.info("👈 Click 'Process Files & Build Engine' to enable chat.")

if prompt := st.chat_input("Ask a question about the uploaded files...", key="unified_chat_prompt", disabled=chat_input_disabled):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # Assistant Response Area
    with st.chat_message("assistant"):
        info_message_placeholder = st.empty()
        final_response_placeholder = st.empty()
        full_assistant_response_for_history = ""

        with st.spinner("Thinking... 🤔"):
            pre_check_info_msg = ""
            # Step 10: Pre-Processing LLM Check
            try:
                if final_engine and Settings.llm:
                    num_tools = len(getattr(final_engine, 'query_engine_tools', [])) # Safely get tool count
                    current_filenames = st.session_state.get('processed_filenames', [])
                    example_pdf = next((f for f in current_filenames if f.lower().endswith('.pdf')), '[filename.pdf]')
                    example_csv = next((f for f in current_filenames if f.lower().endswith('.csv')), '[data.csv]')
                    file1_example = current_filenames[0] if current_filenames else '[file1.pdf]'
                    file2_example = current_filenames[1] if len(current_filenames) > 1 else '[file2.csv]'

                    # Construct the enhanced check prompt for the LLM
                    special_check_prompt = f"""Analyze the user query provided below based on the available files.

User Query:
{prompt}

Available Files:
{', '.join(current_filenames)}

Number of Files: {len(current_filenames)}

Instructions:
1. Check if the query mentions specific filenames from the 'Available Files' list.
2. Check if the query asks for a comparison between files (e.g., uses 'compare', 'difference', 'vs', 'than').
3. If comparison, check if a clear linking factor (e.g., date, ID, category) is mentioned.

Determine Warning Rules:
- Rule 1: Is the query general (no specific filenames) AND number of files > 2?
- Rule 2: Is the query a comparison BUT lacks a clear linking factor?

Response Rules:
- If Rule 1 applies (and Rule 2 does not), respond ONLY with: "INFO: Your query seems general. Processing across all {len(current_filenames)} files might take longer. For focused results next time, try specifying files (e.g., 'Summarize `{example_pdf}`' or 'Show data for `{example_csv}`'). I will now proceed with your general query..."
- If Rule 2 applies (and Rule 1 does not), respond ONLY with: "INFO: Your comparison query doesn't specify how to link the data (e.g., by date, ID). I'll attempt the comparison based on the content, but for a more precise result next time, please include the linking factor (e.g., 'Compare `{file1_example}` and `{file2_example}` for `[linking_factor]`). Proceeding with your query now..."
- If BOTH Rule 1 and Rule 2 apply, respond ONLY with: "INFO: Your query is general and the comparison basis is unclear. Processing across all {len(current_filenames)} files and attempting comparison may take longer and yield limited results. Please specify filenames and a linking factor next time for better results. Proceeding with your query now..."
- If NEITHER Rule 1 nor Rule 2 apply, respond ONLY with the exact text: "OK"
"""
                    logger.info("Performing pre-query check with LLM...")
                    check_response = Settings.llm.complete(special_check_prompt)
                    check_response_text = check_response.text

                    # Handle Dummy LLM response explicitly for testing flow
                    if "This is a dummy response" in check_response_text:
                        logger.warning("Pre-check LLM is a dummy, simulating 'OK'.")
                        check_response_text = "OK" # Treat dummy as OK

                    if check_response_text != "OK" and check_response_text:
                         pre_check_info_msg = check_response_text
                         info_message_placeholder.info(pre_check_info_msg) # Display immediately
                         full_assistant_response_for_history += pre_check_info_msg + "\n\n"
                else:
                    logger.warning("Engine or LLM not available for pre-query check.")

            except Exception as pre_check_err: logger.error(f"Pre-query check error: {pre_check_err}", exc_info=True)

            # Step 12: Execute Main Engine
            final_engine_response_str = ""
            try:
                if final_engine:
                    logger.info(f"--- Querying SubQuestionQueryEngine: {prompt} ---")
                    response_obj = final_engine.query(prompt) # Pass original prompt
                    # Handle potential Response object or direct string
                    if hasattr(response_obj, 'response') and isinstance(response_obj.response, str):
                        final_engine_response_str = response_obj.response
                    elif isinstance(response_obj, str):
                         final_engine_response_str = response_obj
                    elif response_obj is not None: # Fallback for other types
                         final_engine_response_str = str(response_obj)
                    else: final_engine_response_str = "Engine returned no response."
                    logger.info(f"--- SubQuestionQueryEngine response received ---")
                else: final_engine_response_str = "Error: Query engine is not available."
            except Exception as query_err:
                logger.error(f"Engine query error: {query_err}", exc_info=True)
                final_engine_response_str = f"Sorry, an error occurred: {query_err}"

        # Display final response (use markdown for potential formatting)
        final_response_placeholder.markdown(final_engine_response_str)
        full_assistant_response_for_history += final_engine_response_str

    st.session_state.chat_messages.append({"role": "assistant", "content": full_assistant_response_for_history})

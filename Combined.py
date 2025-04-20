```python
# --- Imports ---
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
    SQLDatabase,  # Not directly used by custom SQL, but good to have
    QueryBundle,
    PromptTemplate,
    SimpleDirectoryReader,
)
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.agent import ReActAgent
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
    ResponseMode
)

# LlamaIndex LLM Imports
from llama_index.core.llms import (
    LLM,
    CompletionResponse,
    ChatResponse,
    ChatMessage,
    MessageRole,
    LLMMetadata,
)

# LlamaIndex Embeddings & Vector Stores
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus

# --- Configuration ---
QDRANT_PERSIST_DIR = "./qdrant_storage_combined_agent"
# NOTE: Using a prefix for PDF collections for easier cleanup
QDRANT_PDF_COLLECTION_PREFIX = "agent_pdf_coll_"
SQL_DB_DIR = "./sql_database_combined_agent"
SQL_DB_FILENAME = "csv_data_combined.db"
SQL_DB_PATH = os.path.join(SQL_DB_DIR, SQL_DB_FILENAME)
SQL_DB_URL = f"sqlite:///{SQL_DB_PATH}"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384
LLM_MODEL_NAME = "custom_abc_llm" # Placeholder name

# Create directories if they don't exist
os.makedirs(QDRANT_PERSIST_DIR, exist_ok=True)
os.makedirs(SQL_DB_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# --- Helper Functions for CSV Description (Copied from Script 2) ---
# ==============================================================================

def got_type(list_):
    def judge(string):
        s_val = str(string) if string is not None else "";
        if not s_val: return "string"
        try: int(s_val); return "int"
        except ValueError:
            try: float(s_val); return "float"
            except ValueError: return "string"
    return [judge(str(x) if x is not None else "") for x in list_]

def column_des(df):
    def single_des(name,data):
        description = "{\"Column Name\": \"" + name + "\"" + ", "; valid_data = data.dropna().tolist();
        if not valid_data: return ""
        pre_len = len(data); post_len = len(valid_data); types = got_type(valid_data)
        if "string" in types: type_ = "string"; data_proc = [str(x) for x in valid_data]
        elif "float" in types: type_ = "float"; data_proc = np.array([float(x) for x in valid_data])
        else: type_ = "int"; data_proc = np.array([int(x) for x in valid_data])
        description += "\"Type\": \"" + type_ + "\", "
        if type_ in ["int", "float"]:
            if data_proc.size > 0: min_ = data_proc.min(); max_ = data_proc.max(); description += "\"MIN\": " + str(min_) + ", \"MAX\": " + str(max_)
            else: description += "\"MIN\": null, \"MAX\": null"
        elif type_ == "string":
            values = list(set(["\"" + str(x).strip().replace('"',"'") + "\"" for x in data_proc]))
            random.shuffle(values);
            if len(values) > 15: values = values[:random.randint(5, 10)]
            numerates = ", ".join(values); description += "\"Sample Values\": [" + numerates + "]"
        description += ", \"Contains NaN\": " + str(post_len != pre_len); return description + "}"
    columns_dec = [single_des(c,df[c]) for c in df.columns]; random.shuffle(columns_dec)
    return "\n".join([x for x in columns_dec if x])

def generate_table_description(df: pd.DataFrame, table_name: str, source_csv_name: str) -> str:
    rows_count, columns_count = df.shape
    description = f"Table Name: '{table_name}' (derived from CSV: '{source_csv_name}')\n"
    description += f"Contains {rows_count} rows and {columns_count} columns.\n"
    # Use original columns for description context if available before renaming
    # However, the function expects the df passed to it, which might be the renamed one.
    # Let's assume df passed might be renamed, so we use its columns.
    description += f"SQL Table Columns: {', '.join(df.columns)}\n"
    description += f"--- Column Details and Sample Data ---\n"
    col_descriptions = column_des(df) # Use the columns from the DataFrame passed
    description += col_descriptions if col_descriptions else "No detailed column descriptions generated."
    return description

# ==============================================================================
# --- Custom LLM Implementation (Copied from Script 2) ---
# ==============================================================================

def abc_response(prompt: str) -> str:
    logger.info(f"MyCustomLLM received prompt (first 100 chars): {prompt[:100]}...")
    # Simulate thinking time
    # import time
    # time.sleep(0.5)
    response = f"This is a dummy response from MyCustomLLM for the prompt starting with: {prompt[:50]}..."
    logger.info(f"MyCustomLLM generated response (first 100 chars): {response[:100]}...")
    return response

class MyCustomLLM(LLM):
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
        # In a real scenario, use asyncio.to_thread or similar for non-blocking calls
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        logger.info("MyCustomLLM: acomplete() called - calling sync complete()")
        # In a real scenario, use asyncio.to_thread or similar for non-blocking calls
        return self.complete(prompt, **kwargs)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        logger.warning("MyCustomLLM: stream_chat() called - NotImplementedError")
        # Simulate single response for non-streaming agent
        yield self.chat(messages, **kwargs)
        # raise NotImplementedError("Streaming chat not supported by MyCustomLLM.")

    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        logger.warning("MyCustomLLM: stream_complete() called - NotImplementedError")
        # Simulate single response for non-streaming agent
        yield self.complete(prompt, **kwargs)
        # raise NotImplementedError("Streaming complete not supported by MyCustomLLM.")

    # --- Async Streaming Stubs (Required by ABC) ---
    # These yield immediately after raising, satisfying the AsyncGenerator type hint
    # but indicating no actual streaming support.
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        logger.warning("MyCustomLLM: astream_chat() called - NotImplementedError")
        # Simulate single response for non-streaming agent
        # In a real async implementation, you might yield chunks here.
        # For a dummy, yielding the single response after the error might be confusing.
        # Let's just make it clear it's not supported.
        raise NotImplementedError("Async streaming chat not supported by MyCustomLLM.")
        yield

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        logger.warning("MyCustomLLM: astream_complete() called - NotImplementedError")
        raise NotImplementedError("Async streaming complete not supported by MyCustomLLM.")
        yield

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=LLM_MODEL_NAME, is_chat_model=True) # Assuming it behaves like a chat model

# ==============================================================================
# --- Custom SQL Engine (Copied from Script 2) ---
# ==============================================================================

class CustomSQLEngine:
    """Custom SQL engine with no direct LlamaIndex dependencies"""

    def __init__(
        self,
        sql_engine: sqlalchemy.engine.Engine,
        table_name: str,
        llm_callback: Callable[[str], str],
        table_description: Optional[str] = None,
        verbose: bool = True
    ):
        self.sql_engine = sql_engine
        self.table_name = table_name
        self.llm_callback = llm_callback
        self.table_description = table_description or ""
        self.verbose = verbose

        # Prompt templates (same as Script 2)
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

    def _get_schema_info(self) -> str:
        """Get detailed schema information for the table."""
        try:
            metadata = sqlalchemy.MetaData()
            metadata.reflect(bind=self.sql_engine, only=[self.table_name])
            table = metadata.tables.get(self.table_name)

            if not table:
                return f"Error: Could not find table '{self.table_name}' in database."

            columns = []
            for column in table.columns:
                col_type = str(column.type)
                constraints = []
                if column.primary_key: constraints.append("PRIMARY KEY")
                if not column.nullable: constraints.append("NOT NULL")
                # Note: UNIQUE and FOREIGN KEY constraints might not be reliably created by pandas to_sql
                # if column.unique: constraints.append("UNIQUE")
                # if column.foreign_keys:
                #     for fk in column.foreign_keys: constraints.append(f"FOREIGN KEY → {fk.target_fullname}")

                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                columns.append(f"  - \"{column.name}\": {col_type}{constraint_str}") # Quote column names for clarity

            schema_info = f"Actual Schema for table \"{self.table_name}\":\nColumns:\n" + "\n".join(columns)
            return schema_info

        except Exception as e:
            logger.error(f"Error getting schema info for {self.table_name}: {e}", exc_info=True)
            return f"Error retrieving schema for table {self.table_name}: {e}"

    def _clean_sql(self, sql: str) -> str:
        """Clean SQL query from LLM artifacts."""
        # Remove markdown code blocks
        sql = re.sub(r'```sql|```', '', sql)
        # Remove "sql" if it appears at the start
        sql = re.sub(r'^sql\s+', '', sql, flags=re.IGNORECASE)
        # Remove SQL comments -- ... and /* ... */
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        return sql.strip()

    def _is_safe_sql(self, sql: str) -> bool:
        """Check if SQL query is safe (basic read-only check)."""
        lower_sql = sql.lower().strip()
        # Allow SELECT and potentially CTEs (WITH)
        if not lower_sql.startswith('select') and not lower_sql.startswith('with'):
            logger.warning(f"SQL Safety Check: Query does not start with SELECT or WITH: {sql}")
            return False

        # Disallow keywords that modify data or schema
        dangerous_keywords = [
            r'\bdrop\b', r'\bdelete\b', r'\btruncate\b', r'\bupdate\b',
            r'\binsert\b', r'\balter\b', r'\bcreate\b', r'\breplace\b',
            r'\bgrant\b', r'\brevoke\b', r'\battach\b', r'\bdetach\b'
        ]
        for pattern in dangerous_keywords:
            if re.search(pattern, lower_sql):
                 # Allow CREATE TEMPORARY TABLE specifically if needed, but generally avoid CREATE
                 # if pattern == r'\bcreate\b' and re.search(r'\bcreate\s+temporary\s+table\b', lower_sql):
                 #      continue # Allow temporary table creation
                 logger.warning(f"SQL Safety Check: Found potentially dangerous keyword matching '{pattern}' in: {sql}")
                 return False
        return True

    def _format_results(self, result_df: pd.DataFrame) -> str:
        """Format query results into a readable string."""
        if result_df.empty:
            return "The query returned no results."

        # For large results, limit the output
        max_rows_to_show = 20
        if len(result_df) > max_rows_to_show:
            result_str = f"Query returned {len(result_df)} rows. Showing first {max_rows_to_show}:\n\n"
            # Use .to_markdown for better formatting in Streamlit
            return result_str + result_df.head(max_rows_to_show).to_markdown(index=False)
        else:
            return result_df.to_markdown(index=False)

    def _execute_sql(self, sql: str) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """Execute SQL query and return results or error message."""
        try:
            if not self._is_safe_sql(sql):
                logger.error(f"SQL Safety Check Failed for query: {sql}")
                return False, "SQL query failed safety check (only SELECT queries are allowed)."

            # Execute the query using pandas read_sql_query
            if self.verbose: logger.info(f"Executing safe SQL on table {self.table_name}: {sql}")
            with self.sql_engine.connect() as connection:
                 result_df = pd.read_sql_query(sql, connection)
            return True, result_df

        except sqlalchemy.exc.SQLAlchemyError as db_err:
            error_msg = f"Database Error: {db_err}"
            if self.verbose: logger.error(f"SQL execution error: {error_msg}\nQuery: {sql}", exc_info=True)
            return False, error_msg
        except Exception as e:
            error_msg = f"General Error: {e}"
            if self.verbose: logger.error(f"SQL execution error: {error_msg}\nQuery: {sql}", exc_info=True)
            return False, error_msg

    def _execute_with_retry(self, sql: str, max_retries: int = 1) -> Tuple[bool, Union[pd.DataFrame, str], str]:
        """Execute SQL with retry and correction attempts."""
        current_sql = sql
        original_sql = sql

        for attempt in range(max_retries + 1):
            if self.verbose: logger.info(f"SQL Execution Attempt {attempt+1}/{max_retries+1}")

            success, result = self._execute_sql(current_sql)

            if success:
                if self.verbose: logger.info("SQL execution successful")
                return True, result, current_sql # result is DataFrame here

            # --- Execution Failed ---
            error_message = str(result) # result is error string here
            logger.warning(f"SQL attempt {attempt+1} failed. Error: {error_message}")

            # If we've exhausted retries, return the final error
            if attempt == max_retries:
                logger.error(f"SQL failed after {max_retries+1} attempts. Final error: {error_message}")
                return False, f"SQL execution failed: {error_message}", current_sql

            # --- Try to fix the query ---
            if self.verbose: logger.info("Attempting to fix SQL using LLM...")

            try:
                # Get current schema info for the fix prompt
                schema_info = self._get_schema_info()

                # Generate fix prompt
                fix_prompt = self.sql_fix_prompt.format(
                    failed_sql=current_sql,
                    error_msg=error_message,
                    table_name=self.table_name,
                    table_description=f"{self.table_description}\n\n{schema_info}"
                )

                # Get fixed SQL from LLM
                fixed_sql_response = self.llm_callback(fix_prompt)
                fixed_sql = self._clean_sql(fixed_sql_response)

                if fixed_sql and fixed_sql.lower() != current_sql.lower():
                    current_sql = fixed_sql
                    if self.verbose: logger.info(f"LLM proposed fixed SQL: {current_sql}")
                    # Loop will continue and try executing this new query
                else:
                    if self.verbose: logger.warning("LLM did not provide a different SQL query or failed to fix. Stopping retries.")
                    # Don't retry if LLM didn't change the query
                    return False, f"SQL execution failed: {error_message} (LLM correction failed)", original_sql

            except Exception as fix_error:
                logger.error(f"Error during LLM SQL fix attempt: {fix_error}", exc_info=True)
                # Stop retries if the fix attempt itself fails
                return False, f"SQL execution failed: {error_message} (LLM fix attempt failed: {fix_error})", original_sql

        # Should not be reached if loop logic is correct
        return False, "Unexpected error during SQL execution loop", original_sql

    def query(self, query_text: str) -> str:
        """Process a natural language query and return results as a formatted string."""
        if self.verbose: logger.info(f"CustomSQLEngine received query for table {self.table_name}: {query_text}")

        try:
            # Get schema info for generation prompt
            schema_info = self._get_schema_info()

            # Generate SQL query from natural language
            if self.verbose: logger.info("Generating initial SQL query from natural language...")

            generate_prompt = self.sql_prompt_template.format(
                query_str=query_text,
                table_name=self.table_name,
                table_description=f"{self.table_description}\n\n{schema_info}"
            )

            # Get SQL from LLM
            sql_response = self.llm_callback(generate_prompt)
            sql_query = self._clean_sql(sql_response)

            if not sql_query:
                logger.error("LLM failed to generate any SQL query.")
                return "Error: Could not generate SQL query from your question."

            if self.verbose: logger.info(f"LLM generated SQL: {sql_query}")

            # Execute SQL query with retry/fix mechanism
            success, result, final_sql = self._execute_with_retry(sql_query)

            if not success:
                # Result is the error message string here
                logger.error(f"Final SQL execution failed. Error: {result}")
                # Format error nicely for the agent/user
                return f"I encountered an error trying to query the data for table '{self.table_name}':\n{result}\n\nSQL attempted:\n```sql\n{final_sql}\n```"

            # --- Execution Succeeded ---
            # Result is the pandas DataFrame here
            if self.verbose: logger.info(f"SQL query successful. Formatting results from DataFrame.")
            formatted_results = self._format_results(result)

            # Prepare response string
            response = (
                f"Executed SQL query on table '{self.table_name}' and found the following:\n\n"
                f"{formatted_results}\n\n"
                # f"(Source: {self.table_name} table from CSV)\n" # Agent should add sourcing
                # f"SQL query used: ```sql\n{final_sql}\n```" # Maybe too verbose for user? Agent sees it.
            )
            return response

        except Exception as e:
            logger.error(f"Unexpected error in CustomSQLEngine.query for table {self.table_name}: {e}", exc_info=True)
            return f"An unexpected error occurred while processing your query for table '{self.table_name}': {e}"

# ==============================================================================
# --- Global Settings and Initialization ---
# ==============================================================================

# Use Streamlit's caching for resources that are expensive to create
@st.cache_resource
def get_llm():
    logger.info("Initializing Custom LLM (cached)...")
    return MyCustomLLM()

@st.cache_resource
def get_embed_model():
    logger.info(f"Initializing Embedding Model (cached): {EMBEDDING_MODEL_NAME}")
    try:
        # Potential place for device selection if using GPU: model_kwargs={'device': 'cuda'}
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        # Test embedding immediately
        test_embedding = embed_model.get_query_embedding("test")
        actual_dim = len(test_embedding)
        if actual_dim != EXPECTED_EMBEDDING_DIM:
            raise ValueError(f"Embed dim mismatch! Expected {EXPECTED_EMBEDDING_DIM}, Got {actual_dim}.")
        logger.info(f"Embedding model loaded successfully. Dimension: {actual_dim}")
        return embed_model
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize embedding model: {e}", exc_info=True)
        st.error(f"Embedding model initialization failed: {e}")
        return None # Indicate failure

@st.cache_resource
def get_qdrant_client() -> Optional[qdrant_client.QdrantClient]:
    logger.info(f"Initializing Qdrant client (cached) (Path: {QDRANT_PERSIST_DIR})...")
    try:
        client = qdrant_client.QdrantClient(path=QDRANT_PERSIST_DIR)
        # Perform a quick check to see if client is operational
        client.get_collections() # This will raise error if connection fails
        logger.info("Qdrant client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize Qdrant client: {e}", exc_info=True)
        st.error(f"Qdrant vector database initialization failed: {e}. PDF analysis will be unavailable.")
        return None # Indicate failure

def configure_global_settings() -> bool:
    """Configure LlamaIndex global settings. Returns True on success."""
    logger.info("Configuring LlamaIndex Global Settings...")
    try:
        llm = get_llm()
        embed_model = get_embed_model()

        if llm is None or embed_model is None:
             logger.error("LLM or Embedding Model failed to initialize. Cannot configure settings.")
             return False

        Settings.llm = llm
        Settings.embed_model = embed_model
        # Node parser settings
        Settings.node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=200)
        # Other settings (optional)
        Settings.num_output = 512
        Settings.context_window = 4096 # Adjust based on the real LLM context window

        logger.info("LlamaIndex Global Settings configured successfully.")
        return True
    except Exception as e:
        logger.error(f"Error configuring LlamaIndex Settings: {e}", exc_info=True)
        st.error(f"Core component configuration failed: {e}")
        return False

# ==============================================================================
# --- Data Processing Functions ---
# ==============================================================================

def process_pdf(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[Document]:
    """Processes a single uploaded PDF file into a LlamaIndex Document."""
    if not uploaded_file:
        return None
    file_name = uploaded_file.name
    logger.info(f"Processing PDF: {file_name}")
    try:
        with io.BytesIO(uploaded_file.getvalue()) as pdf_stream:
            reader = PyPDF2.PdfReader(pdf_stream)
            text_content = ""
            num_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n\n" # Add separator between pages
                    else:
                        logger.warning(f"No text found on page {page_num + 1}/{num_pages} of '{file_name}'.")
                except Exception as page_err:
                    logger.error(f"Error extracting text from page {page_num + 1} of '{file_name}': {page_err}", exc_info=True)
                    # Optionally add a placeholder or skip the page
                    text_content += f"[Error reading page {page_num + 1}]\n\n"

        if not text_content.strip():
            st.warning(f"No text could be extracted from PDF '{file_name}'. It might be image-based or corrupted.")
            logger.warning(f"No text content extracted from '{file_name}'.")
            return None

        # Create a single document for the whole PDF
        doc = Document(text=text_content, metadata={"file_name": file_name})
        logger.info(f"Successfully extracted text from PDF '{file_name}'. Total length: {len(text_content)}")
        return doc
    except Exception as e:
        logger.error(f"Failed to process PDF '{file_name}': {e}", exc_info=True)
        st.error(f"Error reading PDF file '{file_name}': {e}")
        return None

def process_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[pd.DataFrame]:
    """Processes a single uploaded CSV file into a pandas DataFrame."""
    if not uploaded_file:
        return None
    file_name = uploaded_file.name
    logger.info(f"Processing CSV: {file_name}")
    try:
        # Try standard UTF-8 first
        try:
            df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for '{file_name}'. Attempting latin1 encoding.")
            # Fallback to latin1 if UTF-8 fails
            df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), encoding='latin1')

        # Remove potential unnamed columns from Excel exports
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        logger.info(f"Loaded CSV '{file_name}'. Shape: {df.shape}")

        if df.empty:
            st.warning(f"CSV file '{file_name}' is empty or contains no usable data.")
            logger.warning(f"CSV '{file_name}' resulted in an empty DataFrame.")
            return None

        return df
    except Exception as e:
        logger.error(f"Failed to process CSV '{file_name}': {e}", exc_info=True)
        st.error(f"Error reading CSV file '{file_name}': {e}")
        return None

# ==============================================================================
# --- Tool Creation Functions ---
# ==============================================================================

def sanitize_for_name(filename: str, max_len: int = 40) -> str:
    """Sanitizes a filename to be used in collection/table names."""
    # Remove extension
    name = Path(filename).stem
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r'\W+', '_', name) # Use underscore instead of removing
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure it doesn't start with a number (important for some DBs/identifiers)
    if name and name[0].isdigit():
        name = '_' + name # Prepend underscore
    # Truncate if too long
    name = name[:max_len].lower()
    if not name: # Handle empty filenames after sanitization
        name = f"file_{random.randint(1000, 9999)}"
    return name

def create_pdf_vector_tool(
    pdf_document: Document,
    qdrant_client_instance: qdrant_client.QdrantClient
) -> Optional[QueryEngineTool]:
    """Creates a QueryEngineTool for a single PDF document using its own Qdrant collection."""
    if not pdf_document or not pdf_document.text.strip():
        st.error("PDF tool creation failed: Invalid document provided.")
        return None
    if not qdrant_client_instance:
        st.error("PDF tool creation failed: Qdrant client is unavailable.")
        return None

    file_name = pdf_document.metadata.get("file_name", f"unknown_pdf_{random.randint(1000,9999)}.pdf")
    sanitized_name = sanitize_for_name(file_name)
    collection_name = f"{QDRANT_PDF_COLLECTION_PREFIX}{sanitized_name}" # Use prefix
    tool_name = f"pdf_{sanitized_name}_query_tool"

    logger.info(f"Creating tool for PDF: '{file_name}' (Collection: {collection_name}, Tool: {tool_name})")

    try:
        # 1. Create the unique Qdrant Collection for this PDF
        logger.info(f"Creating Qdrant collection: '{collection_name}' with dim {EXPECTED_EMBEDDING_DIM}")
        try:
             # This might raise if collection exists, handled below or use client.recreate_collection
             qdrant_client_instance.create_collection(
                 collection_name=collection_name,
                 vectors_config=VectorParams(size=EXPECTED_EMBEDDING_DIM, distance=Distance.COSINE)
             )
             logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as create_exc:
             # Handle case where collection might already exist if cleanup failed?
             # Or if recreate is preferred: qdrant_client.recreate_collection(...)
             logger.warning(f"Could not create collection '{collection_name}' (might exist or other error): {create_exc}. Attempting to proceed.")
             # Check if it exists now, otherwise fail
             try:
                 qdrant_client_instance.get_collection(collection_name=collection_name)
                 logger.info(f"Collection '{collection_name}' found.")
             except Exception as get_exc:
                 logger.error(f"Failed to ensure Qdrant collection '{collection_name}' exists: {get_exc}", exc_info=True)
                 st.error(f"Qdrant collection error for '{file_name}': {get_exc}")
                 return None


        # 2. Setup LlamaIndex components for this collection
        vector_store = QdrantVectorStore(client=qdrant_client_instance, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info(f"Storage context ready for '{collection_name}'.")

        # 3. Index the document into its collection
        logger.info(f"Indexing document '{file_name}' into '{collection_name}'...")
        # We assume Settings.node_parser and Settings.embed_model are configured globally
        index = VectorStoreIndex.from_documents(
            [pdf_document], # Index only this document
            storage_context=storage_context,
            show_progress=True # Provides feedback in console
        )
        logger.info(f"Document '{file_name}' indexed successfully.")

        # 4. Create query engine for this specific index
        pdf_query_engine = index.as_query_engine(
            similarity_top_k=3, # Retrieve top 3 similar chunks
            response_mode="compact" # Optimize context for LLM
            # LLM is implicitly taken from global Settings
        )
        logger.info(f"Query engine created for index '{collection_name}'.")

        # 5. Create the QueryEngineTool
        tool_description = f"""Provides information from the specific PDF document named '{file_name}'. Use this tool for any questions requiring text search, summarization, or content understanding within '{file_name}'."""
        pdf_tool = QueryEngineTool(
            query_engine=pdf_query_engine,
            metadata=ToolMetadata(
                name=tool_name,
                description=tool_description
            )
        )
        logger.info(f"QueryEngineTool '{tool_name}' created successfully for PDF '{file_name}'.")
        return pdf_tool

    except Exception as e:
        logger.error(f"Failed to create PDF tool for '{file_name}': {e}", exc_info=True)
        st.error(f"Error setting up PDF tool for '{file_name}': {e}")
        return None

def create_custom_sql_tool(
    df: pd.DataFrame,
    csv_file_name: str,
    sql_alchemy_engine: sqlalchemy.engine.Engine # Pass the shared engine
) -> Optional[QueryEngineTool]:
    """Creates a QueryEngineTool for a CSV DataFrame using the CustomSQLEngine."""
    if df is None or df.empty:
        st.error(f"CSV tool creation failed for '{csv_file_name}': DataFrame is empty.")
        return None
    if not sql_alchemy_engine:
        st.error(f"CSV tool creation failed for '{csv_file_name}': SQLAlchemy Engine not provided.")
        return None

    # Sanitize names for SQL table and tool
    sanitized_base_name = sanitize_for_name(csv_file_name)
    table_name = f"csv_{sanitized_base_name}" # Make table names predictable
    tool_name = f"csv_{sanitized_base_name}_query_tool"

    logger.info(f"Creating tool for CSV: '{csv_file_name}' (Table: {table_name}, Tool: {tool_name})")

    try:
        # --- Prepare DataFrame for SQL ---
        original_columns = df.columns.tolist()
        logger.info(f"Original CSV columns for '{csv_file_name}': {original_columns}")

        # Clean column names for SQL compatibility
        cleaned_column_map = {}
        seen_cleaned_names = set()
        for i, col in enumerate(df.columns):
            # Basic cleaning: replace non-alphanumeric with _, lower, handle leading digits
            cleaned_col = re.sub(r'\W+|^(?=\d)', '_', str(col)).lower().strip('_')
            cleaned_col = cleaned_col or f"column_{i}" # Handle empty columns

            # Ensure uniqueness
            final_cleaned_col = cleaned_col
            suffix = 1
            while final_cleaned_col in seen_cleaned_names:
                final_cleaned_col = f"{cleaned_col}_{suffix}"
                suffix += 1
            seen_cleaned_names.add(final_cleaned_col)
            cleaned_column_map[col] = final_cleaned_col

        df_renamed = df.rename(columns=cleaned_column_map)
        cleaned_columns = df_renamed.columns.tolist()
        logger.info(f"Cleaned SQL column names for table '{table_name}': {cleaned_columns}")

        # Ensure string columns are stored as TEXT (avoid potential issues with length limits)
        # Also handle potential datetime conversions if needed, though pandas usually handles this well.
        dtype_mapping = {}
        for col in df_renamed.select_dtypes(include=['object', 'string']).columns:
             # df_renamed[col] = df_renamed[col].astype(str) # Ensure conversion
             dtype_mapping[col] = sqlalchemy.types.TEXT # Map object/string to TEXT

        # --- Load data into SQLite table ---
        logger.info(f"Saving DataFrame {df_renamed.shape} to SQL table '{table_name}'...")
        df_renamed.to_sql(
            table_name,
            sql_alchemy_engine,
            index=False,
            if_exists='replace', # Overwrite table if it exists from previous runs
            chunksize=1000, # Load in chunks for potentially large files
            dtype=dtype_mapping # Specify TEXT type for string columns
            )
        logger.info(f"DataFrame for '{csv_file_name}' saved successfully to table '{table_name}'.")

        # --- Generate Description for the LLM ---
        logger.info(f"Generating description for table '{table_name}'...")
        # Pass the renamed df to generate description based on actual table structure
        table_desc = generate_table_description(df_renamed, table_name, csv_file_name)
        # Add notes about cleaned names if mapping exists and is different
        if original_columns != cleaned_columns:
             column_mapping_info = "\nColumn Name Mapping (Original CSV -> SQL Table):\n"
             for orig, clean in cleaned_column_map.items():
                  if orig != clean: # Only show changed names
                     column_mapping_info += f"- '{orig}' -> '{clean}'\n"
             # table_desc += column_mapping_info # Decided against adding this to reduce prompt length, description shows SQL names.
        logger.info(f"Description generated for table '{table_name}'.")

        # --- Instantiate Custom SQL Engine ---
        logger.info(f"Instantiating CustomSQLEngine for table '{table_name}'...")

        # LLM Callback function (uses globally configured Settings.llm)
        def llm_callback(prompt: str) -> str:
            """Forwards prompts to the globally configured LLM."""
            try:
                if not Settings.llm: raise ValueError("Global LLM not configured in Settings.")
                # Use the complete method for SQL generation/fixing prompts
                llm_result = Settings.llm.complete(prompt)
                return llm_result.text
            except Exception as e:
                logger.error(f"Error in LLM callback for CustomSQLEngine: {e}", exc_info=True)
                # Return error message so the engine knows LLM failed
                return f"LLM Error: {e}"

        custom_sql_engine_instance = CustomSQLEngine(
            sql_engine=sql_alchemy_engine, # Use the shared engine
            table_name=table_name,
            llm_callback=llm_callback,
            table_description=table_desc,
            verbose=True # Enable detailed logging from the custom engine
        )

        # --- Wrap Custom Engine for LlamaIndex Tool Interface ---
        # This adapter class makes our custom engine look like a LlamaIndex QueryEngine
        class CustomSQLQueryEngineWrapper(BaseQueryEngine):
            """Adapter to make CustomSQLEngine compatible with LlamaIndex tools."""
            def __init__(self, engine: CustomSQLEngine):
                self._engine = engine
                # Get metadata from custom engine if needed, or set default
                # super().__init__(metadata=...) # Removed as it's handled differently now
                super().__init__() # Initialize base class

            def _query(self, query_bundle: QueryBundle) -> str: # Changed return type for simplicity
                """Main query interface called by LlamaIndex."""
                # Custom engine expects a string, get it from the bundle
                query_text = query_bundle.query_str
                # Call the custom engine's query method
                result_str = self._engine.query(query_text)
                # Custom engine already returns a formatted string, so just return it
                # For Response object: return Response(response=result_str)
                # For now, returning string might work for ReAct agent, but Response object is safer.
                # Let's stick to the CustomSQLEngine's string output for simplicity as defined.
                return result_str # Return the formatted string result

            async def _aquery(self, query_bundle: QueryBundle) -> str: # Changed return type
                """Async query interface."""
                # Basic async wrapper for the synchronous query method
                # For true async, CustomSQLEngine would need async methods
                logger.info("CustomSQLQueryEngineWrapper: aquery called - using sync query")
                import asyncio
                # Run the synchronous query method in a thread pool
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, self._query, query_bundle)
                return result

            # Implement abstract property from BaseQueryEngine
            @property
            def query_engine(self):
                # This property might not be strictly needed if we don't use methods
                # relying on it, but good practice to implement.
                # Return self or the underlying engine if appropriate.
                return self


        wrapped_engine = CustomSQLQueryEngineWrapper(custom_sql_engine_instance)
        logger.info(f"CustomSQLEngine wrapped successfully for table '{table_name}'.")

        # --- Create the Final QueryEngineTool ---
        tool_description = f"""Queries a SQL table named '{table_name}' which contains data derived from the CSV file '{csv_file_name}'. Use this tool for questions requiring structured data lookup, filtering, calculations (e.g., SUM, COUNT, AVG), or aggregation based on the columns in '{csv_file_name}'. Columns available: {', '.join(cleaned_columns)}."""

        csv_tool = QueryEngineTool(
            query_engine=wrapped_engine, # Use the wrapped engine
            metadata=ToolMetadata(
                name=tool_name,
                description=tool_description
            )
        )
        logger.info(f"QueryEngineTool '{tool_name}' created successfully for CSV '{csv_file_name}'.")
        return csv_tool

    except sqlalchemy.exc.SQLAlchemyError as db_err:
        logger.error(f"Database error during CSV tool creation for '{csv_file_name}' (table '{table_name}'): {db_err}", exc_info=True)
        st.error(f"Database error processing CSV '{csv_file_name}': {db_err}")
        return None
    except Exception as e:
        logger.error(f"Failed to create CSV tool for '{csv_file_name}' (table '{table_name}'): {e}", exc_info=True)
        st.error(f"Error setting up CSV tool for '{csv_file_name}': {e}")
        return None

# ==============================================================================
# --- Agent Setup Function ---
# ==============================================================================

def setup_agent(
    uploaded_pdf_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    uploaded_csv_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    qdrant_client_instance: qdrant_client.QdrantClient
) -> Optional[ReActAgent]:
    """Processes files, creates tools, and sets up the ReActAgent."""
    st.info("🚀 Starting agent setup...")
    start_time = datetime.datetime.now()
    processed_filenames = []
    agent_tools = []

    # 1. Configure Global Settings (LLM, Embedder)
    if not configure_global_settings():
        st.error("Agent setup failed: Core component configuration error.")
        logger.error("Agent setup aborted due to settings configuration failure.")
        return None

    # 2. Cleanup Old Resources
    # Delete old SQLite DB file
    if os.path.exists(SQL_DB_PATH):
        try:
            logger.warning(f"Removing existing SQLite DB file: {SQL_DB_PATH}")
            os.remove(SQL_DB_PATH)
            logger.info("Old DB file removed.")
        except OSError as e:
            logger.error(f"Error removing old DB file '{SQL_DB_PATH}': {e}", exc_info=True)
            st.error(f"Could not remove old database file. Proceeding, but might cause issues: {e}")

    # Delete old Qdrant PDF collections
    if qdrant_client_instance:
        logger.warning(f"Attempting to clean up old Qdrant collections with prefix '{QDRANT_PDF_COLLECTION_PREFIX}'...")
        cleaned_count = 0
        try:
            collections = qdrant_client_instance.get_collections().collections
            for collection in collections:
                if collection.name.startswith(QDRANT_PDF_COLLECTION_PREFIX):
                    logger.info(f"Deleting old Qdrant collection: {collection.name}")
                    try:
                        qdrant_client_instance.delete_collection(collection_name=collection.name)
                        cleaned_count += 1
                    except Exception as del_exc:
                         logger.error(f"Failed to delete Qdrant collection '{collection.name}': {del_exc}")
                         st.warning(f"Could not delete old Qdrant collection '{collection.name}'.")
            logger.info(f"Qdrant cleanup finished. Deleted {cleaned_count} collections.")
        except Exception as list_exc:
            logger.error(f"Failed to list Qdrant collections for cleanup: {list_exc}")
            st.warning("Could not perform Qdrant cleanup.")
    else:
        logger.warning("Qdrant client not available, skipping Qdrant cleanup.")


    # 3. Process PDFs and Create Tools
    pdf_success_count = 0
    if uploaded_pdf_files:
        st.write(f"Processing {len(uploaded_pdf_files)} PDF file(s)...")
        if not qdrant_client_instance:
             st.error("Cannot process PDFs: Qdrant client failed to initialize.")
        else:
            for uploaded_pdf in uploaded_pdf_files:
                with st.spinner(f"Processing PDF: {uploaded_pdf.name}..."):
                    pdf_doc = process_pdf(uploaded_pdf)
                    if pdf_doc:
                        pdf_tool = create_pdf_vector_tool(pdf_doc, qdrant_client_instance)
                        if pdf_tool:
                            agent_tools.append(pdf_tool)
                            processed_filenames.append(uploaded_pdf.name)
                            pdf_success_count += 1
                        else:
                             logger.error(f"Tool creation failed for PDF: {uploaded_pdf.name}")
                             # Error message already shown by create_pdf_vector_tool
                    else:
                         logger.error(f"Text extraction failed for PDF: {uploaded_pdf.name}")
                         # Error message already shown by process_pdf
        st.write(f"✅ Finished processing PDFs. {pdf_success_count}/{len(uploaded_pdf_files)} successful.")


    # 4. Process CSVs and Create Tools
    csv_success_count = 0
    sql_alchemy_engine = None # Initialize outside the loop
    if uploaded_csv_files:
        st.write(f"Processing {len(uploaded_csv_files)} CSV file(s)...")
        try:
            # Create the single SQLAlchemy engine instance for all CSVs
            logger.info(f"Creating SQLAlchemy engine for SQLite DB: {SQL_DB_URL}")
            sql_alchemy_engine = sqlalchemy.create_engine(SQL_DB_URL)
            # Test connection briefly
            with sql_alchemy_engine.connect() as conn:
                 logger.info("SQLAlchemy engine connected successfully.")

            for uploaded_csv in uploaded_csv_files:
                 with st.spinner(f"Processing CSV: {uploaded_csv.name}..."):
                     csv_df = process_csv(uploaded_csv)
                     if csv_df is not None:
                         csv_tool = create_custom_sql_tool(csv_df, uploaded_csv.name, sql_alchemy_engine)
                         if csv_tool:
                             agent_tools.append(csv_tool)
                             processed_filenames.append(uploaded_csv.name)
                             csv_success_count += 1
                         else:
                              logger.error(f"Tool creation failed for CSV: {uploaded_csv.name}")
                     else:
                          logger.error(f"DataFrame creation failed for CSV: {uploaded_csv.name}")

        except Exception as db_eng_err:
             logger.error(f"Failed to create or connect SQLAlchemy engine: {db_eng_err}", exc_info=True)
             st.error(f"Database engine initialization failed: {db_eng_err}. CSV processing aborted.")
             # Ensure engine is None if it failed
             sql_alchemy_engine = None
        finally:
            # Dispose engine if created - important for releasing resources if needed,
            # though for SQLite file DB it might be less critical than server DBs.
            # if sql_alchemy_engine:
            #     sql_alchemy_engine.dispose() # Might cause issues if CustomSQLEngine needs it later? Revisit if needed.
            pass
        st.write(f"✅ Finished processing CSVs. {csv_success_count}/{len(uploaded_csv_files)} successful.")

    # 5. Check if any tools were created
    if not agent_tools:
        st.error("Agent setup failed: No tools could be created from the uploaded files.")
        logger.error("Agent setup aborted: No tools were successfully created.")
        return None

    st.success(f"Successfully created {len(agent_tools)} tools for {len(processed_filenames)} files.")
    logger.info(f"Total tools created: {len(agent_tools)}. Processed files: {processed_filenames}")

    # 6. Create the ReActAgent
    st.info("Building the agent with the created tools...")
    logger.info("Attempting to create ReActAgent...")

    # Define the enhanced system prompt including the clarification rule
    system_prompt = f"""You are an AI assistant designed to analyze and answer questions based on information contained in the uploaded files.
The available files are: {', '.join(processed_filenames)}.
You have access to specific tools, one for each file. Use the tool descriptions to understand which tool corresponds to which file and what kind of information it holds (text from PDF or structured data from CSV via SQL).

Your goal is to answer the user's query accurately. Follow these steps:
1.  **Analyze the Query:** Understand what information the user is asking for and which file(s) likely contain it.
2.  **Select Tool(s):** Choose the appropriate tool(s) based on the query and the tool descriptions (matching filenames and data types).
3.  **Plan Action:** Decide the input/query to give to the selected tool.
4.  **CRITICAL CLARIFICATION STEP:** If the user asks to COMPARE information between two or more files (e.g., "compare sales in file A with targets in file B"), and the basis for comparison is unclear (e.g., common date range, product ID, category), DO NOT immediately use the tools. Instead, your response MUST be a question asking the user for clarification. Ask something like: "To compare X in '[filename1]' and Y in '[filename2]', what common information (like date, ID, category) should I use to link them?". Only use the tools for comparison after receiving clarification or if the linking information is explicitly provided in the original query. For non-comparison queries, proceed directly.
5.  **Execute Action:** Call the selected tool with the planned input.
6.  **Observe Result:** Analyze the output received from the tool.
7.  **Synthesize Answer:** If you have enough information, formulate the final answer. If not, repeat steps 2-6 (e.g., using another tool or refining the query for the same tool).
8.  **Final Response:** Provide the answer to the user. If data comes from specific files, mention them (e.g., "Based on 'report.pdf'..." or "According to 'sales.csv'...")."""

    try:
        agent = ReActAgent.from_tools(
            tools=agent_tools,
            llm=Settings.llm, # Use globally configured LLM
            verbose=True, # Log agent's thought process to console
            system_prompt=system_prompt
        )
        logger.info("ReActAgent created successfully.")
        st.success("🎉 Agent is ready!")
        end_time = datetime.datetime.now()
        st.caption(f"Agent setup took {(end_time - start_time).total_seconds():.2f} seconds.")
        # Store processed filenames along with the agent
        st.session_state.processed_filenames = processed_filenames
        return agent
    except Exception as e:
        logger.error(f"Failed to create ReActAgent: {e}", exc_info=True)
        st.error(f"Agent creation failed: {e}")
        return None

# ==============================================================================
# --- Streamlit App UI ---
# ==============================================================================

st.set_page_config(page_title="Multi-Doc/CSV Agent", layout="wide")
st.title("📄📊 Multi-Document & CSV Analysis Agent")
st.markdown("""
Upload one or more PDF and/or CSV files. The agent will use specialized tools
(Vector Search for PDFs, SQL Querying for CSVs) to answer your questions across the uploaded data.

**Note:** Uses a dummy LLM for structure demonstration - answers will be placeholders.
""")

# Initialize Qdrant Client (cached) - check if successful
qdrant_client_instance = get_qdrant_client()


# --- Sidebar for Uploads and Processing ---
with st.sidebar:
    st.header("1. Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF and/or CSV files",
        type=['pdf', 'csv'],
        accept_multiple_files=True,
        key="file_uploader"
    )

    st.header("2. Process Files")
    process_button_disabled = not uploaded_files # Disable if no files are uploaded

    if st.button("Process Files & Build Agent", type="primary", disabled=process_button_disabled, key="process_button"):
        # Clear previous agent and messages on reprocessing
        if 'agent' in st.session_state:
            del st.session_state['agent']
            logger.info("Cleared previous agent from session state.")
        if 'messages' in st.session_state:
            del st.session_state['messages']
            logger.info("Cleared previous messages from session state.")
        st.session_state.messages = [] # Initialize fresh message list
        if 'processed_filenames' in st.session_state:
             del st.session_state.processed_filenames
        st.session_state.process_button_clicked = True # Mark that processing was attempted


        if uploaded_files:
            # Separate files by type
            pdf_files_to_process = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
            csv_files_to_process = [f for f in uploaded_files if f.name.lower().endswith('.csv')]

            logger.info(f"Processing {len(pdf_files_to_process)} PDF(s) and {len(csv_files_to_process)} CSV(s).")

            # Call the main setup function
            agent_instance = setup_agent(
                uploaded_pdf_files=pdf_files_to_process,
                uploaded_csv_files=csv_files_to_process,
                qdrant_client_instance=qdrant_client_instance
            )

            # Store the result in session state
            st.session_state.agent = agent_instance # Will be None if setup failed

            if agent_instance is None:
                 # Error messages should already be shown by setup_agent
                 logger.error("Agent setup returned None.")
            else:
                 logger.info("Agent successfully created and stored in session state.")

        else:
            st.warning("Please upload at least one PDF or CSV file.")

# --- Display Config Info ---
st.divider()
st.header("Configuration Info")
st.info(f"LLM: {LLM_MODEL_NAME} (Custom Dummy)")
st.info(f"Embedding: {EMBEDDING_MODEL_NAME} ({EXPECTED_EMBEDDING_DIM} dims)")
if qdrant_client_instance:
    st.info(f"Vector Store: Qdrant (Prefix: {QDRANT_PDF_COLLECTION_PREFIX})")
    st.caption(f"Qdrant Path: {os.path.abspath(QDRANT_PERSIST_DIR)}")
else:
    st.warning("Vector Store: Qdrant Client Failed!")

st.info(f"CSV DB: SQLite")
st.caption(f"DB Path: {os.path.abspath(SQL_DB_PATH)}")

# --- Main Chat Interface ---
st.header("💬 Chat with the Agent")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get the agent from session state
agent = st.session_state.get('agent', None)
processed_files_list = st.session_state.get('processed_filenames', [])

# Determine if chat input should be disabled
chat_input_disabled = agent is None
if not uploaded_files:
    st.info("👈 Upload files using the sidebar to get started.")
elif agent is None and st.session_state.get('process_button_clicked', False): # Check if button was clicked but agent failed
    st.warning("Agent initialization failed. Please check logs or try reprocessing. Chat is disabled.")
elif agent is None:
    st.info("👈 Click 'Process Files & Build Agent' in the sidebar to enable chat.")

# Get user input
if prompt := st.chat_input("Ask a question about the uploaded files...", key="chat_prompt", disabled=chat_input_disabled):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Use a placeholder for streaming-like effect
        full_response_content = ""
        logger.info(f"--- Sending query to agent: {prompt} ---")
        with st.spinner("Agent thinking... 🤔"):
            try:
                # Use agent.chat for conversational interaction
                response = agent.chat(prompt)
                full_response_content = str(response) # Convert response object to string

            except Exception as e:
                logger.error(f"Error during agent chat: {e}", exc_info=True)
                # Use traceback.format_exc() for more detailed error logging if needed
                # logger.error(f"Traceback: {traceback.format_exc()}")
                st.error(f"An error occurred while processing your request: {e}")
                full_response_content = f"Sorry, I encountered an error: {e}"

        logger.info(f"--- Agent raw response received (full): --- \n{full_response_content}\n --- End Raw Response ---")
        message_placeholder.markdown(full_response_content) # Display the full response

    # Add the assistant's response to the history
    st.session_state.messages.append({"role": "assistant", "content": full_response_content})

# Logic to track if process button was clicked (moved inside the button's 'if' block)
# if 'process_button' in st.session_state and st.session_state.process_button:
#     st.session_state.process_button_clicked = True
# else:
    # Reset if the button wasn't the last interaction maybe? Or handle differently.
    # This helps distinguish between first load and failed processing state.
    # pass
```

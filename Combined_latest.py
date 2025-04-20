
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
    # SQLDatabase, # Not directly used by custom SQL, but import is harmless
    QueryBundle,
    PromptTemplate,
    # SimpleDirectoryReader, # Not needed for direct upload processing
)
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import BaseQueryEngine, SubQuestionQueryEngine # Import SubQuestionQueryEngine
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
QDRANT_PERSIST_DIR = "./qdrant_storage_unified_subq"
# Using a prefix for PDF collections for easier cleanup and uniqueness per PDF
QDRANT_PDF_COLLECTION_PREFIX = "unified_subq_pdf_coll_"
SQL_DB_DIR = "./sql_database_unified_subq"
SQL_DB_FILENAME = "csv_data_unified_subq.db"
SQL_DB_PATH = os.path.join(SQL_DB_DIR, SQL_DB_FILENAME)
SQL_DB_URL = f"sqlite:///{SQL_DB_PATH}"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384
LLM_MODEL_NAME = "custom_abc_llm" # Placeholder name

# Create directories if they don't exist
os.makedirs(QDRANT_PERSIST_DIR, exist_ok=True)
os.makedirs(SQL_DB_DIR, exist_ok=True)

# Setup Logging
# Check if logger already has handlers to prevent duplicate logs in Streamlit reruns
if not logging.getLogger(__name__).hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# --- Helper Functions for CSV Description (From Script 2) ---
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
        # Prioritize string if mixed, then float, then int
        if "string" in types: type_ = "string"; data_proc = [str(x) for x in valid_data]
        elif "float" in types: type_ = "float"; data_proc = np.array([float(x) for x in valid_data])
        else: type_ = "int"; data_proc = np.array([int(x) for x in valid_data])
        description += "\"Type\": \"" + type_ + "\", "
        if type_ in ["int", "float"]:
            # Ensure data_proc is not empty before calling min/max
            if data_proc.size > 0:
                 min_ = data_proc.min(); max_ = data_proc.max();
                 description += "\"MIN\": " + str(min_) + ", \"MAX\": " + str(max_)
            else:
                 description += "\"MIN\": null, \"MAX\": null" # Handle empty numeric column after dropna
        elif type_ == "string":
            # Safely handle potential non-string values before string operations
            values = list(set(["\"" + str(x).strip().replace('"',"'") + "\"" for x in data_proc]))
            random.shuffle(values);
            if len(values) > 15: values = values[:random.randint(5, 10)] # Limit samples
            numerates = ", ".join(values); description += "\"Sample Values\": [" + numerates + "]"
        description += ", \"Contains NaN\": " + str(post_len != pre_len); return description + "}"
    # Handle potential errors during description generation for a column
    columns_dec = []
    for c in df.columns:
        try:
            desc = single_des(c, df[c])
            if desc: # Only add if description was successfully generated
                columns_dec.append(desc)
        except Exception as e:
            logger.warning(f"Could not generate description for column '{c}': {e}")
            columns_dec.append("{\"Column Name\": \"" + str(c) + "\", \"Error\": \"Could not generate description\"}")

    random.shuffle(columns_dec)
    return "\n".join(columns_dec)

def generate_table_description(df: pd.DataFrame, table_name: str, source_csv_name: str) -> str:
    """Generates description for a table based on a DataFrame."""
    try:
        rows_count, columns_count = df.shape
        description = f"Table Name: '{table_name}' (derived from CSV: '{source_csv_name}')\n"
        description += f"Contains {rows_count} rows and {columns_count} columns.\n"
        # Use original columns for description context if available before renaming
        # However, the function expects the df passed to it, which might be the renamed one.
        # Let's assume df passed might be renamed, so we use its columns.
        description += f"SQL Table Columns: {', '.join(df.columns)}\n" # Use columns of the df passed in
        description += f"--- Column Details and Sample Data ---\n"
        col_descriptions = column_des(df) # Use the columns from the DataFrame passed
        description += col_descriptions if col_descriptions else "No detailed column descriptions generated."
        return description
    except Exception as e:
        logger.error(f"Failed to generate table description for {table_name} from {source_csv_name}: {e}", exc_info=True)
        return f"Error generating description for table '{table_name}'. Columns: {', '.join(df.columns)}. Error: {e}"

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
    # Handle empty or potentially problematic names after sanitization
    if not name or name in ["_", "__"]: # Added check for just underscores
        name = f"file_{random.randint(1000, 9999)}"
    return name


# ==============================================================================
# --- Custom LLM Implementation (Identical in both scripts) ---
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

    # --- Streaming Stubs ---
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        logger.warning("MyCustomLLM: stream_chat() called - Returning single response")
        # Simulate single response for non-streaming agent
        yield self.chat(messages, **kwargs)
        # raise NotImplementedError("Streaming chat not supported by MyCustomLLM.")

    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        logger.warning("MyCustomLLM: stream_complete() called - Returning single response")
        # Simulate single response for non-streaming agent
        yield self.complete(prompt, **kwargs)
        # raise NotImplementedError("Streaming complete not supported by MyCustomLLM.")

    # --- Async Streaming Stubs (Required by ABC but not implemented) ---
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        logger.warning("MyCustomLLM: astream_chat() called - NotImplementedError")
        raise NotImplementedError("Async streaming chat not supported by MyCustomLLM.")
        yield # Required for AsyncGenerator type hint

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        logger.warning("MyCustomLLM: astream_complete() called - NotImplementedError")
        raise NotImplementedError("Async streaming complete not supported by MyCustomLLM.")
        yield # Required for AsyncGenerator type hint

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=LLM_MODEL_NAME, is_chat_model=True) # Assuming it behaves like a chat model

# ==============================================================================
# --- Custom SQL Engine (From Script 2) ---
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
            # Ensure table exists before reflecting, handle potential reflection error
            inspector = sqlalchemy.inspect(self.sql_engine)
            if not inspector.has_table(self.table_name):
                 return f"Error: Table '{self.table_name}' does not exist in the database."

            metadata.reflect(bind=self.sql_engine, only=[self.table_name])
            table = metadata.tables.get(self.table_name)

            if not table: # Should be redundant after has_table check, but safe
                return f"Error: Could not find table '{self.table_name}' after reflection."

            columns = []
            for column in table.columns:
                col_type = str(column.type)
                constraints = []
                if column.primary_key: constraints.append("PRIMARY KEY")
                if not column.nullable: constraints.append("NOT NULL")
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                columns.append(f"  - \"{column.name}\": {col_type}{constraint_str}")

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
                 # Allow CREATE TEMPORARY TABLE specifically if needed? No, keep it simple.
                 logger.warning(f"SQL Safety Check: Found potentially dangerous keyword matching '{pattern}' in: {sql}")
                 return False
        return True

    def _format_results(self, result_df: pd.DataFrame) -> str:
        """Format query results into a readable string."""
        if result_df.empty:
            return "The query returned no results."

        max_rows_to_show = 20
        max_cols_to_show = 15
        original_shape = result_df.shape

        df_to_show = result_df
        show_cols_truncated = False
        if original_shape[1] > max_cols_to_show:
            df_to_show = df_to_show.iloc[:, :max_cols_to_show]
            show_cols_truncated = True

        show_rows_truncated = False
        if original_shape[0] > max_rows_to_show:
            df_to_show = df_to_show.head(max_rows_to_show)
            show_rows_truncated = True

        result_str = ""
        if show_rows_truncated or show_cols_truncated:
             result_str += f"Query returned {original_shape[0]} rows and {original_shape[1]} columns. "
             parts = []
             if show_rows_truncated: parts.append(f"first {max_rows_to_show} rows")
             if show_cols_truncated: parts.append(f"first {max_cols_to_show} columns")
             result_str += f"Showing { ' and '.join(parts)}:\n\n"
        else:
             result_str += "Query Result:\n\n"

        # Use .to_markdown for better formatting in Streamlit
        try:
            markdown_result = df_to_show.to_markdown(index=False)
        except Exception as md_err:
            logger.error(f"Error converting DataFrame to Markdown: {md_err}")
            markdown_result = df_to_show.to_string(index=False) # Fallback to string

        return result_str + markdown_result

    def _execute_sql(self, sql: str) -> Tuple[bool, Union[pd.DataFrame, str]]:
        """Execute SQL query and return results or error message."""
        try:
            if not self._is_safe_sql(sql):
                logger.error(f"SQL Safety Check Failed for query: {sql}")
                return False, "SQL query failed safety check (only SELECT queries are allowed)."

            if self.verbose: logger.info(f"Executing safe SQL on table {self.table_name}: {sql}")
            with self.sql_engine.connect() as connection:
                 # Using a timeout with connect() might be driver specific,
                 # pd.read_sql_query doesn't have a universal timeout.
                 result_df = pd.read_sql_query(sql, connection)
            return True, result_df

        except sqlalchemy.exc.SQLAlchemyError as db_err:
            error_msg = f"Database Error: {db_err}"
            if self.verbose: logger.error(f"SQL execution error: {error_msg}\nQuery: {sql}", exc_info=False) # Reduce noise
            return False, error_msg
        except Exception as e:
            error_msg = f"General Error executing SQL: {e}"
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

            if attempt == max_retries:
                logger.error(f"SQL failed after {max_retries+1} attempts. Final error: {error_message}")
                return False, f"SQL execution failed: {error_message}", current_sql

            # --- Try to fix the query ---
            if self.verbose: logger.info("Attempting to fix SQL using LLM...")
            try:
                schema_info = self._get_schema_info()
                fix_prompt = self.sql_fix_prompt.format(
                    failed_sql=current_sql,
                    error_msg=error_message,
                    table_name=self.table_name,
                    table_description=f"{self.table_description}\n\n{schema_info}"
                )
                fixed_sql_response = self.llm_callback(fix_prompt)
                fixed_sql = self._clean_sql(fixed_sql_response)

                if fixed_sql and fixed_sql.lower() != current_sql.lower():
                    current_sql = fixed_sql
                    if self.verbose: logger.info(f"LLM proposed fixed SQL: {current_sql}")
                    # Loop will continue and try executing this new query
                else:
                    if self.verbose: logger.warning("LLM did not provide a different SQL query or failed to fix. Stopping retries.")
                    return False, f"SQL execution failed: {error_message} (LLM correction failed)", original_sql
            except Exception as fix_error:
                logger.error(f"Error during LLM SQL fix attempt: {fix_error}", exc_info=True)
                return False, f"SQL execution failed: {error_message} (LLM fix attempt failed: {fix_error})", original_sql

        # Should not be reached
        return False, "Unexpected error during SQL execution loop", original_sql

    def query(self, query_text: str) -> str:
        """Process a natural language query and return results as a formatted string."""
        if self.verbose: logger.info(f"CustomSQLEngine received query for table {self.table_name}: {query_text}")
        try:
            schema_info = self._get_schema_info()
            if "Error:" in schema_info: # Check if schema retrieval failed
                 return f"Error: Could not retrieve schema for table '{self.table_name}'. Cannot proceed."

            if self.verbose: logger.info("Generating initial SQL query from natural language...")
            generate_prompt = self.sql_prompt_template.format(
                query_str=query_text,
                table_name=self.table_name,
                table_description=f"{self.table_description}\n\n{schema_info}"
            )

            sql_response = self.llm_callback(generate_prompt)
            # Check if LLM callback returned an error indicator
            if sql_response.startswith("LLM Error:"):
                 logger.error(f"LLM callback failed: {sql_response}")
                 return f"Error: Failed to get response from language model for SQL generation. {sql_response}"

            sql_query = self._clean_sql(sql_response)
            if not sql_query:
                logger.error("LLM failed to generate any SQL query.")
                return "Error: Could not generate SQL query from your question."
            if self.verbose: logger.info(f"LLM generated SQL: {sql_query}")

            success, result, final_sql = self._execute_with_retry(sql_query)

            if not success:
                logger.error(f"Final SQL execution failed. Error: {result}")
                # Result is the error message string here
                return f"I encountered an error trying to query the data for table '{self.table_name}':\n{result}\n\nSQL attempted:\n```sql\n{final_sql}\n```"

            # --- Execution Succeeded ---
            # Result is the pandas DataFrame here
            if self.verbose: logger.info(f"SQL query successful. Formatting results from DataFrame.")
            formatted_results = self._format_results(result) # result is DataFrame

            response = (
                f"Executed SQL query on table '{self.table_name}' and found the following:\n\n"
                f"{formatted_results}\n\n"
                # Intentionally removing the SQL query from the final user output for cleaner interface
                # f"SQL query used: ```sql\n{final_sql}\n```"
            )
            return response

        except Exception as e:
            logger.error(f"Unexpected error in CustomSQLEngine.query for table {self.table_name}: {e}", exc_info=True)
            return f"An unexpected error occurred while processing your query for table '{self.table_name}': {e}"

# ==============================================================================
# --- Custom SQL Engine Wrapper for LlamaIndex ---
# ==============================================================================
# Ensure this inherits from BaseQueryEngine
class CustomSQLQueryEngineWrapper(BaseQueryEngine):
    """Adapter to make CustomSQLEngine compatible with LlamaIndex tools."""
    def __init__(self, engine: CustomSQLEngine, llm: Optional[LLM] = None):
        self._engine = engine
        # Ensure necessary attributes for BaseQueryEngine are present
        # Attempt to get LLM from Settings if not provided
        self._llm = llm or Settings.llm
        if not self._llm:
             # Fallback or raise error if LLM is crucial and not found
             logger.warning("LLM not provided to CustomSQLQueryEngineWrapper and not found in Settings. Using a default might fail.")
             # self._llm = SomeDefaultLLM() # Or raise error
             raise ValueError("LLM must be available via argument or Settings for CustomSQLQueryEngineWrapper")

        # Initialize base class - callback_manager might be optional depending on usage
        super().__init__(llm=self._llm, callback_manager=None)

    def _get_prompt_modules(self) -> Dict[str, Any]:
         """Get prompt sub-modules."""
         # Required by BaseQueryEngine potentially
         return {}

    @property
    def llm(self) -> LLM:
         return self._llm

    # Note: synthesizer might not be strictly needed if _query returns final string
    # but implementing it satisfies the abstract property if required by base class.
    @property
    def synthesizer(self) -> BaseSynthesizer:
         """Get the response synthesizer."""
         return get_response_synthesizer(llm=self._llm, response_mode=ResponseMode.NO_TEXT) # Use NO_TEXT if _query returns final string

    def _query(self, query_bundle: QueryBundle) -> str: # Return type hint changed back to str for consistency
        """Main query interface called by LlamaIndex."""
        logger.info(f"CustomSQLQueryEngineWrapper: _query called for table {self._engine.table_name}")
        query_text = query_bundle.query_str
        # Call the custom engine's query method which returns a formatted string
        result_str = self._engine.query(query_text)
        # Wrap the final string in a Response object as expected by LlamaIndex >= 0.10
        # If using older LlamaIndex that expects string, return result_str directly.
        # Let's return string for now based on previous context, but Response is safer.
        # from llama_index.core.base.response.schema import Response
        # return Response(response=result_str)
        return result_str # Return formatted string directly

    async def _aquery(self, query_bundle: QueryBundle) -> str: # Return type hint changed back to str
        """Async query interface."""
        logger.info(f"CustomSQLQueryEngineWrapper: _aquery called for table {self._engine.table_name} - using sync query")
        import asyncio
        loop = asyncio.get_running_loop()
        # Run the synchronous query method in a thread pool executor
        result = await loop.run_in_executor(None, self._query, query_bundle)
        return result

# ==============================================================================
# --- Global Settings and Initialization ---
# ==============================================================================
@st.cache_resource
def get_llm():
    logger.info("Initializing Custom LLM (cached)...")
    return MyCustomLLM()

@st.cache_resource
def get_embed_model():
    logger.info(f"Initializing Embedding Model (cached): {EMBEDDING_MODEL_NAME}")
    try:
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        test_embedding = embed_model.get_query_embedding("test")
        actual_dim = len(test_embedding)
        if actual_dim != EXPECTED_EMBEDDING_DIM:
            raise ValueError(f"Embed dim mismatch! Expected {EXPECTED_EMBEDDING_DIM}, Got {actual_dim}.")
        logger.info(f"Embedding model loaded successfully. Dimension: {actual_dim}")
        return embed_model
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize embedding model: {e}", exc_info=True)
        st.error(f"Embedding model initialization failed: {e}")
        return None

@st.cache_resource
def get_qdrant_client() -> Optional[qdrant_client.QdrantClient]:
    logger.info(f"Initializing Qdrant client (cached) (Path: {QDRANT_PERSIST_DIR})...")
    try:
        client = qdrant_client.QdrantClient(path=QDRANT_PERSIST_DIR)
        client.get_collections() # Quick check if operational
        logger.info("Qdrant client initialized successfully.")
        return client
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize Qdrant client: {e}", exc_info=True)
        st.error(f"Qdrant vector database initialization failed: {e}. PDF analysis may be unavailable.")
        return None

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
        Settings.node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=200)
        Settings.num_output = 512
        Settings.context_window = 4096 # Adjust based on the real LLM

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
    # Using the improved version from combined script discussion
    if not uploaded_file: return None
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
                        text_content += page_text + "\n\n"
                    else:
                        logger.warning(f"No text found on page {page_num + 1}/{num_pages} of '{file_name}'.")
                except Exception as page_err:
                    logger.error(f"Error extracting text from page {page_num + 1} of '{file_name}': {page_err}", exc_info=True)
                    text_content += f"[Error reading page {page_num + 1}]\n\n"
        if not text_content.strip():
            st.warning(f"No text could be extracted from PDF '{file_name}'. It might be image-based or corrupted.")
            return None
        doc = Document(text=text_content, metadata={"file_name": file_name})
        logger.info(f"Successfully extracted text from PDF '{file_name}'. Length: {len(text_content)}")
        return doc
    except Exception as e:
        logger.error(f"Failed to process PDF '{file_name}': {e}", exc_info=True)
        st.error(f"Error reading PDF file '{file_name}': {e}")
        return None

def process_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[pd.DataFrame]:
    """Processes a single uploaded CSV file into a pandas DataFrame."""
    # Using the improved version from combined script discussion
    if not uploaded_file: return None
    file_name = uploaded_file.name
    logger.info(f"Processing CSV: {file_name}")
    try:
        try: # Try UTF-8
            df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        except UnicodeDecodeError: # Fallback to latin1
            logger.warning(f"UTF-8 decoding failed for '{file_name}'. Attempting latin1.")
            df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), encoding='latin1')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Remove unnamed columns
        logger.info(f"Loaded CSV '{file_name}'. Shape: {df.shape}")
        if df.empty:
            st.warning(f"CSV file '{file_name}' is empty.")
            return None
        return df
    except Exception as e:
        logger.error(f"Failed to process CSV '{file_name}': {e}", exc_info=True)
        st.error(f"Error reading CSV file '{file_name}': {e}")
        return None

# ==============================================================================
# --- Tool Creation Functions (Combined Logic - Unique per file) ---
# ==============================================================================
def create_pdf_tool(
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
    collection_name = f"{QDRANT_PDF_COLLECTION_PREFIX}{sanitized_name}" # UNIQUE collection name
    tool_name = f"pdf_{sanitized_name}_tool" # Use 'tool' suffix for consistency

    logger.info(f"Creating tool for PDF: '{file_name}' (Collection: {collection_name}, Tool: {tool_name})")
    try:
        # 1. Create unique Qdrant Collection (handle potential existence)
        logger.info(f"Attempting creation of Qdrant collection: '{collection_name}'")
        try:
             qdrant_client_instance.create_collection(
                 collection_name=collection_name,
                 vectors_config=VectorParams(size=EXPECTED_EMBEDDING_DIM, distance=Distance.COSINE)
             )
             logger.info(f"Collection '{collection_name}' created.")
        except Exception as create_exc:
             logger.warning(f"Could not create collection '{collection_name}' (may exist): {create_exc}. Checking...")
             try:
                 qdrant_client_instance.get_collection(collection_name=collection_name)
                 logger.info(f"Confirmed collection '{collection_name}' exists.")
             except Exception as get_exc:
                 logger.error(f"Failed to ensure Qdrant collection '{collection_name}': {get_exc}", exc_info=True)
                 st.error(f"Qdrant error for '{file_name}': {get_exc}")
                 return None

        # 2. Setup LlamaIndex components for this collection
        vector_store = QdrantVectorStore(client=qdrant_client_instance, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info(f"Storage context ready for '{collection_name}'.")

        # 3. Index the document into its collection (using global Settings)
        logger.info(f"Indexing document '{file_name}' into '{collection_name}'...")
        index = VectorStoreIndex.from_documents(
            [pdf_document], storage_context=storage_context, show_progress=True
        )
        logger.info(f"Document '{file_name}' indexed into '{collection_name}'.")

        # 4. Create query engine for this specific index (using global Settings)
        pdf_query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
        logger.info(f"Query engine created for index '{collection_name}'.")

        # 5. Create the QueryEngineTool
        tool_description = f"""Provides information from the specific PDF document named '{file_name}'. Use this tool for any questions requiring text search, summarization, or content understanding within '{file_name}'."""
        pdf_tool = QueryEngineTool(
            query_engine=pdf_query_engine,
            metadata=ToolMetadata(name=tool_name, description=tool_description)
        )
        logger.info(f"QueryEngineTool '{tool_name}' created for PDF '{file_name}'.")
        return pdf_tool
    except Exception as e:
        logger.error(f"Failed to create PDF tool for '{file_name}': {e}", exc_info=True)
        st.error(f"Error setting up PDF tool for '{file_name}': {e}")
        return None

def create_csv_tool(
    df: pd.DataFrame,
    csv_file_name: str,
    sql_alchemy_engine: sqlalchemy.engine.Engine # Pass the shared engine
) -> Optional[QueryEngineTool]:
    """Creates a QueryEngineTool for a CSV DataFrame using the CustomSQLEngine."""
    if df is None or df.empty:
        st.error(f"CSV tool creation failed: DataFrame is empty for '{csv_file_name}'.")
        return None
    if not sql_alchemy_engine:
        st.error(f"CSV tool creation failed: SQLAlchemy Engine not provided for '{csv_file_name}'.")
        return None

    sanitized_base_name = sanitize_for_name(csv_file_name)
    table_name = f"csv_tbl_{sanitized_base_name}" # Unique table name
    tool_name = f"csv_{sanitized_base_name}_tool" # Unique tool name

    logger.info(f"Creating tool for CSV: '{csv_file_name}' (Table: {table_name}, Tool: {tool_name})")
    try:
        # --- Prepare DataFrame ---
        original_columns = df.columns.tolist()
        logger.info(f"Original CSV columns: {original_columns}")
        cleaned_column_map = {}
        seen_cleaned_names = set()
        for i, col in enumerate(df.columns):
            cleaned_col = re.sub(r'\W+|^(?=\d)', '_', str(col)).lower().strip('_')
            cleaned_col = cleaned_col or f"column_{i}"
            final_cleaned_col = cleaned_col
            suffix = 1
            while final_cleaned_col in seen_cleaned_names:
                final_cleaned_col = f"{cleaned_col}_{suffix}"
                suffix += 1
            seen_cleaned_names.add(final_cleaned_col)
            cleaned_column_map[col] = final_cleaned_col
        df_renamed = df.rename(columns=cleaned_column_map)
        cleaned_columns = df_renamed.columns.tolist()
        logger.info(f"Cleaned SQL columns: {cleaned_columns}")
        dtype_mapping = {col: sqlalchemy.types.TEXT for col in df_renamed.select_dtypes(include=['object', 'string']).columns}

        # --- Load to SQL ---
        logger.info(f"Saving DataFrame {df_renamed.shape} to SQL table '{table_name}'...")
        df_renamed.to_sql(
            name=table_name, con=sql_alchemy_engine, index=False,
            if_exists='replace', chunksize=1000, dtype=dtype_mapping
        )
        logger.info(f"DataFrame saved to table '{table_name}'.")

        # --- Generate Description ---
        logger.info(f"Generating description for table '{table_name}'...")
        table_desc = generate_table_description(df_renamed, table_name, csv_file_name) # Use renamed df
        logger.info(f"Description generated for table '{table_name}'.")

        # --- Instantiate Custom SQL Engine ---
        logger.info(f"Instantiating CustomSQLEngine for table '{table_name}'...")
        def llm_callback(prompt: str) -> str:
            try:
                if not Settings.llm: raise ValueError("Global LLM not configured.")
                return Settings.llm.complete(prompt).text
            except Exception as e:
                logger.error(f"Error in LLM callback for CustomSQLEngine: {e}", exc_info=True)
                return f"LLM Error: {e}"
        custom_sql_engine_instance = CustomSQLEngine(
            sql_engine=sql_alchemy_engine, table_name=table_name,
            llm_callback=llm_callback, table_description=table_desc, verbose=True
        )

        # --- Wrap Custom Engine ---
        wrapped_engine = CustomSQLQueryEngineWrapper(custom_sql_engine_instance)
        logger.info(f"CustomSQLEngine wrapped successfully for table '{table_name}'.")

        # --- Create Final Tool ---
        tool_description = f"""Queries a SQL table named '{table_name}' derived from CSV '{csv_file_name}'. Use for structured data lookup, filtering, calculations (SUM, COUNT, AVG), or aggregation. Available SQL columns: {', '.join(cleaned_columns)}."""
        csv_tool = QueryEngineTool(
            query_engine=wrapped_engine,
            metadata=ToolMetadata(name=tool_name, description=tool_description)
        )
        logger.info(f"QueryEngineTool '{tool_name}' created for CSV '{csv_file_name}'.")
        return csv_tool
    except sqlalchemy.exc.SQLAlchemyError as db_err:
        logger.error(f"Database error for CSV '{csv_file_name}' (table '{table_name}'): {db_err}", exc_info=True)
        st.error(f"Database error processing CSV '{csv_file_name}': {db_err}")
        return None
    except Exception as e:
        logger.error(f"Failed to create CSV tool for '{csv_file_name}' (table '{table_name}'): {e}", exc_info=True)
        st.error(f"Error setting up CSV tool for '{csv_file_name}': {e}")
        return None

# ==============================================================================
# --- Main Engine Setup Function (Uses SubQuestionQueryEngine) ---
# ==============================================================================
def setup_engine(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    qdrant_client_instance: qdrant_client.QdrantClient
) -> Tuple[Optional[SubQuestionQueryEngine], List[str]]:
    """Processes files, creates tools (1 per file), builds SubQuestionQueryEngine."""
    st.info("🚀 Starting engine setup...")
    start_time = datetime.datetime.now()
    agent_tools = []
    processed_filenames = []

    # 1. Configure Global Settings
    if not configure_global_settings():
        st.error("Engine setup failed: Core component configuration error.")
        logger.error("Engine setup aborted: Settings configuration failure.")
        return None, []

    # 2. Separate Files
    pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
    csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]
    logger.info(f"Found {len(pdf_files)} PDF(s) and {len(csv_files)} CSV(s).")

    # 3. Cleanup Resources
    # SQLite DB
    if os.path.exists(SQL_DB_PATH):
        try:
            logger.warning(f"Removing existing SQLite DB: {SQL_DB_PATH}")
            os.remove(SQL_DB_PATH)
            logger.info("Old DB file removed.")
        except OSError as e:
            logger.error(f"Error removing DB file '{SQL_DB_PATH}': {e}", exc_info=True)
            st.error(f"Could not remove old DB file: {e}") # Non-fatal for now

    # Qdrant Collections
    if qdrant_client_instance:
        logger.warning(f"Cleaning up old Qdrant collections (prefix '{QDRANT_PDF_COLLECTION_PREFIX}')...")
        cleaned_count = 0
        try:
            collections = qdrant_client_instance.get_collections().collections
            for collection in collections:
                if collection.name.startswith(QDRANT_PDF_COLLECTION_PREFIX):
                    logger.info(f"Deleting old Qdrant collection: {collection.name}")
                    try:
                        qdrant_client_instance.delete_collection(collection_name=collection.name, timeout=60)
                        cleaned_count += 1
                    except Exception as del_exc:
                         logger.error(f"Failed to delete Qdrant collection '{collection.name}': {del_exc}")
                         st.warning(f"Could not delete old collection '{collection.name}'.")
            logger.info(f"Qdrant cleanup finished. Deleted {cleaned_count} collections.")
        except Exception as list_exc:
            logger.error(f"Failed to list Qdrant collections for cleanup: {list_exc}", exc_info=True)
            st.warning("Could not perform Qdrant cleanup.")
    else:
        logger.warning("Qdrant client not available, skipping Qdrant cleanup.")


    # 4. Process PDFs and Create Tools
    pdf_success_count = 0
    if pdf_files:
        st.write(f"Processing {len(pdf_files)} PDF file(s)...")
        if not qdrant_client_instance:
             st.error("Cannot process PDFs: Qdrant client failed.")
        else:
            pdf_progress = st.progress(0)
            for i, uploaded_pdf in enumerate(pdf_files):
                progress_text_pdf = f"Processing PDF: {uploaded_pdf.name} ({i+1}/{len(pdf_files)})..."
                pdf_progress.progress((i / len(pdf_files)), text=progress_text_pdf)
                try:
                    pdf_doc = process_pdf(uploaded_pdf)
                    if pdf_doc:
                        pdf_tool = create_pdf_tool(pdf_doc, qdrant_client_instance)
                        if pdf_tool:
                            agent_tools.append(pdf_tool)
                            processed_filenames.append(uploaded_pdf.name)
                            pdf_success_count += 1
                        # Error logging/display done within create_pdf_tool
                except Exception as pdf_loop_err:
                    logger.error(f"Error in PDF processing loop for {uploaded_pdf.name}: {pdf_loop_err}", exc_info=True)
                    st.error(f"Failed processing {uploaded_pdf.name}: {pdf_loop_err}")
            pdf_progress.progress(1.0, text="PDF Processing Complete.")
            # Consider removing progress bar after completion: pdf_progress.empty()


    # 5. Process CSVs and Create Tools
    csv_success_count = 0
    sql_alchemy_engine = None # Initialize
    if csv_files:
        st.write(f"Processing {len(csv_files)} CSV file(s)...")
        try:
            logger.info(f"Creating SQLAlchemy engine for SQLite: {SQL_DB_URL}")
            sql_alchemy_engine = sqlalchemy.create_engine(SQL_DB_URL)
            with sql_alchemy_engine.connect() as conn: logger.info("SQLAlchemy engine connected.") # Test connection

            csv_progress = st.progress(0)
            for i, uploaded_csv in enumerate(csv_files):
                 progress_text_csv = f"Processing CSV: {uploaded_csv.name} ({i+1}/{len(csv_files)})..."
                 csv_progress.progress((i / len(csv_files)), text=progress_text_csv)
                 try:
                     csv_df = process_csv(uploaded_csv)
                     if csv_df is not None:
                         csv_tool = create_csv_tool(csv_df, uploaded_csv.name, sql_alchemy_engine)
                         if csv_tool:
                             agent_tools.append(csv_tool)
                             processed_filenames.append(uploaded_csv.name)
                             csv_success_count += 1
                         # Error logging/display done within create_csv_tool
                 except Exception as csv_loop_err:
                     logger.error(f"Error in CSV processing loop for {uploaded_csv.name}: {csv_loop_err}", exc_info=True)
                     st.error(f"Failed processing {uploaded_csv.name}: {csv_loop_err}")
            csv_progress.progress(1.0, text="CSV Processing Complete.")
            # Consider removing progress bar after completion: csv_progress.empty()

        except Exception as db_eng_err:
             logger.error(f"Failed SQLAlchemy engine init: {db_eng_err}", exc_info=True)
             st.error(f"Database engine error: {db_eng_err}. CSV processing failed.")
             sql_alchemy_engine = None # Ensure unusable

    st.write(f"✅ Finished processing. Created {pdf_success_count} PDF tool(s), {csv_success_count} CSV tool(s).")

    # 6. Check if any tools were created
    if not agent_tools:
        st.error("Engine setup failed: No tools could be created.")
        logger.error("Engine setup aborted: No tools created.")
        return None, []

    st.success(f"Successfully created {len(agent_tools)} tools for {len(processed_filenames)} files.")
    logger.info(f"Total tools created: {len(agent_tools)}. Files: {processed_filenames}")

    # 7. Create the SubQuestionQueryEngine
    st.info("Building the main query engine...")
    logger.info("Attempting to create SubQuestionQueryEngine...")
    try:
        # Ensure Settings.llm is valid before creating engine
        if not Settings.llm:
             raise ValueError("LLM is not configured in Settings.")

        # SubQuestionQueryEngine uses the settings from the tools/query_engines and global Settings
        final_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=agent_tools,
            # service_context=None, # Not needed when using Settings
            use_async=False, # Keep synchronous for simplicity
            verbose=True
        )
        logger.info("SubQuestionQueryEngine created successfully.")
        st.success("🎉 Query Engine is ready!")
        end_time = datetime.datetime.now()
        st.caption(f"Engine setup took {(end_time - start_time).total_seconds():.2f} seconds.")
        return final_engine, processed_filenames # Return engine and filenames

    except Exception as e:
        logger.error(f"Failed to create SubQuestionQueryEngine: {e}", exc_info=True)
        st.error(f"Query Engine creation failed: {e}")
        return None, processed_filenames # Return None for engine, but keep filenames if tools were made

# ==============================================================================
# --- Streamlit App UI (Unified - Uses SubQuestionQueryEngine) ---
# ==============================================================================
st.set_page_config(page_title="Unified SubQuery Engine", layout="wide")
st.title("📄📊 Unified PDF & CSV Analysis Engine")
st.markdown("""
Upload PDF and/or CSV files. The engine uses vector search for PDFs (one index per PDF)
and a custom SQL engine for CSVs (one table per CSV), coordinated by a Sub-Question Query Engine.
""")

# --- Initialize Core Components & Session State ---
qdrant_client_instance = get_qdrant_client()
# Use more descriptive names in session state
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'processed_filenames' not in st.session_state:
    st.session_state.processed_filenames = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False # Track if processing finished

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF and/or CSV files",
        type=['pdf', 'csv'],
        accept_multiple_files=True,
        key="unified_file_uploader"
    )

    st.header("2. Process Files")
    process_button_disabled = not uploaded_files or qdrant_client_instance is None
    if qdrant_client_instance is None:
         st.error("Qdrant client failed. File processing disabled.")

    if st.button("Process Files & Build Engine", type="primary", disabled=process_button_disabled, key="unified_process_button"):
        # Clear previous state before processing
        st.session_state.query_engine = None
        st.session_state.chat_messages = []
        st.session_state.processed_filenames = []
        st.session_state.processing_complete = False

        if uploaded_files:
            # Call the main setup function
            engine_instance, processed_names = setup_engine(
                uploaded_files=uploaded_files,
                qdrant_client_instance=qdrant_client_instance
            )
            st.session_state.query_engine = engine_instance
            st.session_state.processed_filenames = processed_names
            st.session_state.processing_complete = True # Mark processing attempted/finished

            if engine_instance is None:
                 logger.error("Engine setup returned None.")
                 # Error messages displayed within setup_engine
            else:
                 logger.info("Engine successfully created and stored in session state.")
                 # Clear any lingering progress bars if they weren't cleared in setup
                 # This is harder without passing the progress objects back,
                 # relying on setup_engine showing final status.
        else:
            st.warning("Please upload at least one file.")

    # --- Display Config Info ---
    st.sidebar.divider()
    st.sidebar.header("Configuration Info")
    st.sidebar.info(f"LLM: {LLM_MODEL_NAME} (Custom Dummy)")
    st.sidebar.info(f"Embedding: {EMBEDDING_MODEL_NAME}")
    if qdrant_client_instance:
        st.sidebar.info(f"Vector Store: Qdrant (Prefix: {QDRANT_PDF_COLLECTION_PREFIX})")
        st.sidebar.caption(f"Qdrant Path: {os.path.abspath(QDRANT_PERSIST_DIR)}")
    else:
        st.sidebar.warning("Vector Store: Qdrant Client Failed!")
    st.sidebar.info(f"CSV DB: SQLite ({SQL_DB_FILENAME})")
    st.sidebar.caption(f"DB Path: {os.path.abspath(SQL_DB_PATH)}")


# --- Main Chat Area ---
st.header("💬 Chat with the Engine")

final_engine = st.session_state.get('query_engine', None)
processed_files_list = st.session_state.get('processed_filenames', [])
processing_done = st.session_state.get('processing_complete', False)

# Display initial guidance message AFTER processing is complete and successful
if processing_done and final_engine and not st.session_state.chat_messages: # Show only once if no messages yet
    with st.chat_message("assistant"):
        greeting_msg = f"Hello! I've processed the following files: `{', '.join(processed_files_list)}`.\n\n"
        if len(processed_files_list) > 2:
            greeting_msg += "**Tip:** Since multiple files are loaded, asking questions that mention specific filenames helps me answer faster and more accurately. For example:\n"
            # Try to find one example PDF and one CSV if available
            example_pdf = next((f for f in processed_files_list if f.lower().endswith('.pdf')), None)
            example_csv = next((f for f in processed_files_list if f.lower().endswith('.csv')), None)
            if example_pdf: greeting_msg += f"- 'Summarize `{example_pdf}`'\n"
            if example_csv: greeting_msg += f"- 'What is the total revenue in `{example_csv}`?'\n"
            if example_pdf and example_csv: greeting_msg += f"- 'Compare findings in `{example_pdf}` with data in `{example_csv}` using `[common topic/column]`'\n"
            greeting_msg += "\n" # Add spacing
        greeting_msg += "How can I help you analyze these documents?"
        st.markdown(greeting_msg)
        # Add this initial message to history so it persists
        st.session_state.chat_messages.append({"role": "assistant", "content": greeting_msg})


# Display chat history (including the initial message if added)
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Determine chat input status
chat_input_disabled = final_engine is None

# User guidance messages based on state
if not uploaded_files and not final_engine:
    st.info("👈 Upload files and click 'Process Files & Build Engine' to start.")
elif final_engine is None and processing_done:
    st.warning("Engine initialization failed. Please check logs or file processing messages. Chat is disabled.")
elif final_engine is None and not processing_done:
     st.info("👈 Click 'Process Files & Build Engine' in the sidebar to enable chat.")


# Chat Input Logic
if prompt := st.chat_input("Ask a question about the uploaded files...", key="unified_chat_prompt", disabled=chat_input_disabled):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Assistant Response Area ---
    with st.chat_message("assistant"):
        info_message_placeholder = st.empty() # Placeholder for pre-check info
        final_response_placeholder = st.empty() # Placeholder for actual response
        full_assistant_response_for_history = "" # Store combined response

        with st.spinner("Thinking... 🤔"):
            pre_check_info_msg = ""
            # Step 10: Pre-Processing LLM Check (Enhanced)
            try:
                if final_engine: # Ensure engine exists before proceeding
                    num_tools = len(final_engine.query_engine_tools) # Get tool count from engine
                    current_filenames = st.session_state.get('processed_filenames', [])

                    # Construct the enhanced check prompt
                    special_check_prompt = f"""Analyze the user query provided below based on the available files.

User Query:
{prompt}

Available Files:
{', '.join(current_filenames)}

Number of Files: {len(current_filenames)}

Instructions:
1. Check if the query mentions specific filenames from the 'Available Files' list.
2. Check if the query asks for a comparison between information from different files (e.g., uses words like 'compare', 'difference', 'vs', 'than').
3. If it IS a comparison, check if a clear linking factor (like a specific date, common column name, ID, category, or topic) is mentioned.

Determine if a warning or suggestion is needed:
- Rule 1: Is the query general (mentions no specific filenames) AND are there more than 2 files available?
- Rule 2: Is the query a comparison BUT lacks a clear linking factor?

Response Rules:
- If Rule 1 applies (and Rule 2 does not), respond ONLY with: "INFO: Your query seems general. Processing across all {len(current_filenames)} files might take longer. For focused results next time, try specifying files (e.g., 'Summarize `{current_filenames[0] if current_filenames else '[filename.pdf]'}`'). Proceeding with the general query now..."
- If Rule 2 applies (and Rule 1 does not), respond ONLY with: "INFO: Your comparison query doesn't specify how to link the data (e.g., by date, ID, category). I'll attempt the comparison based on the content, but for a more precise result next time, please include the linking factor (e.g., 'Compare `{current_filenames[0] if current_filenames else '[file1.pdf]'}` and `{current_filenames[1] if len(current_filenames)>1 else '[file2.csv]'}` for `[linking_factor]`). Proceeding with your query now..."
- If BOTH Rule 1 and Rule 2 apply, respond ONLY with: "INFO: Your query is general and the comparison basis is unclear. Processing across all {len(current_filenames)} files and attempting comparison may take longer and yield limited results. Please specify filenames and a linking factor next time for better results. Proceeding with your query now..."
- If NEITHER Rule 1 nor Rule 2 apply, respond ONLY with: "OK"
"""
                    if Settings.llm:
                        logger.info("Performing pre-query check with LLM...")
                        check_response = Settings.llm.complete(special_check_prompt)
                        # NOTE: MyCustomLLM will return its dummy response. A real LLM is needed for this check to function.
                        # For testing flow, we can simulate the check:
                        # is_vague = not any(fname in prompt for fname in current_filenames)
                        # is_comparison = any(word in prompt.lower() for word in ['compare', ' vs ', 'difference', ' than '])
                        # needs_link = is_comparison # Simplified: assume link always needed if comparing
                        # if is_vague and len(current_filenames) > 2: check_response_text = "INFO: Vague..."
                        # elif needs_link: check_response_text = "INFO: Comparison..."
                        # else: check_response_text = "OK"
                        check_response_text = check_response.text

                        if check_response_text != "OK" and check_response_text:
                            # Check against dummy response - if it's the dummy, treat as OK for now
                            if "This is a dummy response" in check_response_text:
                                 logger.warning("Pre-check LLM is a dummy, skipping dynamic warning.")
                                 pre_check_info_msg = "" # Don't show dummy response as warning
                            else:
                                 pre_check_info_msg = check_response_text # Store the actual helpful message
                                 info_message_placeholder.info(pre_check_info_msg) # Display immediately
                                 full_assistant_response_for_history += pre_check_info_msg + "\n\n" # Add to history log
                    else:
                         logger.warning("LLM not available for pre-query check.")

            except Exception as pre_check_err:
                logger.error(f"Error during pre-query check: {pre_check_err}", exc_info=True)
                # Don't block execution, just log the error

            # Step 12: Execute Main Engine
            final_engine_response_str = ""
            try:
                if final_engine:
                    logger.info(f"--- Querying SubQuestionQueryEngine: {prompt} ---")
                    response_obj = final_engine.query(prompt) # Pass original prompt
                    final_engine_response_str = str(response_obj) # Convert Response object to string
                    logger.info(f"--- SubQuestionQueryEngine response received ---")
                else:
                    final_engine_response_str = "Error: Query engine is not available."
                    st.error(final_engine_response_str)

            except Exception as query_err:
                logger.error(f"Error during SubQuestionQueryEngine query: {query_err}", exc_info=True)
                final_engine_response_str = f"Sorry, an error occurred while processing your query: {query_err}"
                st.error(final_engine_response_str)

        # Display final response
        final_response_placeholder.markdown(final_engine_response_str)
        full_assistant_response_for_history += final_engine_response_str # Add final answer to history log

    # Add the complete assistant turn to history
    st.session_state.chat_messages.append({"role": "assistant", "content": full_assistant_response_for_history})
  

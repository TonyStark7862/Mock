

import streamlit as st
import os
import io
import PyPDF2  # Kept for fallback/basic info if needed, main parsing via fitz
import pandas as pd
import numpy as np
import random
import sqlalchemy # For SQL DB connection
from pathlib import Path
import re
import csv
import logging
from logging.handlers import RotatingFileHandler # Enhanced logging
import datetime
import traceback
from typing import Any, Sequence, List, Generator, AsyncGenerator, Union, Dict, Tuple, Optional, Callable
import hashlib # For file hashing
import fitz  # PyMuPDF for advanced PDF parsing
import base64 # For image display/download
from PIL import Image # For image operations if needed, used implicitly by st.image
import time # For simulating delays if needed (e.g., in dummies)
import pickle # Potentially for FAISS cache if re-enabled

# --- LlamaIndex Core Imports ---
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings,
    QueryBundle,
    PromptTemplate,
)
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode # Use TextNode for richer metadata
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter # Use SentenceSplitter for better chunking control
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import BaseQueryEngine, SubQuestionQueryEngine
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    BaseSynthesizer,
    ResponseMode
)
from llama_index.core.callbacks import CallbackManager # Import CallbackManager

# --- LlamaIndex LLM Imports ---
from llama_index.core.llms import (
    LLM,
    CompletionResponse,
    ChatResponse,
    ChatMessage,
    MessageRole,
    LLMMetadata,
)
# --- LlamaIndex Response Schema ---
from llama_index.core.base.response.schema import Response # Import Response

# --- LlamaIndex Embeddings & Vector Stores ---
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams, PointStruct, UpdateStatus

# --- Langchain Imports (If needed, e.g., for alternative splitting/retrieval) ---
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Alternative splitter if preferred

# ==============================================================================
# --- Configuration & Directory Setup (Code 2 Style) ---
# ==============================================================================

# --- Global Base Path Definition ---
try:
    BASE_PATH = Path(__file__).parent.resolve()
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., some notebooks)
    BASE_PATH = Path(".").resolve()
print(f"BASE_PATH detected as: {BASE_PATH}") # Debug print

# --- Derived Paths ---
LOG_DIR = BASE_PATH / "logs_merged"
FEEDBACK_DIR = BASE_PATH / "feedback_merged"
IMAGES_DIR = BASE_PATH / "images_merged" # Store extracted images here
VECTORDB_DIR = BASE_PATH / "vectordb_merged" # Qdrant storage path
SQL_DB_DIR = BASE_PATH / "sql_database_merged" # SQLite storage path
MODEL_CACHE_DIR = BASE_PATH / "models_cache" # Cache for embedding models

# --- Create Directories ---
LOG_DIR.mkdir(exist_ok=True)
FEEDBACK_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
VECTORDB_DIR.mkdir(exist_ok=True)
SQL_DB_DIR.mkdir(exist_ok=True)
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# --- Logging Setup (Code 2 Style - Robust) ---
LOG_FILE = LOG_DIR / "merged_app.log"
logger = logging.getLogger("merged_rag_engine")
logger.setLevel(logging.INFO)
# Clear existing handlers if any (useful for Streamlit re-runs)
if logger.hasHandlers():
    logger.handlers.clear()
# Rotating File Handler
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3, encoding='utf-8') # 10MB per file, 3 backups
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
# Console Handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Simpler format for console
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

logger.info(f"Logging configured. Log file: {LOG_FILE}")

# --- Feedback File Path ---
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.csv"

# --- Database & Vector Store Config (Code 1 adapted paths) ---
QDRANT_PATH = VECTORDB_DIR / "qdrant_data" # Specific path for Qdrant storage
QDRANT_PDF_COLLECTION_PREFIX = "pdf_coll_" # Prefix for unique PDF collections in Qdrant
SQL_DB_FILENAME = "csv_data.db"
SQL_DB_PATH = SQL_DB_DIR / SQL_DB_FILENAME
SQL_DB_URL = f"sqlite:///{SQL_DB_PATH.resolve()}" # Use resolved absolute path

# --- Model Configuration (Code 1 Style) ---
# Ensure consistency, using model from Code 1 as primary
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384 # Dimension for all-MiniLM-L6-v2
LLM_MODEL_NAME = "custom_abc_llm" # Placeholder name (user indicates it's replaced)

# ==============================================================================
# --- Helper Functions (Combined & Adapted) ---
# ==============================================================================

# --- CSV Description Helpers (From Code 1) ---
def got_type(list_):
    # Safely determines the likely type of data in a list
    def judge(string):
        s_val = str(string) if string is not None else ""
        if not s_val: return "string"
        try: int(s_val); return "int"
        except ValueError:
            try: float(s_val); return "float"
            except ValueError: return "string"
    return [judge(str(x) if x is not None else "") for x in list_]

def column_des(df):
    # Generates descriptions for each column in a DataFrame
    def single_des(name, data):
        description = "{\"Column Name\": \"" + name + "\"" + ", "
        valid_data = data.dropna().tolist()
        if not valid_data: return "" # Skip empty columns
        pre_len = len(data)
        post_len = len(valid_data)
        types = got_type(valid_data)

        if "string" in types:
            type_ = "string"
            data_proc = [str(x) for x in valid_data]
        elif "float" in types:
            type_ = "float"
            data_proc = np.array([float(x) for x in valid_data])
        else:
            type_ = "int"
            data_proc = np.array([int(x) for x in valid_data])

        description += "\"Type\": \"" + type_ + "\", "

        if type_ in ["int", "float"]:
            if data_proc.size > 0:
                min_ = data_proc.min()
                max_ = data_proc.max()
                description += "\"MIN\": " + str(min_) + ", \"MAX\": " + str(max_)
            else:
                description += "\"MIN\": null, \"MAX\": null"
        elif type_ == "string":
            # Sample unique values, handling potential quotes within strings
            values = list(set(["\"" + str(x).strip().replace('"', "'") + "\"" for x in data_proc]))
            random.shuffle(values)
            if len(values) > 15:
                values = values[:random.randint(5, 10)] # Limit sample size
            numerates = ", ".join(values)
            description += "\"Sample Values\": [" + numerates + "]"

        description += ", \"Contains NaN\": " + str(post_len != pre_len)
        return description + "}"

    columns_dec = []
    for c in df.columns:
        try:
            desc = single_des(c, df[c])
            if desc: columns_dec.append(desc)
        except Exception as e:
            logger.warning(f"Could not generate description for CSV column '{c}': {e}")
            # Provide a basic fallback description
            columns_dec.append("{\"Column Name\": \"" + str(c) + "\", \"Error\": \"Could not generate description\"}")

    random.shuffle(columns_dec) # Shuffle to avoid bias if context is truncated
    return "\n".join(columns_dec)

def generate_table_description(df: pd.DataFrame, table_name: str, source_csv_name: str) -> str:
    # Generates a summary description for a SQL table derived from a CSV
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
        # Fallback description on error
        return f"Error generating description for table '{table_name}'. Columns: {', '.join(df.columns)}. Error: {e}"

# --- Sanitization Helper (From Code 1) ---
def sanitize_for_name(filename: str, max_len: int = 40) -> str:
    # Creates a safe name for use in filenames, table names, collection names etc.
    name = Path(filename).stem
    name = re.sub(r'\W+', '_', name) # Replace non-alphanumeric with underscore
    name = name.strip('_')
    if not name: # Handle cases where filename was only symbols
        name = f"file_{random.randint(1000, 9999)}"
    if name[0].isdigit(): # Ensure name doesn't start with a digit
        name = '_' + name
    name = name[:max_len].lower() # Truncate and lowercase
    # Final check for empty or just underscore names
    if not name or name in ["_", "__"]:
        name = f"file_{random.randint(1000, 9999)}"
    return name

# --- Feedback Helpers (From Code 2) ---
def init_feedback_csv():
    # Creates the feedback CSV file with header if it doesn't exist
    try:
        if not FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Define header, including source details
                writer.writerow(['timestamp', 'email', 'question', 'files_processed', 'response', 'sources_info_pdf', 'sources_info_csv_sql', 'error_info'])
            logger.info(f"Created new feedback file at {FEEDBACK_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize feedback CSV at {FEEDBACK_FILE}: {e}", exc_info=True)

def log_feedback(email, question, files_processed, response, sources_info_pdf="N/A", sources_info_csv_sql="N/A", error_info="N/A"):
    # Logs interaction details to the feedback CSV file
    try:
        # Ensure the file exists before attempting to append
        if not FEEDBACK_FILE.exists():
            init_feedback_csv()
            # Check again, if init failed, log error and return
            if not FEEDBACK_FILE.exists():
                logger.error("Feedback file could not be created. Cannot log feedback.")
                return

        with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Sanitize response slightly for CSV logging
            response_log = str(response).replace('\n', '\\n').replace('\r', '').replace('"', "'")
            # Ensure other fields are strings
            question_log = str(question).replace('\n', ' ').replace('"', "'")
            files_log = ", ".join(files_processed) if isinstance(files_processed, list) else str(files_processed)

            writer.writerow([
                timestamp, str(email), question_log, files_log,
                response_log, str(sources_info_pdf), str(sources_info_csv_sql), str(error_info)
            ])
        # logger.info(f"Logged feedback from {email} for question: {question_log[:50]}...") # Avoid logging full question potentially
    except Exception as e:
        logger.error(f"Error logging feedback: {e}", exc_info=True)

# --- Login Helpers (From Code 2) ---
def validate_email(email):
    # Basic email format validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def check_login():
    # Checks if the user is marked as logged in in the session state
    return st.session_state.get("logged_in", False)

def show_login():
    # Renders the login form
    st.title("Login")
    st.write("Please enter your email to use the application.")
    email = st.text_input("Email", key="login_email", placeholder="Enter your email address")
    if st.button("Login", key="login_button"):
        if validate_email(email):
            st.session_state.logged_in = True
            st.session_state.user_email = email
            logger.info(f"User logged in: {email}")
            st.rerun() # Rerun to load the main app UI
        else:
            st.error("Please enter a valid email address.")
            logger.warning(f"Invalid login attempt with email: {email}")

# --- Image Processing Placeholder (From Code 2 - Adapted) ---
def image_response(image_bytes: bytes, image_filename: str, page_num: int) -> str:
    """
    Placeholder: Replace with actual image understanding model call.
    Takes image bytes and metadata, returns a textual explanation.
    """
    # --- IMPORTANT ---
    # This is where you integrate your actual image analysis API/model.
    # It should take 'image_bytes' as input.
    # Ensure your real implementation doesn't have caching issues or state leakage.
    # --- END IMPORTANT ---

    logger.info(f"Simulating image explanation for: {image_filename} (Page {page_num})")
    time.sleep(0.1) # Simulate API call delay
    try:
        image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        # Simulate a basic response based on filename/context
        return f"Simulated analysis of {image_filename} from page {page_num}. [Hash: {image_hash}]"
    except Exception as e:
        logger.error(f"Error during dummy image explanation for {image_filename}: {e}")
        return f"Error generating explanation for {image_filename}."

# --- PDF Processing with PyMuPDF (fitz) (From Code 2 - Adapted for Merged) ---
def process_pdf_with_fitz(file_bytes: bytes, file_hash: str, file_name: str, enable_image_analysis: bool) -> List[Dict[str, Any]]:
    """
    Extract text, images, URLs from PDF bytes using PyMuPDF.
    Conditionally generates image explanations.
    Stores images locally based on file_hash. Returns structured page content.
    Uses absolute paths for reliability.
    """
    pages_content = []
    # Create a dedicated directory for this PDF's images using its hash
    pdf_image_dir = IMAGES_DIR.resolve() / file_hash # Use resolved absolute path
    try:
        pdf_image_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing PDF '{file_name}' with PyMuPDF (Hash: {file_hash}). Image analysis enabled: {enable_image_analysis}")
        logger.info(f"Images will be saved to: {pdf_image_dir}")

        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        page_count_total = pdf_document.page_count

        # Use context manager for Streamlit spinner/progress if called from Streamlit context
        # This function might be called outside Streamlit (e.g., background task), so check existence
        progress_bar = None
        if hasattr(st, 'progress'):
             progress_bar = st.progress(0, text=f"Processing '{file_name}': Page 1/{page_count_total}...")

        for page_num in range(page_count_total):
            page_text_content = ""
            page_urls = []
            page_images_metadata = []
            current_page_num_display = page_num + 1

            if progress_bar:
                 progress_bar.progress(current_page_num_display / page_count_total, text=f"Processing '{file_name}': Page {current_page_num_display}/{page_count_total}...")

            try:
                page = pdf_document[page_num]

                # 1. Extract Text
                page_text_content = page.get_text("text", sort=True) # Basic text extraction

                # 2. Extract URLs
                links = page.get_links()
                page_urls = list(set([link["uri"] for link in links if "uri" in link and link["uri"].startswith("http")]))

                # 3. Extract Images
                image_list = page.get_images(full=True)
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    image_bytes_content = None
                    try:
                        base_image = pdf_document.extract_image(xref)
                        if base_image and base_image.get("image"):
                            image_bytes_content = base_image["image"]
                            image_ext = base_image.get("ext", "png") # Default to png if ext missing
                            # Create a unique, descriptive filename
                            image_filename = f'img_p{current_page_num_display}_{img_index + 1}.{image_ext}'
                            # Use resolved absolute path for saving
                            absolute_image_path = pdf_image_dir / image_filename

                            # Save the image file
                            with open(absolute_image_path, 'wb') as img_file:
                                img_file.write(image_bytes_content)

                            # Generate explanation only if enabled AND bytes were extracted
                            explanation = "Image analysis disabled."
                            if enable_image_analysis and image_bytes_content:
                                try:
                                    explanation = image_response(image_bytes_content, image_filename, current_page_num_display)
                                except Exception as img_resp_err:
                                     logger.error(f"Image explanation API failed for {image_filename} (Page {current_page_num_display}): {img_resp_err}", exc_info=True)
                                     explanation = f"Error during image analysis: {img_resp_err}"

                            page_images_metadata.append({
                                "path": str(absolute_image_path), # Store absolute path string
                                "filename": image_filename,
                                "explanation": explanation
                            })
                        else:
                             logger.warning(f"Could not extract image data for xref {xref} on page {current_page_num_display} of '{file_name}'.")
                    except Exception as img_proc_err:
                        logger.error(f"Error processing image xref {xref} on page {current_page_num_display} of '{file_name}': {img_proc_err}", exc_info=True)
                        # Optionally add a placeholder if image processing fails
                        page_images_metadata.append({
                            "path": None,
                            "filename": f"error_xref_{xref}",
                            "explanation": f"Error processing image: {img_proc_err}"
                         })

            except Exception as page_err:
                 logger.error(f"Error processing page {current_page_num_display} of '{file_name}': {page_err}", exc_info=True)
                 page_text_content = f"[Error processing page {current_page_num_display}]" # Add error marker to text

            # Store structured data for the page
            page_data = {
                "page_num": current_page_num_display,
                "text": page_text_content or "", # Ensure text is not None
                "urls": page_urls,
                "images": page_images_metadata # List of dicts {path, filename, explanation}
            }
            pages_content.append(page_data)

        # Close the PDF document
        pdf_document.close()
        if progress_bar: progress_bar.empty() # Clear progress bar
        logger.info(f"Successfully extracted content from {len(pages_content)} pages of '{file_name}'.")

    except Exception as e:
        logger.error(f"Fatal error processing PDF '{file_name}' with PyMuPDF: {e}", exc_info=True)
        if 'pdf_document' in locals() and pdf_document: pdf_document.close() # Ensure closure on error
        if progress_bar: progress_bar.empty() # Ensure progress bar cleared
        # Re-raise the exception so the calling function knows processing failed
        raise RuntimeError(f"Failed to process PDF '{file_name}' with PyMuPDF.") from e

    return pages_content

# --- Enhanced Chunking (Adapted from Code 2) ---
def create_enhanced_nodes(pages_content: List[Dict[str, Any]], file_name: str, file_hash: str) -> List[TextNode]:
    """
    Creates LlamaIndex TextNode objects from page content.
    Associates rich metadata (page, urls, images with paths/explanations) with each node.
    Uses SentenceSplitter for potentially better chunking.
    """
    # Configure splitter (adjust chunk_size/overlap as needed)
    # Using SentenceSplitter which might be more robust for diverse text.
    # Adjust chunk_size and overlap based on embedding model context window and desired granularity.
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
    nodes = []
    node_counter = 0

    logger.info(f"Creating enhanced nodes for '{file_name}'...")
    for page_data in pages_content:
        page_num = page_data["page_num"]
        page_text = page_data["text"]
        page_urls = page_data["urls"]
        page_images = page_data["images"] # List of image metadata dicts

        # Split the text from the current page
        text_chunks = text_splitter.split_text(page_text)

        for chunk_text in text_chunks:
            node_id = f"{file_hash}_p{page_num}_c{node_counter}"
            # Metadata should be serializable (strings, numbers, lists/dicts thereof)
            metadata = {
                "file_name": file_name,
                "file_hash": file_hash, # Link back to the original file
                "page_num": page_num,
                "urls": page_urls, # URLs from the source page
                "images": page_images # Image metadata list {path, filename, explanation} from the source page
            }
            node = TextNode(
                id_=node_id,
                text=chunk_text,
                metadata=metadata,
                # Excluded fields can help prevent embedding unnecessary text
                excluded_embed_metadata_keys=["file_name", "file_hash", "page_num", "urls", "images"], # Don't embed metadata itself
                excluded_llm_metadata_keys=["file_hash"] # LLM might not need the hash
            )
            nodes.append(node)
            node_counter += 1

    logger.info(f"Created {len(nodes)} enhanced TextNodes for '{file_name}'.")
    return nodes

# --- Image Display Helper (From Code 2 - For Expander) ---
def display_image_expander(image_path: str, filename: str) -> str:
    """Generates HTML to display a small image preview from an absolute path for the source expander."""
    if not image_path or not Path(image_path).is_file():
        logger.warning(f"Expander: Image path not found or invalid: {image_path}")
        return f"<div style='text-align: center; font-style: italic; font-size: 0.8em; color: grey; border: 1px dashed grey; padding: 5px; margin: 5px;'>[Image not found: {filename}]</div>"
    try:
        # Read bytes and encode for inline HTML display
        img_bytes = Path(image_path).read_bytes()
        b64_encoded = base64.b64encode(img_bytes).decode()
        # Determine mime type for broad compatibility
        ext = Path(image_path).suffix.lstrip('.').lower()
        mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'] else 'image/png' # Fallback
        if ext == 'svg': mime_type = 'image/svg+xml'
        # Use smaller max-width for expander view
        return f'<img src="data:{mime_type};base64,{b64_encoded}" alt="{filename}" title="{filename}" style="max-width: 80px; height: auto; display: block; margin: 5px auto; border: 1px solid #ddd;">'
    except Exception as e:
        logger.error(f"Expander: Error creating image tag for {image_path}: {e}")
        return f"<div style='text-align: center; font-style: italic; font-size: 0.8em; color: red; border: 1px dashed red; padding: 5px; margin: 5px;'>[Error displaying: {filename}]</div>"


# ==============================================================================
# --- Custom LLM Implementation (Placeholder - From Code 1) ---
# ==============================================================================
# Assume user replaces this with their actual LLM integration
def abc_response(prompt: str) -> str:
    """ Placeholder LLM response function. """
    logger.info(f"MyCustomLLM received prompt (first 100 chars): {prompt[:100]}...")
    # Simulate LLM response based on prompt structure (e.g., SQL vs RAG)
    response = f"This is a dummy response from MyCustomLLM for the prompt starting with: {prompt[:50]}..."
    if "SELECT" in prompt[:100] or "table name" in prompt.lower()[:100]:
         response = f"SELECT COUNT(*) FROM dummy_table; -- Dummy SQL Response from LLM for prompt: {prompt[:50]}..." # Simulate SQL
    # Add simple simulation for image marker if image paths are in prompt
    elif "[Image:" in prompt or "Path:" in prompt:
        response += "\nSee details in [Image: /dummy/path/image.png]." # Simulate image marker usage

    logger.info(f"MyCustomLLM generated response (first 100 chars): {response[:100]}...")
    return response

class MyCustomLLM(LLM):
    """Placeholder LLM implementing the LlamaIndex LLM interface."""
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        logger.info("MyCustomLLM: chat() called")
        # Combine messages into a single prompt string for the dummy function
        prompt = "\n".join([f"{m.role.value}: {m.content}" for m in messages])
        response_text = abc_response(prompt)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=response_text))

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        logger.info("MyCustomLLM: complete() called")
        response_text = abc_response(prompt)
        return CompletionResponse(text=response_text)

    # --- Async methods simply call sync methods ---
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        logger.info("MyCustomLLM: achat() called - calling sync chat()")
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        logger.info("MyCustomLLM: acomplete() called - calling sync complete()")
        return self.complete(prompt, **kwargs)

    # --- Streaming methods yield single response (non-streaming) ---
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        logger.warning("MyCustomLLM: stream_chat() called - Returning single response")
        yield self.chat(messages, **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        logger.warning("MyCustomLLM: stream_complete() called - Returning single response")
        yield self.complete(prompt, **kwargs)

    # --- Async Streaming methods (Not Implemented) ---
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        logger.error("MyCustomLLM: astream_chat() called - NotImplementedError")
        raise NotImplementedError("Async streaming chat not supported by MyCustomLLM.")
        yield # Required for async generator type hint

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        logger.error("MyCustomLLM: astream_complete() called - NotImplementedError")
        raise NotImplementedError("Async streaming complete not supported by MyCustomLLM.")
        yield # Required for async generator type hint

    @property
    def metadata(self) -> LLMMetadata:
        # Provides metadata about the LLM
        return LLMMetadata(
            model_name=LLM_MODEL_NAME,
            is_chat_model=True # Can behave like a chat model via chat() method
        )

# ==============================================================================
# --- Custom SQL Engine (From Code 1 - Adapted LLM Call) ---
# ==============================================================================
class CustomSQLEngine:
    """Generates and executes SQL queries based on natural language using an LLM."""
    def __init__(
        self, sql_engine: sqlalchemy.engine.Engine, table_name: str,
        table_description: Optional[str] = None, verbose: bool = True
    ):
        self.sql_engine = sql_engine
        self.table_name = table_name
        # LLM callback is now implicitly Settings.llm
        if not Settings.llm:
             raise ValueError("LLM must be configured in llama_index.core.Settings for CustomSQLEngine")
        self.llm = Settings.llm
        self.table_description = table_description or ""
        self.verbose = verbose

        # --- SQL Generation Prompts ---
        self.sql_prompt_template = PromptTemplate("""You are an expert SQL query generator for SQLite.

Based on the user's natural language query: "{query_str}"

Generate a single, executable SQL query for the following SQL table:
Table Name: {table_name}

{table_description}

IMPORTANT GUIDELINES FOR SQLITE:
1. Use double quotes for table and column names IF they contain spaces or special characters (usually avoided by cleaning). Stick to standard SQL syntax otherwise. Column names provided are already cleaned.
2. For string comparisons, use LIKE with % wildcards for partial matches (e.g., "column" LIKE '%value%'). String comparisons are case-sensitive by default. Use LOWER() for case-insensitive search if needed (e.g., LOWER("column") LIKE LOWER('%value%')).
3. For aggregations (SUM, COUNT, AVG, MIN, MAX), always use GROUP BY for any non-aggregated columns included in the SELECT list.
4. Handle NULL values explicitly using `IS NULL` or `IS NOT NULL`. Use `COALESCE("column", default_value)` to provide defaults if needed.
5. Avoid complex joins. Assume the query targets only the single table: "{table_name}".
6. For date/time operations, use standard SQLite functions like DATE(), TIME(), DATETIME(), STRFTIME(). Assume date columns are stored in a format SQLite understands (e.g., YYYY-MM-DD HH:MM:SS).
7. Keep queries simple and direct. Only select columns needed to answer the query.
8. Ensure correct quoting: Double quotes for identifiers (table/column names), single quotes for string literals.

Return ONLY the executable SQL query without any explanation, markdown formatting, comments, or preamble like ```sql.
Example: SELECT COUNT(*) FROM "{table_name}" WHERE "Some_Column" > 10;
Another Example: SELECT "Category", AVG("Value") FROM "{table_name}" GROUP BY "Category";
""")

        self.sql_fix_prompt = PromptTemplate("""The following SQL query for SQLite failed:
```sql
{failed_sql}
```
Error message: {error_msg}

Table information:
Table Name: {table_name}
{table_description}

Please fix the SQL query to work with SQLite following these guidelines:
Fix syntax errors (quoting, commas, keywords). Ensure column names match exactly those provided in the description (case-sensitive). Column names are: {column_names}
Replace incompatible functions with SQLite equivalents.
Simplify the query logic if it seems overly complex or likely caused the error. Check aggregations and GROUP BY clauses.
Ensure data types in comparisons are appropriate (e.g., don't compare text to numbers directly without casting if necessary). Check string literals use single quotes.
Double-check table name correctness: "{table_name}"

Return ONLY the corrected executable SQL query without any explanation or formatting.
""")

    def _get_schema_info(self) -> Tuple[Optional[str], List[str]]:
        # Gets schema and column names, handling potential errors
        try:
            inspector = sqlalchemy.inspect(self.sql_engine)
            if not inspector.has_table(self.table_name):
                return f"Error: Table '{self.table_name}' does not exist in the database.", []

            metadata_obj = sqlalchemy.MetaData()
            # Reflect only the specific table to avoid loading others
            metadata_obj.reflect(bind=self.sql_engine, only=[self.table_name])
            table = metadata_obj.tables.get(self.table_name)

            if table is None: # Should not happen if has_table passed, but check anyway
                 return f"Error: Could not find table '{self.table_name}' metadata after reflection.", []

            columns_info = []
            column_names = []
            for column in table.columns:
                col_type = str(column.type)
                constraints = []
                if column.primary_key: constraints.append("PRIMARY KEY")
                # Nullability check might vary slightly by DB dialect/reflection
                # if not column.nullable: constraints.append("NOT NULL") # Keep it simple for now
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                columns_info.append(f"  - \"{column.name}\": {col_type}{constraint_str}")
                column_names.append(column.name)

            schema_str = f"Actual Schema for table \"{self.table_name}\":\nColumns:\n" + "\n".join(columns_info)
            return schema_str, column_names

        except sqlalchemy.exc.SQLAlchemyError as db_err:
             logger.error(f"Database error getting schema info for {self.table_name}: {db_err}", exc_info=True)
             return f"Database Error retrieving schema for table {self.table_name}: {db_err}", []
        except Exception as e:
            logger.error(f"General error getting schema info for {self.table_name}: {e}", exc_info=True)
            return f"Error retrieving schema for table {self.table_name}: {e}", []

    def _clean_sql(self, sql: str) -> str:
        # Cleans SQL string received from LLM
        sql = re.sub(r'```sql|```', '', sql) # Remove markdown code fences
        sql = re.sub(r'^sql\s+', '', sql, flags=re.IGNORECASE) # Remove leading 'sql' keyword
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE) # Remove single-line comments
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL) # Remove multi-line comments
        sql = sql.strip().rstrip(';') # Remove leading/trailing whitespace and trailing semicolon
        return sql

    def _is_safe_sql(self, sql: str) -> bool:
        # Basic safety check for SQL query (read-only)
        lower_sql = sql.lower().strip()
        # Allow SELECT and WITH (for Common Table Expressions)
        if not (lower_sql.startswith('select') or lower_sql.startswith('with')):
            logger.warning(f"SQL safety check failed: Query does not start with SELECT or WITH. Query: {sql}")
            return False
        # Deny potentially harmful keywords
        dangerous_keywords = [
            r'\bdrop\b', r'\bdelete\b', r'\btruncate\b', r'\bupdate\b', r'\binsert\b',
            r'\balter\b', r'\bcreate\b', r'\breplace\b', r'\bgrant\b', r'\brevoke\b',
            r'\battach\b', r'\bdetach\b', r'\banalyze\b', r'\breindex\b', r'\bvacuum\b',
            r'\bpragma\b' # Be cautious with pragma in SQLite
        ]
        for pattern in dangerous_keywords:
            if re.search(pattern, lower_sql):
                logger.warning(f"SQL safety check failed: Dangerous keyword '{pattern}' found. Query: {sql}")
                return False
        return True

    def _format_results(self, result_df: pd.DataFrame) -> str:
       # Formats DataFrame results into a markdown string for display
        if result_df.empty:
            return "The query returned no results."

        max_rows_to_show = 20
        max_cols_to_show = 15
        original_shape = result_df.shape
        df_to_show = result_df

        show_cols_truncated = original_shape[1] > max_cols_to_show
        show_rows_truncated = original_shape[0] > max_rows_to_show

        if show_cols_truncated:
            df_to_show = df_to_show.iloc[:, :max_cols_to_show]
        if show_rows_truncated:
            df_to_show = df_to_show.head(max_rows_to_show)

        result_str = ""
        if show_rows_truncated or show_cols_truncated:
            result_str += f"Query returned {original_shape[0]} rows and {original_shape[1]} columns. "
            parts = []
            if show_rows_truncated: parts.append(f"first {max_rows_to_show} rows")
            if show_cols_truncated: parts.append(f"first {max_cols_to_show} columns")
            result_str += f"Showing { ' and '.join(parts)}:\n\n"
        else:
            result_str += "Query Result:\n\n"

        try:
            # Use markdown format for better display in Streamlit
            markdown_result = df_to_show.to_markdown(index=False)
        except Exception:
            # Fallback to plain string if markdown fails
            markdown_result = df_to_show.to_string(index=False)

        return result_str + markdown_result

    def _execute_sql(self, sql: str) -> Tuple[bool, Union[pd.DataFrame, str]]:
        # Executes a single SQL query after safety check
        try:
            if not self._is_safe_sql(sql):
                logger.error(f"SQL query failed safety check: {sql}")
                return False, "SQL query failed safety check (potentially unsafe operation)."

            if self.verbose: logger.info(f"Executing safe SQL: {sql}")
            with self.sql_engine.connect() as connection:
                result_df = pd.read_sql_query(sqlalchemy.text(sql), connection) # Use sqlalchemy.text() for safety
            if self.verbose: logger.info(f"SQL execution successful. Rows returned: {len(result_df)}")
            return True, result_df # Return DataFrame on success

        except sqlalchemy.exc.SQLAlchemyError as db_err:
             # More specific error logging
             logger.error(f"Database execution error for SQL: {sql}\nError: {db_err}", exc_info=True)
             # Provide a slightly more user-friendly error message
             error_detail = str(db_err).split('\n')[0] # Get first line of error
             return False, f"Database Error: {error_detail}"
        except Exception as e:
            logger.error(f"General error executing SQL: {sql}\nError: {e}", exc_info=True)
            return False, f"General Error executing SQL: {e}"

    def _execute_with_retry(self, sql: str, max_retries: int = 1) -> Tuple[bool, Union[pd.DataFrame, str], str]:
        # Executes SQL, attempting to fix with LLM on failure
        current_sql = sql
        original_sql = sql
        schema_info, column_names = self._get_schema_info()
        if "Error:" in schema_info:
            return False, f"Schema error prevented query execution: {schema_info}", original_sql
        if not column_names:
             return False, f"Could not retrieve column names for table '{self.table_name}'. Cannot execute query.", original_sql

        full_table_description = f"{self.table_description}\n\n{schema_info}"

        for attempt in range(max_retries + 1):
            if self.verbose: logger.info(f"SQL Execution Attempt {attempt + 1}/{max_retries + 1}")
            success, result = self._execute_sql(current_sql)

            if success:
                if self.verbose: logger.info(f"SQL successfully executed on attempt {attempt + 1}.")
                return True, result, current_sql # result is DataFrame

            # Execution failed, prepare for retry if attempts remain
            error_message = str(result) # Contains the error string
            logger.warning(f"SQL execution failed on attempt {attempt + 1}. Error: {error_message}\nSQL: {current_sql}")

            if attempt == max_retries:
                logger.error(f"SQL query failed after {max_retries + 1} attempts. Final error: {error_message}")
                return False, f"SQL query failed: {error_message}", current_sql # Return final failed SQL

            # Try to fix the SQL using LLM
            if self.verbose: logger.info("Attempting SQL fix using LLM...")
            try:
                fix_prompt = self.sql_fix_prompt.format(
                    failed_sql=current_sql,
                    error_msg=error_message,
                    table_name=self.table_name,
                    table_description=full_table_description,
                    column_names = ", ".join([f'"{c}"' for c in column_names]) # Provide clean column names for fix prompt
                 )
                # Use the configured LLM via Settings
                fixed_sql_response = self.llm.complete(fix_prompt).text
                fixed_sql = self._clean_sql(fixed_sql_response)

                if fixed_sql and fixed_sql.lower() != current_sql.lower():
                    current_sql = fixed_sql
                    if self.verbose: logger.info(f"LLM proposed SQL fix: {current_sql}")
                    # Continue to the next iteration to try the fixed SQL
                else:
                    logger.warning("LLM fix attempt failed or produced identical SQL. Aborting retry.")
                    # Return original failure if fix doesn't work
                    return False, f"SQL query failed: {error_message} (LLM fix failed or produced no change)", original_sql

            except Exception as fix_error:
                logger.error(f"Error occurred during LLM SQL fix attempt: {fix_error}", exc_info=True)
                # Return original failure if fix attempt itself causes an error
                return False, f"SQL query failed: {error_message} (Error during LLM fix attempt: {fix_error})", original_sql

        # Should not be reached, but included as a fallback
        logger.error("Exited SQL retry loop unexpectedly.")
        return False, "Unexpected error during SQL retry logic.", original_sql

    def query(self, query_text: str) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str]]:
        """
        Takes natural language query, generates SQL, executes it (with retry),
        and returns formatted results string, raw DataFrame, and final SQL used.
        """
        if self.verbose: logger.info(f"Custom SQL Engine Query received for table '{self.table_name}': {query_text}")

        try:
            # Get schema info needed for generation and fixing
            schema_info, column_names = self._get_schema_info()
            if "Error:" in schema_info:
                logger.error(f"Schema error for table {self.table_name}: {schema_info}")
                return f"Schema error prevented query: {schema_info}", None, None
            if not column_names:
                logger.error(f"Could not retrieve column names for table {self.table_name}")
                return f"Could not retrieve column names for table '{self.table_name}'. Cannot execute query.", None, None

            full_table_description = f"{self.table_description}\n\n{schema_info}"

            # Generate the initial SQL query using LLM
            generate_prompt = self.sql_prompt_template.format(
                query_str=query_text,
                table_name=self.table_name,
                table_description=full_table_description
            )

            logger.info(f"Generating SQL query for: {query_text}")
            sql_response_text = self.llm.complete(generate_prompt).text
            # Handle potential empty or error responses from LLM here if needed

            sql_query = self._clean_sql(sql_response_text)
            if not sql_query:
                logger.error("LLM failed to generate any SQL query.")
                return "Error: LLM failed to generate SQL query.", None, None
            if self.verbose: logger.info(f"LLM generated initial SQL: {sql_query}")

            # Execute the query with retry/fix logic
            success, result_data, final_sql = self._execute_with_retry(sql_query)

            if not success:
                # result_data contains the error message string here
                logger.error(f"Final SQL query execution failed for table '{self.table_name}'. Error: {result_data}")
                # Return error message, None for DataFrame, and the final SQL attempted
                return f"Error querying table '{self.table_name}':\n{result_data}\n\nSQL Attempted:\n```sql\n{final_sql}\n```", None, final_sql
            else:
                # result_data contains the DataFrame here
                formatted_results = self._format_results(result_data)
                logger.info(f"Query successful for table '{self.table_name}'. Returning formatted results and DataFrame.")
                # Return formatted string, the result DataFrame, and the successful SQL
                return f"Query result from table '{self.table_name}':\n\n{formatted_results}", result_data, final_sql

        except Exception as e:
            logger.critical(f"Unexpected error in CustomSQLEngine query method for table '{self.table_name}': {e}", exc_info=True)
            return f"Unexpected critical error querying table '{self.table_name}': {e}", None, None

# ==============================================================================
# --- Custom SQL Engine Wrapper (From Code 1 - Adapted for DataFrame Return) ---
# ==============================================================================
class CustomSQLQueryEngineWrapper(BaseQueryEngine):
    """
    Adapter to make the CustomSQLEngine compatible with LlamaIndex tools,
    returning both formatted string and raw DataFrame.
    """
    def __init__(self, engine: CustomSQLEngine):
        self._engine = engine
        # LLM and CallbackManager are implicitly handled by Settings or the outer agent
        super().__init__(callback_manager=Settings.callback_manager)

    @property
    def engine(self) -> CustomSQLEngine:
        return self._engine

    def _get_prompt_modules(self) -> Dict[str, Any]:
        """Get prompt sub-modules."""
        # This engine doesn't use separate modules like Pydantic programs
        return {}

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Run the query."""
        logger.info(f"CustomSQLWrapper: _query received for table {self._engine.table_name}: {query_bundle.query_str}")
        formatted_str, df_result, sql_used = self._engine.query(query_bundle.query_str)

        # Create metadata dictionary to pass back extra info
        response_metadata = {
            "sql_query": sql_used,
            "raw_dataframe": df_result, # Pass the DataFrame if successful
            "formatted_string_result": formatted_str # Keep the string result too
        }

        # The primary response text should be the formatted string for direct use by LLM/synthesizer
        # Add the extra info to the response metadata
        return Response(response=formatted_str or "Query executed.", metadata=response_metadata)

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Run the query asynchronously (delegating to sync method)."""
        logger.info(f"CustomSQLWrapper: _aquery received for table {self._engine.table_name} - using sync _query")
        # LlamaIndex typically handles running sync methods in async contexts if needed
        return self._query(query_bundle)


# ==============================================================================
# --- Global Settings and Initialization (Combined & Cached) ---
# ==============================================================================

# Use Streamlit's caching for expensive resources like models and clients
@st.cache_resource
def load_llm():
    """Loads the LLM instance (placeholder)."""
    logger.info(f"Initializing LLM: {LLM_MODEL_NAME} (cached)...")
    try:
        llm = MyCustomLLM()
        # Perform a basic check if possible (e.g., metadata call)
        _ = llm.metadata
        logger.info("LLM initialized successfully.")
        return llm
    except Exception as e:
         logger.critical(f"Fatal: Failed to initialize LLM placeholder: {e}", exc_info=True)
         st.error(f"Fatal Error: Could not initialize Language Model. Application cannot proceed: {e}")
         st.stop() # Stop execution if core component fails

@st.cache_resource
def load_embedding_model():
    """Loads the Sentence Transformer embedding model (Code 2 Style Loading)."""
    logger.info(f"Initializing Embedding Model: {EMBEDDING_MODEL_NAME} (cached)...")
    try:
        with st.spinner("Loading embedding model (this might take a moment on first run)..."):
            # Use the dedicated cache directory
            local_model_full_path = MODEL_CACHE_DIR / EMBEDDING_MODEL_NAME.split('/')[-1] # Use last part of name for dir

            if local_model_full_path.exists():
                logger.info(f"Loading embedding model from local cache: {local_model_full_path}")
                embed_model = HuggingFaceEmbedding(model_name=str(local_model_full_path))
            else:
                logger.warning(f"Local cache not found at {local_model_full_path}. Downloading {EMBEDDING_MODEL_NAME}...")
                # Download and save to the cache directory
                embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))
                # Note: HuggingFaceEmbedding handles the actual saving via cache_folder now.
                # We no longer need SentenceTransformer(...).save() explicitly here.
                logger.info(f"Model downloaded/loaded and cached via HuggingFaceEmbedding to: {MODEL_CACHE_DIR}")

            # Verify embedding dimension
            test_embedding = embed_model.get_query_embedding("test")
            actual_dim = len(test_embedding)
            if actual_dim != EXPECTED_EMBEDDING_DIM:
                raise ValueError(f"Embedding dimension mismatch! Expected {EXPECTED_EMBEDDING_DIM}, but model '{EMBEDDING_MODEL_NAME}' generated {actual_dim}.")
            logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully. Dimension: {actual_dim}")
            return embed_model
    except Exception as e:
        logger.critical(f"Fatal: Failed to load/initialize embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
        st.error(f"Fatal Error: Could not load Embedding Model ({EMBEDDING_MODEL_NAME}). Application cannot proceed: {e}")
        st.stop()

@st.cache_resource
def setup_qdrant_client():
    """Sets up and returns Qdrant client instance (Code 2 Style)."""
    logger.info(f"Initializing Qdrant client (cached). Path: {QDRANT_PATH}")
    try:
        # Ensure the parent directory exists
        QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Using path for local persistence
        client = qdrant_client.QdrantClient(path=str(QDRANT_PATH), timeout=60) # Increased timeout
        # Perform a basic operation to check connectivity/status
        client.get_collections()
        logger.info("Qdrant client initialized and connection verified.")
        return client
    except Exception as e:
        # Handle potential locking errors specifically if using path-based client
        if "already accessed" in str(e).lower() or "lock" in str(e).lower():
             logger.critical(f"Fatal: Qdrant lock error on path {QDRANT_PATH}. Another process might be using it. Error: {e}", exc_info=True)
             st.error(f"Fatal Error: Vector Database directory ({QDRANT_PATH}) is locked by another process. Please ensure no other instances are running and try again.")
        else:
             logger.critical(f"Fatal: Failed to initialize Qdrant client at {QDRANT_PATH}: {e}", exc_info=True)
             st.error(f"Fatal Error: Could not connect to Vector Database. Application cannot proceed: {e}")
        st.stop()

def configure_global_settings() -> bool:
    """Configures LlamaIndex global settings with cached resources."""
    try:
        logger.info("Configuring LlamaIndex Global Settings...")
        llm = load_llm()
        embed_model = load_embedding_model()
        # qdrant_client is loaded separately when needed, not stored globally here

        if llm is None or embed_model is None:
             logger.critical("LLM or Embedding Model failed to load. Cannot configure settings.")
             return False

        Settings.llm = llm
        Settings.embed_model = embed_model
        # Configure node parser globally if needed, or locally during processing
        # Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100) # Example
        Settings.chunk_size = 512 # Set default chunk size for consistency if parser not set globally
        Settings.chunk_overlap = 100 # Set default overlap
        Settings.context_window = 4096 # Adjust based on LLM capability
        Settings.num_output = 512 # Adjust based on desired output length

        # Optional: Global callback manager
        # Settings.callback_manager = CallbackManager([])

        logger.info("LlamaIndex Global Settings configured successfully.")
        return True
    except Exception as e:
        logger.critical(f"Critical error during global settings configuration: {e}", exc_info=True)
        st.error(f"Fatal Error: Failed to configure core components. Application cannot proceed: {e}")
        # Don't st.stop() here, let the calling function handle failure if needed
        return False

# ==============================================================================
# --- Data Processing Functions (Combined & Adapted) ---
# ==============================================================================

# --- PDF Processing (Uses Fitz) ---
# process_pdf_with_fitz defined in Helper Functions section

# --- CSV Processing (From Code 1) ---
def process_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[pd.DataFrame]:
    """Reads CSV file into a Pandas DataFrame, handling potential errors."""
    df = None
    file_name = uploaded_file.name
    logger.info(f"Processing CSV: {file_name}")
    try:
        # Read bytes once
        file_bytes = uploaded_file.getvalue()
        # Try UTF-8 first
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for '{file_name}', trying latin1 encoding.")
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='latin1', low_memory=False)
        except Exception as read_err:
             logger.error(f"Pandas read_csv failed for '{file_name}' after encoding attempts: {read_err}", exc_info=True)
             st.error(f"Error reading CSV '{file_name}': Could not parse the file content.")
             return None

        # Remove potential unnamed columns often created during export
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        logger.info(f"Loaded CSV '{file_name}'. Shape: {df.shape}")
        if df.empty:
            st.warning(f"CSV file '{file_name}' is empty or contains no data columns.")
            return None # Treat empty DataFrame as None for consistency
        return df

    except Exception as e:
        logger.error(f"Unexpected error processing CSV '{file_name}': {e}", exc_info=True)
        st.error(f"Unexpected error reading CSV '{file_name}': {e}")
        return None

# ==============================================================================
# --- Tool Creation Functions (Merged & Adapted per file type) ---
# ==============================================================================

def create_pdf_tool(
    pdf_file: st.runtime.uploaded_file_manager.UploadedFile,
    qdrant_client_instance: qdrant_client.QdrantClient,
    enable_image_analysis: bool
) -> Optional[QueryEngineTool]:
    """
    Processes a single PDF file: extracts rich content, creates nodes, indexes in Qdrant,
    and returns a LlamaIndex QueryEngineTool for that PDF.
    Handles errors gracefully.
    """
    file_name = pdf_file.name
    tool = None # Initialize tool as None
    logger.info(f"--- Starting PDF Tool Creation for: {file_name} ---")

    try:
        file_bytes = pdf_file.getvalue()
        if not file_bytes:
             logger.warning(f"PDF file '{file_name}' is empty. Skipping tool creation.")
             st.warning(f"Skipping empty PDF file: '{file_name}'")
             return None
        file_hash = hashlib.md5(file_bytes).hexdigest()
        sanitized_name = sanitize_for_name(file_name)
        collection_name = f"{QDRANT_PDF_COLLECTION_PREFIX}{sanitized_name}_{file_hash[:8]}" # Make collection name more unique
        tool_name = f"pdf_{sanitized_name}_tool"

        logger.info(f"PDF: '{file_name}' (Hash: {file_hash})")
        logger.info(f"Qdrant Collection: {collection_name}")
        logger.info(f"Tool Name: {tool_name}")

        # 1. Check if collection exists and potentially skip processing
        collection_exists = False
        collection_has_data = False
        try:
            collection_info = qdrant_client_instance.get_collection(collection_name=collection_name)
            collection_exists = True
            if collection_info.points_count > 0:
                collection_has_data = True
                logger.info(f"Collection '{collection_name}' exists with {collection_info.points_count} points. Using existing data.")
                st.info(f"Using cached data for '{file_name}'.")
            else:
                 logger.info(f"Collection '{collection_name}' exists but is empty. Will process and add data.")
        except Exception:
            logger.info(f"Collection '{collection_name}' not found. Will create and process.")
            collection_exists = False

        # 2. Process PDF and Index Data *only if* needed
        if not collection_has_data:
            logger.info(f"Processing required for '{file_name}'.")
            with st.spinner(f"Processing PDF '{file_name}' (text, images, urls)..."):
                # Extract content using fitz (handles conditional image analysis)
                pages_content = process_pdf_with_fitz(file_bytes, file_hash, file_name, enable_image_analysis)

            if not pages_content: # Handle case where processing itself failed
                 logger.error(f"PDF processing returned no content for '{file_name}'. Skipping tool creation.")
                 st.error(f"Failed to extract content from PDF '{file_name}'. Cannot create tool.")
                 return None # Critical failure for this file

            with st.spinner(f"Creating text nodes for '{file_name}'..."):
                nodes = create_enhanced_nodes(pages_content, file_name, file_hash)

            if not nodes:
                logger.warning(f"No text nodes created from '{file_name}'. Skipping tool creation.")
                st.warning(f"Could not extract text content from '{file_name}' to create search nodes.")
                return None

            # Ensure collection exists before indexing
            if not collection_exists:
                try:
                    with st.spinner(f"Creating vector store collection '{collection_name}'..."):
                        qdrant_client_instance.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(size=EXPECTED_EMBEDDING_DIM, distance=Distance.COSINE)
                        )
                    logger.info(f"Successfully created Qdrant collection '{collection_name}'.")
                    collection_exists = True # Mark as existing now
                except Exception as create_exc:
                    logger.critical(f"Failed to create Qdrant collection '{collection_name}': {create_exc}", exc_info=True)
                    st.error(f"Fatal Error: Could not create vector database collection for '{file_name}'.")
                    # If collection creation fails, we cannot proceed for this file
                    raise RuntimeError(f"Qdrant collection creation failed for {collection_name}") from create_exc

            # Index the nodes into Qdrant
            try:
                 with st.spinner(f"Indexing '{file_name}' into vector store..."):
                    vector_store = QdrantVectorStore(client=qdrant_client_instance, collection_name=collection_name, prefer_grpc=False) # GRPC can sometimes cause issues
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    # Use the global embedding model from Settings
                    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=Settings.embed_model, show_progress=True)
                 logger.info(f"Successfully indexed {len(nodes)} nodes for '{file_name}' into '{collection_name}'.")
            except Exception as index_err:
                logger.error(f"Failed to index nodes for '{file_name}' into '{collection_name}': {index_err}", exc_info=True)
                st.error(f"Error indexing document '{file_name}' into the vector database.")
                # If indexing fails, the tool cannot be created reliably
                return None

        # 3. Create the Query Engine Tool (always runs if collection exists/was created)
        try:
            # Connect to the potentially existing or newly populated vector store
            vector_store = QdrantVectorStore(client=qdrant_client_instance, collection_name=collection_name)
            index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)

            # Create a query engine for this specific index
            # similarity_top_k controls how many chunks are retrieved
            # response_mode influences how LlamaIndex synthesizes the response (if not using an Agent)
            # For an agent tool, the retriever is often more important than the synthesizer.
            pdf_query_engine = index.as_query_engine(
                similarity_top_k=4, # Retrieve top 4 relevant chunks
                response_mode="compact", # Example mode, agent might override
                # Ensure metadata is included in retrieval
                # This depends on the specific retriever used by as_query_engine.
                # VectorIndexRetriever usually includes metadata by default.
            )
            logger.info(f"Query engine created for PDF '{file_name}'.")

            # Create tool description dynamically
            tool_description = (
                f"Provides information extracted from the PDF document named '{file_name}'. "
                f"Use this tool to answer questions about the text content, find specific information, "
                f"summarize sections, or understand concepts discussed in this document. "
                f"The tool can also access associated metadata like URLs and potentially relevant images (paths/explanations) from the document context."
            )

            pdf_tool = QueryEngineTool(
                query_engine=pdf_query_engine,
                metadata=ToolMetadata(name=tool_name, description=tool_description)
            )
            logger.info(f"Successfully created QueryEngineTool: '{tool_name}' for '{file_name}'.")
            tool = pdf_tool # Assign the created tool

        except Exception as tool_err:
             logger.error(f"Failed to create QueryEngineTool for '{file_name}' (Collection: {collection_name}): {tool_err}", exc_info=True)
             st.error(f"Error creating analysis tool for '{file_name}'. It might not be queryable.")
             # Tool creation failed

    except Exception as outer_err:
        # Catch errors from file reading, hashing, or processing steps before tool creation attempt
         logger.error(f"Critical error during PDF tool setup pipeline for '{file_name}': {outer_err}", exc_info=True)
         st.error(f"Failed to set up processing for PDF '{file_name}': {outer_err}")
         # Ensure tool remains None

    logger.info(f"--- Finished PDF Tool Creation for: {file_name} ---")
    return tool # Return the created tool or None if any step failed


def create_csv_tool(
    df: pd.DataFrame, csv_file_name: str, sql_alchemy_engine: sqlalchemy.engine.Engine
) -> Optional[QueryEngineTool]:
    """
    Processes a DataFrame (from a CSV), loads it into a unique SQL table,
    and returns a LlamaIndex QueryEngineTool using the CustomSQLEngineWrapper.
    Handles errors gracefully.
    """
    tool = None # Initialize tool as None
    logger.info(f"--- Starting CSV Tool Creation for: {csv_file_name} ---")

    if df is None or df.empty:
        logger.warning(f"DataFrame for '{csv_file_name}' is empty. Skipping tool creation.")
        st.warning(f"Skipping empty CSV file: '{csv_file_name}'")
        return None
    if not sql_alchemy_engine:
        logger.error("SQLAlchemy engine is not available. Cannot create CSV tool.")
        st.error("Database engine error. Cannot process CSV files.")
        return None

    file_hash = pd.util.hash_pandas_object(df, index=True).sum() # Hash based on content
    sanitized_base_name = sanitize_for_name(csv_file_name)
    # Create a unique table name using sanitized name and hash segment
    table_name = f"csv_tbl_{sanitized_base_name}_{file_hash % 10000:04d}" # Modulo hash for shorter suffix
    tool_name = f"csv_{sanitized_base_name}_tool"

    logger.info(f"CSV: '{csv_file_name}'")
    logger.info(f"SQL Table Name: {table_name}")
    logger.info(f"Tool Name: {tool_name}")

    try:
        # 1. Clean Column Names for SQL compatibility
        original_columns = df.columns.tolist()
        cleaned_column_map = {}
        seen_cleaned_names = set()
        for i, col in enumerate(df.columns):
            # Aggressive cleaning: replace non-alphanum with underscore, lowercase, ensure starts with letter/underscore
            cleaned_col = re.sub(r'\W+|^(?=\d)', '_', str(col)).lower().strip('_')
            # Handle empty column names after cleaning
            if not cleaned_col: cleaned_col = f"column_{i}"
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
        logger.info(f"Original columns: {original_columns}")
        logger.info(f"Cleaned SQL columns: {cleaned_columns}")

        # 2. Load DataFrame into SQL Table
        with st.spinner(f"Loading data from '{csv_file_name}' into database table '{table_name}'..."):
            try:
                # Define dtype mapping for text/object columns to avoid potential issues
                dtype_mapping = {col: sqlalchemy.types.TEXT for col in df_renamed.select_dtypes(include=['object', 'string', 'datetime', 'timedelta']).columns}
                # Increase chunksize for potentially faster loading
                df_renamed.to_sql(
                    name=table_name,
                    con=sql_alchemy_engine,
                    index=False,
                    if_exists='replace', # Replace table if it already exists from a previous run
                    chunksize=5000,
                    dtype=dtype_mapping
                 )
                logger.info(f"Successfully loaded DataFrame {df_renamed.shape} into SQL table '{table_name}'.")
            except sqlalchemy.exc.SQLAlchemyError as db_load_err:
                 logger.error(f"Database error loading data for '{csv_file_name}' into table '{table_name}': {db_load_err}", exc_info=True)
                 st.error(f"Database error loading data from '{csv_file_name}': {db_load_err}")
                 return None # Cannot proceed if data loading fails
            except Exception as load_err:
                 logger.error(f"General error loading data for '{csv_file_name}' to SQL: {load_err}", exc_info=True)
                 st.error(f"Error loading data from '{csv_file_name}' to database: {load_err}")
                 return None

        # 3. Generate Table Description for LLM context
        logger.info(f"Generating description for table '{table_name}'...")
        table_desc = generate_table_description(df_renamed, table_name, csv_file_name)
        # logger.debug(f"Generated Table Description:\n{table_desc}") # Debug log if needed

        # 4. Instantiate Custom SQL Engine and Wrapper
        logger.info("Instantiating Custom SQL Engine...")
        # CustomSQLEngine now implicitly uses Settings.llm
        custom_sql_engine_instance = CustomSQLEngine(
            sql_engine=sql_alchemy_engine,
            table_name=table_name,
            table_description=table_desc,
            verbose=True # Enable verbose logging within the SQL engine
        )

        # Wrap the custom engine for LlamaIndex compatibility
        wrapped_engine = CustomSQLQueryEngineWrapper(custom_sql_engine_instance)
        logger.info("Custom SQL engine wrapped successfully.")

        # 5. Create Tool Metadata
        tool_description = (
            f"Performs SQL queries against the table named '{table_name}', which contains data from the CSV file '{csv_file_name}'. "
            f"Use this tool for tasks involving structured data lookup, filtering rows based on conditions, calculating aggregates (like sum, count, average, min, max), "
            f"or retrieving specific data points from this table. "
            f"Available columns: {', '.join(cleaned_columns)}."
        )

        # 6. Create the Query Engine Tool
        csv_tool = QueryEngineTool(
            query_engine=wrapped_engine,
            metadata=ToolMetadata(name=tool_name, description=tool_description)
        )
        logger.info(f"Successfully created QueryEngineTool: '{tool_name}' for '{csv_file_name}'.")
        tool = csv_tool # Assign the created tool

    except Exception as e:
        logger.error(f"Critical error during CSV tool setup pipeline for '{csv_file_name}': {e}", exc_info=True)
        st.error(f"Failed to set up processing for CSV '{csv_file_name}': {e}")
        # Ensure tool remains None

    logger.info(f"--- Finished CSV Tool Creation for: {csv_file_name} ---")
    return tool # Return the created tool or None if any step failed


# ==============================================================================
# --- Main Engine Setup Function (Merged - Handles Multiple Files) ---
# ==============================================================================
def setup_main_engine(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    qdrant_client_instance: qdrant_client.QdrantClient,
    enable_image_analysis: bool
) -> Tuple[Optional[SubQuestionQueryEngine], List[str]]:
    """
    Sets up the main SubQuestionQueryEngine by processing uploaded PDF and CSV files,
    creating individual tools for each, and handling cleanup and errors.
    """
    st.info("🚀 Starting engine setup...")
    start_time = datetime.datetime.now()
    agent_tools = []
    processed_filenames = []
    global_settings_ok = configure_global_settings() # Ensure settings are loaded

    if not global_settings_ok:
        st.error("Engine setup failed: Core component configuration error.")
        logger.critical("Aborting engine setup due to global settings configuration failure.")
        return None, []

    # Ensure Qdrant client is available for PDFs
    if not qdrant_client_instance:
         st.error("Vector database (Qdrant) client is not available. Cannot process PDF files.")
         logger.error("Qdrant client unavailable, skipping PDF processing.")
         # Allow proceeding with only CSVs if any exist

    # --- Resource Cleanup (Optional but recommended) ---
    # Consider if cleanup is desired on every run. Might be too aggressive.
    # For now, we rely on `if_exists='replace'` for SQL and checking Qdrant collections.
    # Cleanup could be added here if explicit deletion of old files/DBs is needed.
    # Example (Use with caution):
    # if SQL_DB_PATH.exists():
    #     try: logger.warning(f"Removing existing SQLite DB: {SQL_DB_PATH}"); os.remove(SQL_DB_PATH)
    #     except OSError as e: logger.error(f"Failed to remove old DB: {e}")
    # ... Qdrant cleanup (more complex, requires listing/deleting collections by prefix) ...

    # --- Setup SQLAlchemy Engine for CSVs ---
    sql_alchemy_engine = None
    csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]
    if csv_files:
        try:
            logger.info(f"Creating SQLAlchemy engine for SQLite DB at: {SQL_DB_URL}")
            sql_alchemy_engine = sqlalchemy.create_engine(SQL_DB_URL)
            # Test connection
            with sql_alchemy_engine.connect() as conn:
                logger.info("SQLAlchemy engine connected successfully.")
        except Exception as db_eng_err:
            logger.critical(f"Failed to create or connect SQLAlchemy engine: {db_eng_err}", exc_info=True)
            st.error(f"Database engine error: {db_eng_err}. Cannot process CSV files.")
            sql_alchemy_engine = None # Ensure it's None if connection fails

    # --- Process Files and Create Tools ---
    files_to_process = uploaded_files # Process all uploaded files
    total_files = len(files_to_process)
    processed_ok_count = 0
    processed_fail_count = 0

    if not files_to_process:
         st.warning("No files uploaded to process.")
         return None, []

    main_progress = st.progress(0.0, text="Starting file processing...")

    for i, uploaded_file in enumerate(files_to_process):
        file_name = uploaded_file.name
        logger.info(f"Processing file {i+1}/{total_files}: {file_name}")
        main_progress.progress((i + 1) / total_files, text=f"Processing: {file_name} ({i+1}/{total_files})")
        tool = None
        try:
            if file_name.lower().endswith('.pdf'):
                if qdrant_client_instance:
                    tool = create_pdf_tool(uploaded_file, qdrant_client_instance, enable_image_analysis)
                else:
                     logger.warning(f"Skipping PDF '{file_name}' because Qdrant client is unavailable.")
                     st.warning(f"Skipping PDF '{file_name}': Vector DB client not ready.")
                     processed_fail_count += 1
                     continue # Skip to next file

            elif file_name.lower().endswith('.csv'):
                if sql_alchemy_engine:
                     # Process CSV reads the file and returns a DataFrame or None
                     df = process_csv(uploaded_file)
                     if df is not None:
                         tool = create_csv_tool(df, file_name, sql_alchemy_engine)
                     else:
                          # process_csv already logged error/warning
                          processed_fail_count += 1
                          continue # Skip to next file
                else:
                     logger.warning(f"Skipping CSV '{file_name}' because SQLAlchemy engine failed.")
                     st.warning(f"Skipping CSV '{file_name}': Database engine not ready.")
                     processed_fail_count += 1
                     continue # Skip to next file
            else:
                logger.warning(f"Skipping unsupported file type: {file_name}")
                st.warning(f"Skipping unsupported file: {file_name}")
                processed_fail_count += 1
                continue # Skip to next file

            # If tool creation was successful, add it
            if tool:
                agent_tools.append(tool)
                processed_filenames.append(file_name)
                processed_ok_count += 1
                logger.info(f"Successfully created tool for: {file_name}")
            else:
                 # Tool creation function already logged the error and showed st.error
                 logger.error(f"Tool creation failed for: {file_name}")
                 processed_fail_count += 1

        except Exception as process_loop_err:
             logger.error(f"Unhandled error during processing loop for file '{file_name}': {process_loop_err}", exc_info=True)
             st.error(f"An unexpected error occurred while processing '{file_name}'.")
             processed_fail_count += 1

    main_progress.empty() # Clear the main progress bar

    # --- Report Processing Summary ---
    st.write(f"File Processing Summary:")
    st.write(f"- Successfully processed: {processed_ok_count} file(s)")
    if processed_fail_count > 0:
         st.write(f"- Failed or skipped: {processed_fail_count} file(s)")
    if processed_ok_count == 0:
        st.error("Engine setup failed: No tools could be created from the uploaded files. Check logs for details.")
        return None, []
    else:
         st.success(f"Created {len(agent_tools)} tools for: {', '.join(processed_filenames) or 'None'}")
         logger.info(f"Total tools created: {len(agent_tools)}. Files processed successfully: {processed_filenames}")

    # --- Build the Main Query Engine ---
    st.info("Building the main query engine...")
    logger.info("Creating SubQuestionQueryEngine...")
    try:
        # Ensure LLM is available in settings
        if not Settings.llm:
            raise ValueError("LLM not configured in Settings. Cannot create SubQuestionQueryEngine.")

        # use_async=False might be more stable, especially with complex sync/async interactions
        # verbose=True helps in debugging the sub-question process
        final_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=agent_tools,
            llm=Settings.llm, # Explicitly pass LLM from settings
            verbose=True,
            use_async=False # Recommended for stability unless async is strictly needed
        )
        logger.info("SubQuestionQueryEngine created successfully.")
        st.success("Query Engine is ready!")
        end_time = datetime.datetime.now()
        st.caption(f"Total setup time: {(end_time - start_time).total_seconds():.2f} seconds.")
        return final_engine, processed_filenames

    except Exception as e:
        logger.critical(f"Failed to initialize SubQuestionQueryEngine: {e}", exc_info=True)
        st.error(f"Query Engine creation failed: {e}")
        return None, processed_filenames # Return filenames even if engine fails, for context

# ==============================================================================
# --- Streamlit App UI (Merged & Enhanced) ---
# ==============================================================================
def main_app():
    # --- Page Config ---
    st.set_page_config(
        page_title="Unified Multi-Modal RAG Engine",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Unified Document Analysis Engine")
    st.markdown("""
    Upload PDF and/or CSV files. Ask questions about their content.
    The engine analyzes text, tables, and images (if enabled) across your documents.
    """)

    # --- Initialize Resources & Session State ---
    init_feedback_csv() # Ensure feedback file exists
    qdrant_client_instance = setup_qdrant_client() # Get cached Qdrant client

    # Initialize session state keys robustly
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("chat_messages", []) # Stores dicts: {role, content, sources?, dataframe?, sql?}
    st.session_state.setdefault("query_engine", None)
    st.session_state.setdefault("processed_filenames", [])
    st.session_state.setdefault("processing_complete", False) # Flag if processing attempted
    st.session_state.setdefault("engine_ready", False) # Flag if engine built successfully
    st.session_state.setdefault("initial_greeting_sent", False)
    # Add state for the image analysis checkbox
    st.session_state.setdefault("enable_image_analysis", False)

    # --- Login Check ---
    if not check_login():
        show_login()
        return # Stop further execution until logged in

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")
        st.markdown(f"Logged in as: **{st.session_state.user_email}**")
        if st.button("Logout", key="logout_button"):
            logger.info(f"User logged out: {st.session_state.user_email}")
            # Clear relevant session state keys upon logout
            keys_to_clear = [
                "logged_in", "user_email", "chat_messages", "query_engine",
                "processed_filenames", "processing_complete", "engine_ready",
                "initial_greeting_sent"
                # Keep 'enable_image_analysis' maybe? Or reset? Let's reset.
                ,"enable_image_analysis"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            # Reset checkbox state explicitly after clearing
            st.session_state.enable_image_analysis = False
            st.rerun()

        st.divider()

        # 1. File Uploader
        st.header("1. Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDF and/or CSV files",
            type=['pdf', 'csv'],
            accept_multiple_files=True,
            key="file_uploader"
        )

        # 2. Image Analysis Option (Checkbox)
        st.header("2. Options")
        # Use the session state variable to control the checkbox
        # The state is updated automatically by Streamlit when checked/unchecked
        st.session_state.enable_image_analysis = st.checkbox(
            "Enable Image Analysis (Slower Processing)",
            value=st.session_state.enable_image_analysis, # Set initial value from state
            key="image_analysis_checkbox",
            help="If checked, the system will attempt to analyze images within PDFs using an AI model (requires setup). This significantly increases processing time."
        )
        logger.debug(f"Image analysis checkbox state: {st.session_state.enable_image_analysis}")

        # 3. Process Button
        st.header("3. Process Files")
        process_button_disabled = not uploaded_files
        if not qdrant_client_instance:
            st.error("Vector DB client failed. PDF processing disabled.")
            # Disable button if only PDFs are uploaded and Qdrant failed? Maybe allow CSV processing.
            # Let's disable generally if Qdrant fails, as it's a core component assumption.
            process_button_disabled = True

        if st.button("Process Uploaded Files", type="primary", disabled=process_button_disabled, key="process_button"):
            logger.info(f"Process button clicked. Uploaded files: {[f.name for f in uploaded_files]}. Image analysis: {st.session_state.enable_image_analysis}")
            # --- Reset state before processing ---
            st.session_state.query_engine = None
            st.session_state.chat_messages = []
            st.session_state.processed_filenames = []
            st.session_state.processing_complete = False
            st.session_state.engine_ready = False
            st.session_state.initial_greeting_sent = False
            # --- End Reset ---

            if uploaded_files:
                with st.spinner("Processing files and building engine... This may take some time."):
                    # Pass the checkbox state to the setup function
                    engine_instance, processed_names = setup_main_engine(
                        uploaded_files,
                        qdrant_client_instance,
                        st.session_state.enable_image_analysis # Pass current state
                    )
                st.session_state.query_engine = engine_instance
                st.session_state.processed_filenames = processed_names
                st.session_state.processing_complete = True # Mark that processing was attempted
                if engine_instance is not None:
                    st.session_state.engine_ready = True # Mark engine as successfully built
                    logger.info("Engine setup successful and stored in session state.")
                else:
                    st.session_state.engine_ready = False
                    logger.error("Engine setup returned None. Engine is not ready.")
                 # Rerun to update main UI after processing attempt
                st.rerun()
            else:
                st.warning("Please upload files before processing.")

        # --- Configuration Info ---
        st.divider()
        st.header("System Info")
        st.info(f"LLM: {LLM_MODEL_NAME} (Placeholder)")
        st.info(f"Embedding: {EMBEDDING_MODEL_NAME}")
        if qdrant_client_instance:
            st.info(f"Vector Store: Qdrant (Path: {QDRANT_PATH})")
        else:
            st.warning("Vector Store: Qdrant Client Failed!")
        st.info(f"CSV DB: SQLite ({SQL_DB_FILENAME})")
        st.caption(f"Base Path: {BASE_PATH}")


    # --- Main Chat Area ---
    st.header("Chat Interface")

    # Get current state variables
    final_engine = st.session_state.get('query_engine', None)
    processed_files_list = st.session_state.get('processed_filenames', [])
    engine_is_ready = st.session_state.get('engine_ready', False)
    processing_was_attempted = st.session_state.get('processing_complete', False)
    initial_greeting_sent = st.session_state.get('initial_greeting_sent', False)

    # --- Initial Greeting Logic ---
    if engine_is_ready and not initial_greeting_sent:
        with st.chat_message("assistant"):
            greeting_msg = f"Hello {st.session_state.user_email}! I have processed {len(processed_files_list)} file(s): `{', '.join(processed_files_list)}`.\n\n"
            if len(processed_files_list) > 1:
                greeting_msg += "**Tip:** Since multiple files are loaded, asking questions that mention specific filenames usually yields better results (e.g., 'Summarize `report.pdf`' or 'What is the total revenue in `sales_data.csv`?').\n\n"
            greeting_msg += "How can I help you analyze these documents?"
            st.markdown(greeting_msg)
            # Add greeting to history (without sources)
            st.session_state.chat_messages.append({"role": "assistant", "content": greeting_msg})
            st.session_state.initial_greeting_sent = True # Mark as sent


    # --- Display Chat History ---
    if st.session_state.chat_messages:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                response_content = message["content"]
                # --- User Message Rendering ---
                if message["role"] == "user":
                    st.markdown(response_content)
                # --- Assistant Message Rendering (Adapted from Code 2) ---
                elif message["role"] == "assistant":
                    logger.debug(f"Rendering assistant message: {response_content[:100]}...")
                    last_end = 0
                    try:
                        # Regex to find image markers: [Image: /path/to/image.ext]
                        # Make path matching slightly more robust (allows spaces if quoted, etc.)
                        image_pattern = r'\[Image:\s*([^\]]+?)\s*\]'
                        for match in re.finditer(image_pattern, response_content):
                            start, end = match.span()
                            # Display text before the marker
                            if start > last_end:
                                st.markdown(response_content[last_end:start], unsafe_allow_html=True)

                            image_path_str = match.group(1).strip()
                            # Attempt to resolve the path for robustness
                            try:
                                 # Assume paths stored are absolute as per process_pdf_with_fitz
                                 image_path = Path(image_path_str)
                                 is_valid_file = image_path.is_file()
                            except Exception as path_err: # Catch potential path errors
                                 logger.error(f"Error parsing image path '{image_path_str}': {path_err}")
                                 image_path = None
                                 is_valid_file = False

                            logger.debug(f"Found image marker for path: {image_path_str} (Resolved: {image_path}, Exists: {is_valid_file})")

                            if is_valid_file:
                                try:
                                    st.image(str(image_path), width=400) # Display image inline
                                    # Generate download link
                                    try:
                                        filename = image_path.name
                                        mime_type = f"image/{image_path.suffix.lstrip('.').lower()}"
                                        # Basic mime type guessing
                                        if mime_type not in ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp', 'image/bmp', 'image/svg+xml']:
                                            mime_type = 'application/octet-stream' # Fallback
                                        with open(image_path, "rb") as f: img_bytes_dl = f.read()
                                        b64_encoded_dl = base64.b64encode(img_bytes_dl).decode()
                                        # Use st.download_button for better UX
                                        st.download_button(
                                             label=f"Download {filename}",
                                             data=img_bytes_dl,
                                             file_name=filename,
                                             mime=mime_type,
                                             key=f"dl_{image_path_str}_{start}" # Unique key per image instance
                                        )
                                    except Exception as dl_err:
                                        logger.error(f"Error creating download button for {image_path}: {dl_err}", exc_info=True)
                                        st.caption("(Could not create download link)")
                                except Exception as img_disp_err:
                                     logger.error(f"Error displaying image {image_path} with st.image: {img_disp_err}", exc_info=True)
                                     st.error(f"Error displaying image: {image_path.name}")
                            elif image_path_str:
                                st.error(f"Image file not found at path: {image_path_str}")
                                logger.error(f"Render Error: Image file not found at {image_path_str}")
                            else:
                                st.error("Invalid image marker format found in response.")
                                logger.error("Render Error: Invalid image marker format in response.")

                            last_end = end # Move past the processed marker

                        # Display any remaining text after the last marker
                        if last_end < len(response_content):
                            st.markdown(response_content[last_end:], unsafe_allow_html=True)

                    except Exception as render_err:
                        logger.error(f"Critical error parsing/rendering assistant response: {render_err}", exc_info=True)
                        # Fallback: Display the raw content if parsing fails
                        st.markdown(response_content)
                        st.warning("Could not fully parse response for inline images/links.")

                    # --- Source Expander (Combined Logic) ---
                    sources_data = message.get("sources") # List of source nodes/metadata
                    dataframe_result = message.get("dataframe") # DataFrame from SQL query
                    sql_query_used = message.get("sql") # SQL query string

                    has_pdf_sources = any(isinstance(s, BaseNode) and 'pdf' in s.metadata.get('file_name','').lower() for s in sources_data or [])
                    has_csv_sources = dataframe_result is not None or sql_query_used is not None

                    if sources_data or has_csv_sources:
                        with st.expander("View Sources / Details"):
                             # Display SQL Info First if available
                            if sql_query_used:
                                st.markdown("**SQL Query Used:**")
                                st.code(sql_query_used, language="sql")
                            if dataframe_result is not None:
                                st.markdown("**Query Result Data:**")
                                if isinstance(dataframe_result, pd.DataFrame):
                                    # Use st.dataframe for interactive display
                                    st.dataframe(dataframe_result)
                                else:
                                     # Should not happen based on wrapper, but fallback
                                     st.text("Result data is not in expected table format.")
                                st.divider()

                             # Display PDF Node Sources if available
                            if sources_data:
                                st.markdown("**Retrieved PDF Context:**")
                                source_count = 0
                                for i, source_node in enumerate(sources_data):
                                    # Check if it's likely a PDF source node
                                    if isinstance(source_node, BaseNode) and source_node.metadata:
                                        source_count += 1
                                        metadata = source_node.metadata
                                        file_name_src = metadata.get('file_name', 'Unknown File')
                                        page_num_src = metadata.get('page_num', 'Unknown Page')
                                        chunk_id_src = source_node.id_ # Use node ID

                                        st.markdown(f"**Source {source_count}:** File '{file_name_src}', Page {page_num_src} (Node ID: {chunk_id_src})")
                                        # Display text content (use text from node)
                                        st.markdown(f"> {source_node.get_content()}")

                                        # Display URLs from metadata
                                        urls_src = metadata.get("urls", [])
                                        if urls_src: st.markdown(f"**URLs:** {', '.join(urls_src)}")

                                        # Display Images from metadata
                                        images_in_source = metadata.get("images", [])
                                        if images_in_source:
                                            st.markdown("**Associated Images (from source context):**")
                                            # Use columns for compact layout
                                            cols = st.columns(min(len(images_in_source), 4)) # Max 4 cols
                                            for j, img_data in enumerate(images_in_source):
                                                with cols[j % 4]:
                                                    img_filename = img_data.get('filename', 'N/A')
                                                    img_path = img_data.get("path")
                                                    img_expl = img_data.get("explanation", "N/A")
                                                    # Display filename caption
                                                    st.caption(f"{img_filename}")
                                                    # Display image preview using helper
                                                    st.markdown(display_image_expander(img_path, img_filename), unsafe_allow_html=True)
                                                    # Display explanation caption (truncated)
                                                    st.caption(f"Expl: {img_expl[:80]}{'...' if len(img_expl)>80 else ''}")
                                        st.divider()
                                if source_count == 0 and not has_csv_sources:
                                     st.markdown("No specific source context details available.")
    else:
        # Show message if chat history is empty AFTER processing is complete
        if processing_was_attempted and not engine_is_ready:
             st.warning("Engine setup failed or no files were successfully processed. Please check the sidebar for errors and upload valid files.")
        elif not processing_was_attempted:
             st.info("Upload files and click 'Process Uploaded Files' in the sidebar to begin.")


    # --- Chat Input Area ---
    chat_input_disabled = not engine_is_ready
    if chat_input_disabled:
        if processing_was_attempted:
            prompt_placeholder = "Engine not ready. Check processing status."
        else:
            prompt_placeholder = "Upload and process files to enable chat."
    else:
        prompt_placeholder = "Ask a question about the uploaded files..."

    if prompt := st.chat_input(prompt_placeholder, key="chat_prompt_input", disabled=chat_input_disabled):
        logger.info(f"User '{st.session_state.user_email}' asked: {prompt}")
        # Add user message to history immediately
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        # Rerun to display the user message instantly
        st.rerun()

    # --- Handle Query Processing (Triggered AFTER user input causes rerun) ---
    # Check if the last message is from the user, indicating we need to generate a response
    if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
        last_user_prompt = st.session_state.chat_messages[-1]["content"]

        # Check if engine is ready one last time before querying
        if final_engine and engine_is_ready:
            with st.chat_message("assistant"):
                # Use placeholders for immediate feedback during generation
                thinking_placeholder = st.empty()
                response_placeholder = st.empty()
                thinking_placeholder.markdown("Thinking...")

                response_obj = None
                error_msg = None
                start_query_time = time.time()
                try:
                    logger.info(f"--- Querying SubQuestionQueryEngine: {last_user_prompt} ---")
                    # Query the main engine
                    response_obj = final_engine.query(last_user_prompt)
                    logger.info(f"--- SubQuestionQueryEngine response received ---")
                    # logger.debug(f"Raw response object: {response_obj}") # Verbose debug

                except Exception as query_err:
                    logger.error(f"Engine query failed: {query_err}", exc_info=True)
                    error_msg = f"Sorry, an error occurred during the query: {query_err}"
                    # Log error to feedback
                    log_feedback(
                        st.session_state.user_email, last_user_prompt, processed_files_list,
                        "Query Error", error_info=str(query_err)
                    )

                end_query_time = time.time()
                logger.info(f"Engine query took {end_query_time - start_query_time:.2f} seconds.")

                # Process the response or error
                assistant_response_content = ""
                retrieved_sources_for_display = []
                dataframe_for_display = None
                sql_for_display = None

                if error_msg:
                    assistant_response_content = error_msg
                elif response_obj and isinstance(response_obj, Response):
                    assistant_response_content = response_obj.response or "No textual response generated."
                    retrieved_sources_for_display = response_obj.source_nodes # Get source nodes directly

                    # Check metadata for SQL results from our wrapper
                    if response_obj.metadata:
                         dataframe_for_display = response_obj.metadata.get("raw_dataframe")
                         sql_for_display = response_obj.metadata.get("sql_query")
                         # If dataframe exists, log it differently or summarize for text log
                         # Maybe add summary to assistant response?
                         # if dataframe_for_display is not None:
                         #    assistant_response_content += f"\n\n(Retrieved data with {len(dataframe_for_display)} rows from query)"

                         # Log feedback info
                         log_sources_pdf = "; ".join([f"Node:{n.id_}, Pg:{n.metadata.get('page_num','?')}" for n in retrieved_sources_for_display]) if retrieved_sources_for_display else "N/A"
                         log_sources_csv = f"SQL: {sql_for_display}" if sql_for_display else "N/A"
                         log_feedback(
                            st.session_state.user_email, last_user_prompt, processed_files_list,
                            assistant_response_content, log_sources_pdf, log_sources_csv
                         )
                else:
                    assistant_response_content = "Sorry, the engine returned an unexpected response type or no response."
                    logger.warning(f"Unexpected response type from engine: {type(response_obj)}")
                    # Log feedback info for unexpected response
                    log_feedback(
                        st.session_state.user_email, last_user_prompt, processed_files_list,
                        assistant_response_content, error_info="Unexpected response type"
                     )


                # Clear the "Thinking..." message
                thinking_placeholder.empty()

                # Add the complete assistant message to history state
                # We need to update the *last* message which was added as 'user'
                # A better pattern: Don't rerun after user input. Process query directly,
                # then add user AND assistant message, then rerun ONCE.
                # Let's stick to the current pattern for now, but it means we add a new assistant msg.
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": assistant_response_content,
                    "sources": retrieved_sources_for_display,
                    "dataframe": dataframe_for_display,
                    "sql": sql_for_display
                })

                # Rerun one last time to display the complete assistant message and sources
                st.rerun()

        else:
            # Handle case where engine is somehow not ready when processing user input
             st.error("Query engine is not ready. Cannot process query.")
             logger.error("Attempted to process user query but engine is not ready.")
             # Remove the last user message to prevent reprocessing loop? Or just display error.


# ==============================================================================
# --- Application Entry Point ---
# ==============================================================================
if __name__ == "__main__":
    try:
        logger.info("--- Starting Merged Application ---")
        # Configure global settings early
        if not configure_global_settings():
             logger.critical("Application stopping due to failed global settings configuration.")
             st.error("Application failed to initialize core settings. Please check logs.")
             # Don't call st.stop() here, allow potential error display
        else:
             main_app() # Run the main Streamlit app function
    except (SystemExit, KeyboardInterrupt):
        logger.info("Application stopped gracefully.")
    except Exception as e:
        logger.critical(f"Application crashed critically: {e}", exc_info=True)
        # Try to display error in Streamlit if possible
        try: st.error(f"A critical application error occurred: {e}")
        except: pass # Avoid errors if Streamlit itself is broken

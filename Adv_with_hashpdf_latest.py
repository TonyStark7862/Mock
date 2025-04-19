import streamlit as st
# from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import pickle
import hashlib
import os
from pathlib import Path
import io
import re # Import regex library
import csv
import logging
import datetime
from logging.handlers import RotatingFileHandler
import fitz  # PyMuPDF
import base64
from PIL import Image
import tempfile
import time # For simulating delays

# --- Global Base Path Definition ---
# Get the absolute path to the directory containing this script
BASE_PATH = Path(__file__).parent.resolve()
print(f"BASE_PATH detected as: {BASE_PATH}") # Debug print

# --- Derived Paths ---
LOG_DIR = BASE_PATH / "logs"
FEEDBACK_DIR = BASE_PATH / "feedback"
IMAGES_DIR = BASE_PATH / "images"
VECTORDB_DIR = BASE_PATH / "vectordb"
LOCAL_MODEL_PATH = BASE_PATH / "models" / "all-MiniLM-L6-v2"

# --- Create Directories ---
LOG_DIR.mkdir(exist_ok=True)
FEEDBACK_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
VECTORDB_DIR.mkdir(exist_ok=True)
LOCAL_MODEL_PATH.parent.mkdir(exist_ok=True, parents=True) # Ensure model parent dir exists

# Configure logging (using derived path)
LOG_FILE = LOG_DIR / "app.log"
logger = logging.getLogger("rag_pdf_chat")
logger.setLevel(logging.INFO)
# Clear existing handlers if any (useful for Streamlit re-runs)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(LOG_FILE)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)

# Feedback file path (using derived path)
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.csv"

# Flag to determine which vector database to use
USE_FAISS = False  # Set to False to use Qdrant

# --- Global Resources Setup (Cached) ---

@st.cache_resource
def load_sentence_transformer():
    """Loads/downloads Sentence Transformer model."""
    model_name = 'all-MiniLM-L6-v2'
    try:
        with st.spinner("Loading embedding model..."):
            if LOCAL_MODEL_PATH.exists():
                 model = SentenceTransformer(str(LOCAL_MODEL_PATH))
                 logger.info("Loaded embedding model from local path")
                 st.success("✅ Embedding model ready.") # Show success once
                 return model
            else:
                 logger.warning(f"Downloading {model_name}...")
                 # Show spinner during download
                 with st.spinner(f"Downloading {model_name}... This may take a moment."):
                      model = SentenceTransformer(model_name)
                      model.save(str(LOCAL_MODEL_PATH))
                 logger.info(f"Downloaded and saved model to {LOCAL_MODEL_PATH}")
                 st.success("✅ Embedding model ready.") # Show success once
                 return model
    except Exception as e:
        logger.error(f"Fatal: Could not load/download embedding model: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not load embedding model: {e}")
        st.stop()

@st.cache_resource
def setup_qdrant_client():
    """Sets up Qdrant client."""
    try:
        qdrant_path = VECTORDB_DIR / "qdrant_db"
        logger.info(f"Setting up Qdrant client at: {qdrant_path}")
        # Show spinner during setup
        with st.spinner("Connecting to Vector Database..."):
             client = QdrantClient(path=str(qdrant_path))
             client.get_collections() # Verify connection
        logger.info("Qdrant client setup verified.")
        st.success("✅ Vector Database connected.") # Show success once
        return client
    except Exception as e:
        logger.error(f"Fatal: Error setting up Qdrant: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not connect to Vector DB: {e}")
        st.stop() # Stop if DB connection fails

# --- Initialize resources globally if not using FAISS ---
# These will be loaded once per session start due to @st.cache_resource
if not USE_FAISS:
    qdrant_client_global = setup_qdrant_client()
    embedding_model_global = load_sentence_transformer()

# --- Placeholder for Image Explanation Model ---
def image_response(image_bytes, prompt="Describe this image."):
    """Placeholder: Replace with actual image understanding model call."""
    logger.info(f"Simulating image explanation for prompt: {prompt}")
    time.sleep(0.1) # Shorter simulation delay
    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
    return f"Placeholder explanation for image (hash: {image_hash}). Appears relevant to '{prompt[:30]}...'."

# Initialize feedback CSV
def init_feedback_csv():
    if not FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'email', 'question', 'file_name', 'response', 'sources_info'])
        logger.info(f"Created new feedback file at {FEEDBACK_FILE}")

# Log feedback to CSV
def log_feedback(email, question, file_name, response, sources_info="N/A"):
    try:
        with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Sanitize response slightly for CSV logging if needed
            response_log = response.replace('\n', '\\n').replace('\r', '')
            writer.writerow([timestamp, email, question, file_name, response_log, sources_info])
        logger.info(f"Logged feedback from {email}")
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")

# Validate email format
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# Check login state
def check_login():
    return st.session_state.get("logged_in", False)

# Login page
def show_login():
    st.title("Login to RAG PDF Chat")
    email = st.text_input("Email", key="login_email")
    if st.button("Login"):
        if validate_email(email):
            st.session_state.logged_in = True
            st.session_state.user_email = email
            logger.info(f"User logged in: {email}")
            st.rerun()
        else:
            st.error("Please enter a valid email address")
            logger.warning(f"Invalid login attempt with email: {email}")

# --- PDF Processing Functions ---

def process_pdf_with_fitz(file_bytes, file_hash):
    """Extract text, images, URLs. Generate explanations. Store ABSOLUTE paths."""
    try:
        logger.info("Starting PDF processing with PyMuPDF...")
        pdf_assets_dir = IMAGES_DIR / file_hash
        pdf_assets_dir.mkdir(exist_ok=True, parents=True)
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        pages_content = []
        page_count_total = pdf_document.page_count
        progress_bar = st.progress(0)

        for page_num in range(page_count_total):
            page = pdf_document[page_num]
            text = page.get_text()
            links = page.get_links()
            urls = [link["uri"] for link in links if "uri" in link]
            image_list = page.get_images(full=True)
            images_data = []
            progress_bar.progress((page_num + 1) / page_count_total, text=f"Processing Page {page_num + 1}/{page_count_total}...")

            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = pdf_document.extract_image(xref)
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_filename = f'image_p{page_num + 1}_{img_index + 1}.{image_ext}'
                        absolute_image_path = pdf_assets_dir / image_filename
                        with open(absolute_image_path, 'wb') as img_file:
                            img_file.write(image_bytes)
                        explanation = image_response(image_bytes, f"Explain content of {image_filename} on page {page_num+1}")
                        images_data.append({
                            "path": str(absolute_image_path), "filename": image_filename,
                            "ext": image_ext, "explanation": explanation
                        })
                except Exception as img_proc_err:
                    logger.error(f"Error processing image xref {xref} on page {page_num+1}: {img_proc_err}")

            page_data = {"page_num": page_num + 1, "text": text, "urls": urls, "images": images_data}
            pages_content.append(page_data)

        progress_bar.empty()
        logger.info(f"Extracted content from {len(pages_content)} pages")
        return pages_content
    except Exception as e:
        logger.error(f"Error processing PDF with PyMuPDF: {e}", exc_info=True)
        st.error(f"Error during PDF processing: {e}")
        if 'progress_bar' in locals(): progress_bar.empty()
        raise e

def create_enhanced_chunks(pages_content, chunk_size=1000, chunk_overlap=200):
    """Create text chunks with metadata (page, images[path,expl], urls)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    enhanced_chunks = []
    current_chunk_id = 0
    for page_data in pages_content:
        page_text = page_data["text"]
        text_chunks = text_splitter.split_text(page_text)
        for chunk_text in text_chunks:
            enhanced_chunk = {
                "chunk_id": current_chunk_id, "text": chunk_text,
                "page_num": page_data["page_num"], "urls": page_data["urls"],
                "images": page_data["images"]
            }
            enhanced_chunks.append(enhanced_chunk)
            current_chunk_id += 1
    logger.info(f"Created {len(enhanced_chunks)} enhanced chunks")
    return enhanced_chunks

# --- Modified Qdrant PDF Processing (Check FIRST) ---
def process_pdf_qdrant(file_bytes, collection_name):
    """Process PDF and add to Qdrant, checking for existing data FIRST."""
    try:
        logger.info(f"Checking status for Qdrant collection: {collection_name}")
        qdrant_client = setup_qdrant_client() # Use cached client

        # --- CHECK FOR EXISTING DATA FIRST ---
        try:
            collection_info = qdrant_client.get_collection(collection_name=collection_name)
            if collection_info.points_count > 0:
                logger.info(f"Collection '{collection_name}' exists with {collection_info.points_count} points. Skipping processing.")
                st.info(f"Using existing data for '{Path(collection_name).name}'.") # Show user info
                return # Exit early
        except Exception as e:
            if "Not found: Collection" in str(e) or "doesn't exist" in str(e):
                 logger.info(f"Collection '{collection_name}' not found. Proceeding with full processing.")
            else:
                 logger.warning(f"Could not get collection info for {collection_name} (may proceed anyway): {e}")
        # --- END CHECK ---

        logger.info(f"Starting full processing for {collection_name}...")
        file_hash = hashlib.md5(file_bytes).hexdigest()

        with st.spinner("Extracting text, images, and generating explanations..."):
            pages_content = process_pdf_with_fitz(file_bytes, file_hash)
        with st.spinner("Creating text chunks..."):
            enhanced_chunks = create_enhanced_chunks(pages_content)

        # Check if chunks were created before proceeding
        if not enhanced_chunks:
            logger.warning(f"No chunks created for {collection_name}, possibly empty PDF?")
            st.warning("No text content found or processed in the PDF.")
            return

        model = load_sentence_transformer() # Use cached model

        with st.spinner(f"Creating embeddings for {len(enhanced_chunks)} chunks..."):
            text_contents = [chunk["text"] for chunk in enhanced_chunks]
            embeddings = model.encode(text_contents, show_progress_bar=True)

        with st.spinner("Uploading data to vector database..."):
            points_to_upload = []
            for idx, chunk in enumerate(enhanced_chunks):
                 payload = {
                    "text": chunk["text"], "chunk_id": chunk["chunk_id"],
                    "page_num": chunk["page_num"], "urls": chunk.get("urls", []),
                    "images": chunk.get("images", [])
                 }
                 points_to_upload.append(
                    models.PointStruct(id=chunk["chunk_id"], vector=embeddings[idx].tolist(), payload=payload)
                 )
            # Upsert in batches if necessary (Qdrant client handles batching internally for lists)
            qdrant_client.upsert(collection_name=collection_name, points=points_to_upload, wait=True)

        logger.info(f"Successfully added/updated {len(points_to_upload)} chunks in {collection_name}")
        st.success(f"✅ PDF processed. Added/Updated {len(points_to_upload)} document sections.")

    except Exception as e:
        logger.error(f"Error processing PDF for Qdrant: {e}", exc_info=True)
        st.error(f"Error processing PDF: {e}")
        raise e

# --- FAISS PDF Processing (Placeholder - logic moved to main_app) ---
# This function remains as it was, the check is done before calling it.
def process_pdf_faiss(file_bytes, file_hash):
    """Process PDF for FAISS with enhanced metadata."""
    try:
        logger.info("Processing PDF for FAISS index")
        with st.spinner("Extracting text, images, and generating explanations..."):
             pages_content = process_pdf_with_fitz(file_bytes, file_hash)
        with st.spinner("Creating text chunks..."):
             enhanced_chunks = create_enhanced_chunks(pages_content)

        # Check if chunks were created
        if not enhanced_chunks:
             logger.warning(f"No chunks created for FAISS index {file_hash}, possibly empty PDF?")
             st.warning("No text content found or processed in the PDF.")
             # Return None or raise an error, depending on desired behavior
             return None # Indicate failure

        with st.spinner("Creating embeddings..."):
             # Use HuggingFaceEmbeddings for FAISS as per original code
             embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
             texts = [chunk["text"] for chunk in enhanced_chunks]
             metadatas = [{"chunk_id": c["chunk_id"], "page_num": c["page_num"], "urls": c.get("urls", []), "images": c.get("images", [])} for c in enhanced_chunks]
        with st.spinner("Creating FAISS index..."):
             vector_store = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas)
        logger.info("PDF processed successfully with FAISS and enhanced metadata")
        return vector_store
    except Exception as e:
        logger.error(f"Error processing PDF with FAISS: {e}", exc_info=True)
        st.error(f"Error processing PDF with FAISS: {e}")
        raise e

# --- Display Image Function ---
def display_image(image_path):
    """Displays an image from an absolute path."""
    try:
        if not Path(image_path).is_file():
             logger.error(f"Image path is not a file or does not exist: {image_path}")
             return f"[Image not found: {Path(image_path).name}]"
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
            encoded_image = base64.b64encode(image_data).decode()
            # Slightly larger images in expander for better detail
            return f'<img src="data:image/png;base64,{encoded_image}" style="max-width: 300px; height: auto; display: block; margin: 5px; border: 1px solid #ddd;">'
    except Exception as e:
        logger.error(f"Error displaying image {image_path}: {e}")
        return f"[Error displaying image: {Path(image_path).name}]"

# --- Search Chunks (Qdrant) ---
def search_chunks(collection_name, query, limit=5):
    """Search for chunks similar to the query."""
    try:
        logger.info(f"Searching collection {collection_name} for: {query}")
        model = load_sentence_transformer() # Use cached model
        qdrant_client = setup_qdrant_client() # Use cached client
        query_embedding = model.encode([query])[0]
        search_results = qdrant_client.search(
            collection_name=collection_name, query_vector=query_embedding.tolist(),
            limit=limit, with_payload=True
        )
        logger.info(f"Found {len(search_results)} relevant chunks")
        return search_results
    except Exception as e:
        logger.error(f"Error searching chunks in {collection_name}: {e}", exc_info=True)
        st.error(f"Error searching chunks: {e}")
        return []

# --- Placeholder LLM Response ---
def abc_response(prompt_with_context):
    """ Placeholder LLM: Takes enhanced context, returns string with format markers."""
    logger.info("--- Sending prompt to Placeholder LLM ---")
    logger.info("--- End of Placeholder LLM Prompt ---")
    time.sleep(0.5)
    question_marker = "Question:"
    question_start = prompt_with_context.rfind(question_marker)
    question = prompt_with_context[question_start + len(question_marker):].strip() if question_start != -1 else "[Question not found]"
    simulated_answer = f"This is a simulated answer to: '{question}'.\n\n"
    simulated_answer += "Based on the context:\n- Point one mentioned.\n"
    img_match = re.search(r"Path: (.*?/images/.*?\.png)", prompt_with_context)
    if img_match:
         simulated_image_path = img_match.group(1)
         simulated_answer += f"- See illustration: [Image: {simulated_image_path}]\n"
    simulated_answer += "- More info at [Example](https://example.com).\n"
    logger.info(f"Placeholder LLM generated response: {simulated_answer}")
    return simulated_answer

# ========================================
# Main Streamlit Application Function
# ========================================
def main_app():
    st.set_page_config(page_title="RAG PDF Chat Pro", layout="wide")
    init_feedback_csv() # Ensure feedback file exists

    # --- Login Check ---
    if not check_login():
        show_login()
        return

    # --- Main UI ---
    st.title("Chat with your PDF")
    st.sidebar.success(f"Logged in as: {st.session_state.user_email}")

    # --- Sidebar ---
    with st.sidebar:
        if st.button("Logout", key="logout_button"):
            logger.info(f"User logged out: {st.session_state.user_email}")
            keys_to_keep = {'logged_in'}
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep: del st.session_state[key]
            st.session_state.logged_in = False
            st.rerun()

        st.markdown("---")
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose PDF file", type="pdf", key="pdf_uploader")

        # Initialize session state variables robustly
        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("retriever", None) # For FAISS
        st.session_state.setdefault("faiss_cache_path", None) # For FAISS
        st.session_state.setdefault("collection_name", None) # For Qdrant

        # --- PDF Processing Trigger ---
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            process_this_file = False
            target_collection_name = f"pdf_{file_hash}"
            target_faiss_cache = VECTORDB_DIR / f"{file_hash}.pkl"

            # Determine if processing is needed based on session state / cache existence
            if USE_FAISS:
                # For FAISS, check if cache exists first, then session state
                if not target_faiss_cache.exists():
                    process_this_file = True
                    st.session_state.retriever = None # Clear retriever if cache missing
                    st.session_state.faiss_cache_path = str(target_faiss_cache)
                elif st.session_state.retriever is None or st.session_state.faiss_cache_path != str(target_faiss_cache):
                     # Cache exists, but not loaded in session, trigger loading below
                     process_this_file = True # Set true to enter the block, but it will load cache
                     st.session_state.faiss_cache_path = str(target_faiss_cache)
                else:
                     logger.info(f"FAISS data for {uploaded_file.name} already loaded in session.")

            else: # Qdrant
                if st.session_state.collection_name != target_collection_name:
                    process_this_file = True
                    # We will check actual collection existence inside process_pdf_qdrant
                    st.session_state.collection_name = target_collection_name # Update session state

            # --- Processing Execution ---
            if process_this_file:
                logger.info(f"Initiating processing for: {uploaded_file.name} (Hash: {file_hash})")
                st.session_state.messages = [] # Clear messages for new file
                try:
                    if USE_FAISS:
                        # --- FAISS Processing / Cache Loading ---
                        if target_faiss_cache.exists() and st.session_state.retriever is None: # Load if cache exists but not in session
                            try:
                                with st.spinner(f"Loading cached data for {uploaded_file.name}..."):
                                     logger.info(f"Loading FAISS from cache: {target_faiss_cache}")
                                     with open(target_faiss_cache, "rb") as f: vector_store = pickle.load(f)
                                     st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                                     st.success("✅ Loaded data from cache (FAISS).")
                                     logger.info("Successfully loaded vector store from cache")
                            except Exception as e:
                                 logger.error(f"Error loading FAISS cache {target_faiss_cache}: {e}", exc_info=True)
                                 st.error(f"Error loading cache: {e}. Re-processing...")
                                 st.session_state.retriever = None # Ensure re-processing trigger
                                 # Fall through to process if cache load fails

                        if st.session_state.retriever is None: # Process if cache didn't exist or load failed
                             logger.info(f"Processing PDF for FAISS index: {uploaded_file.name}")
                             vector_store = process_pdf_faiss(file_bytes, file_hash)
                             if vector_store: # Check if processing was successful
                                 st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                                 logger.info(f"Saving FAISS cache: {target_faiss_cache}")
                                 with open(target_faiss_cache, "wb") as f: pickle.dump(vector_store, f)
                                 st.success("✅ PDF processed & cached (FAISS).")
                             else:
                                 # Handle case where process_pdf_faiss failed (e.g., empty PDF)
                                 st.error("Failed to process PDF for FAISS.")
                                 st.session_state.retriever, st.session_state.faiss_cache_path = None, None


                    else: # --- Qdrant Processing ---
                         # Client and model are loaded globally now
                         qdrant_client = qdrant_client_global
                         model = embedding_model_global
                         if qdrant_client and model:
                             # Create collection if it doesn't exist (check moved inside process_pdf_qdrant)
                             process_pdf_qdrant(file_bytes, st.session_state.collection_name)
                         else:
                             raise Exception("Qdrant client or embedding model not available.")

                except Exception as e:
                     logger.error(f"Fatal error during PDF processing trigger block for {uploaded_file.name}: {e}", exc_info=True)
                     st.error(f"Error processing PDF: {e}")
                     # Reset state on failure
                     if USE_FAISS: st.session_state.retriever, st.session_state.faiss_cache_path = None, None
                     else: st.session_state.collection_name = None
            else:
                 if uploaded_file: # Avoid logging this if no file is uploaded
                     logger.info(f"File '{uploaded_file.name}' already processed or loaded for session.")

        # --- Clear Chat Button ---
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.messages = []
            logger.info(f"User {st.session_state.user_email} cleared chat")
            st.rerun()

        st.markdown("---")
        st.header("Support")
        st.markdown("support@example.com")

    # --- Load Resources Globally (Done Outside main_app now) ---
    # Ensure variables are accessible if needed within main_app scope
    qdrant_client = None
    embedding_model = None
    if not USE_FAISS:
        qdrant_client = qdrant_client_global
        embedding_model = embedding_model_global

    # --- Display Chat History ---
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "assistant":
                response_content = message["content"]
                actual_sources_for_this_message = message.get("sources", [])
                logger.debug(f"Rendering assistant msg {message_index}. Sources: {len(actual_sources_for_this_message)}")

                # --- Response Parsing and Rendering Logic ---
                last_end = 0
                try:
                    image_pattern = r'\[Image:\s*(.*?)\s*\]' # Pattern to find [Image: path]
                    for match in re.finditer(image_pattern, response_content):
                        start, end = match.span()
                        image_path = match.group(1).strip()
                        if start > last_end: # Display text before marker
                            st.markdown(response_content[last_end:start], unsafe_allow_html=True)

                        # Validate path against sources and filesystem
                        path_is_valid_source = any(
                            img_data.get("path") == image_path
                            for src_chunk in actual_sources_for_this_message
                            for img_data in src_chunk.get("images", [])
                        ) if actual_sources_for_this_message and image_path else False

                        if path_is_valid_source:
                            if Path(image_path).is_file():
                                st.image(image_path, width=400) # Display valid image
                            else:
                                logger.error(f"Image path from sources not found on disk: {image_path}")
                                st.error(f"Image file not found: {Path(image_path).name}")
                        elif image_path:
                            logger.warning(f"LLM hallucinated image path: {image_path}")
                            # Silently ignore hallucinated paths in UI

                        last_end = end
                    if last_end < len(response_content): # Display remaining text
                        st.markdown(response_content[last_end:], unsafe_allow_html=True)
                except Exception as render_err:
                    logger.error(f"Error parsing/rendering assistant response: {render_err}", exc_info=True)
                    st.markdown(response_content, unsafe_allow_html=True) # Fallback
                    st.warning("Could not fully parse response for inline images.")
                # --- End Response Parsing ---

            # --- Document References Expander ---
            if "sources" in message and message["sources"]:
                with st.expander("View Document References"): # Renamed expander
                    for i, source in enumerate(message["sources"]):
                         st.markdown(f"**Reference {i+1} (Chunk ID: {source.get('chunk_id', 'N/A')}, Page: {source.get('page_num', 'Unknown')})**") # Renamed source
                         st.markdown(f"> {source.get('text', '')}") # Text context
                         urls = source.get("urls", [])
                         if urls: st.markdown(f"**URLs:** {', '.join(urls)}")
                         images = source.get("images", [])
                         if images: # Display images associated with this reference
                              st.markdown("**Images Referenced on Page:**")
                              # Use fewer columns for clarity in expander
                              cols = st.columns(min(len(images), 2))
                              for j, img_data in enumerate(images):
                                   with cols[j % 2]:
                                        st.markdown(f"_{img_data.get('filename', 'N/A')}_")
                                        img_path = img_data.get("path")
                                        # Use the display_image function for consistency
                                        if img_path: st.markdown(display_image(img_path), unsafe_allow_html=True)
                                        st.caption(f"{img_data.get('explanation', '')[:100]}...")
                         st.divider()

    # --- Chat Input Logic ---
    if prompt := st.chat_input("Ask a question about your PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        current_file_name = uploaded_file.name if uploaded_file else "No file uploaded"
        retrieved_sources = []
        context_for_llm = ""
        sources_info_log = "N/A" # Default log value

        # Check readiness
        is_ready = False
        if USE_FAISS: is_ready = st.session_state.retriever is not None
        else: is_ready = st.session_state.collection_name is not None and qdrant_client is not None

        if not is_ready:
            st.warning("Please upload and process a PDF file first.")
            # Add a placeholder message if needed, or just do nothing
            # response = "Please upload a document before asking questions."
            # st.session_state.messages.append({"role": "assistant", "content": response, "sources": []})
            # st.rerun()
        else:
            # --- Retrieval ---
            with st.spinner("Searching relevant document sections..."):
                try:
                    if USE_FAISS:
                        docs = st.session_state.retriever.get_relevant_documents(prompt)
                        for doc in docs:
                            meta = doc.metadata if hasattr(doc, 'metadata') else {}
                            retrieved_sources.append({"text": doc.page_content, **meta})
                    else: # Qdrant
                        search_results = search_chunks(st.session_state.collection_name, prompt)
                        for result in search_results:
                             payload = result.payload if hasattr(result, 'payload') else {}
                             retrieved_sources.append({**payload, "chunk_id": payload.get("chunk_id", result.id)})
                    logger.info(f"Retrieved {len(retrieved_sources)} sources.")
                except Exception as e:
                    logger.error(f"Error during retrieval: {e}", exc_info=True); st.error(f"Error searching document: {e}")
                    retrieved_sources = [] # Ensure sources is empty on error

            # --- Context/Prompt Prep & LLM Call ---
            if not retrieved_sources:
                 response = "I couldn't find specific information for your query in the document."
                 sources_info_log = "No sources found"
            else:
                # Build context string for LLM
                context_items = []
                sources_info_log_list = []
                for i, src in enumerate(retrieved_sources):
                     item = f"Reference {i+1} (Page:{src.get('page_num','N/A')}, Chunk:{src.get('chunk_id','N/A')}):\n" # Renamed source -> reference
                     item += f"Text: \"{src.get('text', '')}\"\n"
                     if src.get('urls'): item += f"URLs: {', '.join(src['urls'])}\n"
                     if src.get('images'):
                          item += "Images (Path | Explanation):\n"
                          for img in src['images']: item += f" - {img.get('path','N/A')} | {img.get('explanation','N/A')}\n"
                     item += "---\n"
                     context_items.append(item)
                     sources_info_log_list.append(f"Chunk:{src.get('chunk_id','N/A')}|Pg:{src.get('page_num','N/A')}")

                context_for_llm = "\n".join(context_items)
                sources_info_log = "; ".join(sources_info_log_list)

                # Define the comprehensive prompt for the LLM
                full_prompt = f"""You are an AI assistant analyzing a PDF based ONLY on the provided context (Document References).
Answer the user's question accurately using information solely from the context.
Integrate relevant details directly into your response:
- **URLs:** If a URL from a specific reference supports your point, include it in Markdown: [descriptive text](URL).
- **Images:** If an image (identified by its Path from a specific reference) is directly relevant, include its path using this exact format: [Image: /absolute/path/to/image.ext]. You can refer to its Explanation if useful.

Do not make up information. If the context doesn't contain the answer, state that clearly.

--- Start of Context (Document References) ---
{context_for_llm}
--- End of Context ---

Question: {prompt}

Answer:"""

                # Get response from LLM (Placeholder or Real)
                with st.spinner("Generating response..."):
                    try:
                        # Replace abc_response with your actual LLM call if you haven't already
                        response = abc_response(full_prompt)
                    except Exception as e:
                        logger.error(f"Error calling LLM: {e}", exc_info=True)
                        response = f"Sorry, I encountered an error while generating the response: {str(e)}"

            # --- Add Message to History & Log ---
            st.session_state.messages.append({
                "role": "assistant", "content": response, "sources": retrieved_sources
            })
            log_feedback(st.session_state.user_email, prompt, current_file_name, response, sources_info_log)
            st.rerun() # Trigger re-run to display the new message

# ========================================
# Run the Application
# ========================================
if __name__ == "__main__":
    try:
        logger.info(f"Starting application from main block. BASE_PATH: {BASE_PATH}")
        # Initialize resources on first load if needed (though cache_resource handles this)
        # if not USE_FAISS:
        #     setup_qdrant_client()
        #     load_sentence_transformer()
        main_app()
    except SystemExit:
         logger.info("Application stopped via st.stop().")
    except Exception as e:
        logger.critical(f"Application crashed critically: {e}", exc_info=True)
        try: st.error(f"A critical error occurred: {e}")
        except: pass # May fail if streamlit context is lost

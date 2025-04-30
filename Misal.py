import streamlit as st
# from PyPDF2 import PdfReader # Original was commented out, keeping it so
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models # <-- Ensure this is imported
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
import time

# --- Assume all setup code (Paths, Logging, etc.) is the same as before ---
# --- Make sure logger = logging.getLogger("rag_pdf_chat") is defined ---
logger = logging.getLogger("rag_pdf_chat")
# Ensure logger has handlers, etc. (from your previous code)

# --- process_pdf_with_fitz ---
# Stores "page_label" instead of "page_num"
def process_pdf_with_fitz(file_bytes, file_hash):
    """Extract text, images, URLs. Generate explanations. Store ABSOLUTE paths."""
    try:
        logger.info("Processing PDF with PyMuPDF (using page_label)")
        pdf_assets_dir = IMAGES_DIR / file_hash
        pdf_assets_dir.mkdir(exist_ok=True, parents=True)

        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        pages_content = []
        page_count_total = pdf_document.page_count
        progress_bar = st.progress(0)

        for page_index in range(page_count_total): # Using 0-based index for iteration
            page = pdf_document[page_index]
            text = page.get_text()
            links = page.get_links()
            urls = [link["uri"] for link in links if "uri" in link]
            image_list = page.get_images(full=True)
            images_data = []

            # --- GET VISUAL PAGE LABEL ---
            visual_page_label = page.get_label() # Get the label string (e.g., "iv", "9", "A-1")
            logger.debug(f"Processing Page Index {page_index}: Visual Label='{visual_page_label}'")
            # -----------------------------

            # Update progress bar using label for clarity
            progress_bar.progress((page_index + 1) / page_count_total, text=f"Processing Page '{visual_page_label}' ({page_index + 1}/{page_count_total})...")

            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = pdf_document.extract_image(xref)
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        # Use page index + 1 in filename for uniqueness, label might not be unique/filename-safe
                        image_filename = f'image_p{page_index + 1}_{img_index + 1}.{image_ext}'
                        absolute_image_path = pdf_assets_dir / image_filename

                        with open(absolute_image_path, 'wb') as img_file:
                            img_file.write(image_bytes)

                        # Assuming image_response takes path now based on prior discussion
                        # If it takes bytes, change str(absolute_image_path) back to image_bytes
                        explanation = image_response(str(absolute_image_path), f"Explain content of {image_filename} on page '{visual_page_label}'")

                        images_data.append({
                            "path": str(absolute_image_path),
                            "filename": image_filename,
                            "ext": image_ext,
                            "explanation": explanation
                        })
                except Exception as img_proc_err:
                    logger.error(f"Error processing image xref {xref} on page index {page_index} (label '{visual_page_label}'): {img_proc_err}")

            # --- STORE PAGE LABEL IN METADATA ---
            page_data = {
                "page_label": visual_page_label, # Store the string label
                "text": text,
                "urls": urls,
                "images": images_data
            }
            # --------------------------------------
            pages_content.append(page_data)

        progress_bar.empty()
        logger.info(f"Extracted content from {len(pages_content)} pages")
        return pages_content

    except Exception as e:
        logger.error(f"Error processing PDF with PyMuPDF: {e}", exc_info=True)
        st.error(f"Error during PDF processing: {e}")
        if 'progress_bar' in locals(): progress_bar.empty()
        raise e

# --- create_enhanced_chunks ---
# Uses "page_label", handles image-only pages
def create_enhanced_chunks(pages_content, chunk_size=1000, chunk_overlap=200):
    """
    Create text chunks with metadata (page_label, images[path,expl], urls).
    Handles pages with only images by creating a placeholder chunk.
    Uses 'page_label' consistently for metadata key.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=False) # Ensure add_start_index is False if not needed
    enhanced_chunks = []
    current_chunk_id = 0

    for page_data in pages_content:
        page_text = page_data.get("text", "")
        page_images = page_data.get("images", [])
        page_label = page_data.get("page_label", "Unknown") # <-- Read 'page_label'
        page_urls = page_data.get("urls", [])

        text_chunks = text_splitter.split_text(page_text)

        if not text_chunks and page_images: # Handle image-only
            logger.info(f"Page '{page_label}' has no text but contains {len(page_images)} image(s). Creating placeholder chunk.")
            first_image_explanation = page_images[0].get("explanation", "[Image Explanation Unavailable]")
            placeholder_text = (
                f"Content for page '{page_label}' consists primarily of image(s). "
                f"Description of the first image: {first_image_explanation}"
            )
            enhanced_chunk = {
                "chunk_id": current_chunk_id,
                "text": placeholder_text,
                "page_label": page_label, # <-- Store 'page_label'
                "urls": page_urls,
                "images": page_images
            }
            enhanced_chunks.append(enhanced_chunk)
            current_chunk_id += 1
        else: # Handle pages with text
            for chunk_text in text_chunks:
                if not chunk_text or chunk_text.isspace(): continue
                enhanced_chunk = {
                    "chunk_id": current_chunk_id,
                    "text": chunk_text,
                    "page_label": page_label, # <-- Store 'page_label'
                    "urls": page_urls,
                    "images": page_images
                }
                enhanced_chunks.append(enhanced_chunk)
                current_chunk_id += 1

    logger.info(f"Created {len(enhanced_chunks)} enhanced chunks (including placeholders if any)")
    return enhanced_chunks

# --- process_pdf_qdrant ---
# Uses "page_label" in the payload
def process_pdf_qdrant(file_bytes, collection_name):
    """Process PDF and add to Qdrant with enhanced metadata using page_label."""
    try:
        logger.info(f"Processing PDF for Qdrant collection: {collection_name} (using page_label)")
        file_hash = hashlib.md5(file_bytes).hexdigest()

        with st.spinner("Extracting text, images, labels, and generating explanations..."):
            pages_content = process_pdf_with_fitz(file_bytes, file_hash) # Uses updated fitz function
        with st.spinner("Creating text chunks..."):
            enhanced_chunks = create_enhanced_chunks(pages_content) # Uses updated chunk function

        # Check if any chunks were created before proceeding
        if not enhanced_chunks:
             logger.warning(f"No chunks were created for PDF with hash {file_hash}. Possibly an image-only or empty PDF. Skipping embedding and upload.")
             st.warning("⚠️ No text content found to index in this PDF.")
             # We should still return something or set state to indicate processing finished, but empty
             return # Or handle state appropriately to show 'ready but empty'

        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()

        # --- Collection Exists Check (remains the same) ---
        try:
            collection_info = qdrant_client.get_collection(collection_name=collection_name)
            if collection_info.points_count > 0:
                logger.info(f"Collection '{collection_name}' exists with {collection_info.points_count} points. Assuming processed. Skipping upload.")
                st.info(f"Using existing data for '{collection_name}'.")
                # Make sure session state is set correctly even when skipping upload
                st.session_state.collection_name = collection_name # Ensure this is set
                return # Skip embedding and upload
        except Exception:
            logger.info(f"Collection '{collection_name}' not found or empty. Proceeding to create/upload.")
        # ---------------------------------------------

        with st.spinner("Creating embeddings..."):
            text_contents = [chunk["text"] for chunk in enhanced_chunks]
            embeddings = model.encode(text_contents, show_progress_bar=True)

        with st.spinner("Uploading data to vector database..."):
            points_to_upload = []
            for idx, chunk in enumerate(enhanced_chunks):
                # --- Use page_label in payload ---
                payload = {
                    "text": chunk["text"],
                    "chunk_id": chunk["chunk_id"],
                    "page_label": chunk["page_label"], # <-- Use 'page_label' key
                    "urls": chunk.get("urls", []),
                    "images": chunk.get("images", [])
                }
                # --------------------------------
                points_to_upload.append(
                    models.PointStruct(id=chunk["chunk_id"], vector=embeddings[idx].tolist(), payload=payload)
                )

            # Before upserting, ensure collection exists (might be created in the 'except' block logic in main_app processing)
            # It's better practice to ensure creation within this function if not found above
            try:
                 qdrant_client.get_collection(collection_name=collection_name)
            except Exception:
                 logger.info(f"Collection '{collection_name}' still not found. Creating it now before upsert.")
                 vector_size = model.get_sentence_embedding_dimension()
                 qdrant_client.create_collection(
                     collection_name=collection_name,
                     vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                 )

            qdrant_client.upsert(collection_name=collection_name, points=points_to_upload, wait=True)

        logger.info(f"Successfully added/updated {len(points_to_upload)} chunks in {collection_name}")
        st.success(f"✅ Added/Updated {len(points_to_upload)} chunks.")
        st.session_state.collection_name = collection_name # Ensure state is set after successful upload

    except Exception as e:
        logger.error(f"Error processing PDF for Qdrant: {e}", exc_info=True)
        st.error(f"Error processing PDF: {e}")
        # Reset relevant session state on error? Handled in main_app's try/except/finally
        raise e


# --- process_pdf_faiss ---
# Corrected to use "page_label", just in case USE_FAISS is ever True
def process_pdf_faiss(file_bytes, file_hash):
    """Process PDF for FAISS with enhanced metadata using page_label."""
    try:
        logger.info("Processing PDF with PyMuPDF and creating FAISS index (using page_label)")
        with st.spinner("Extracting text, images, labels, and generating explanations..."):
             pages_content = process_pdf_with_fitz(file_bytes, file_hash) # Uses updated fitz function
        with st.spinner("Creating text chunks..."):
             enhanced_chunks = create_enhanced_chunks(pages_content) # Uses updated chunk function

        # Check if any chunks were created before proceeding
        if not enhanced_chunks:
             logger.warning(f"No chunks were created for PDF with hash {file_hash} for FAISS. Skipping embedding.")
             st.warning("⚠️ No text content found to index in this PDF.")
             return None # Return None or raise an error? Let's return None

        with st.spinner("Creating embeddings..."):
             embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
             texts = [chunk["text"] for chunk in enhanced_chunks]
             # --- Ensure metadata uses 'page_label' ---
             metadatas = [{"chunk_id": c["chunk_id"], "page_label": c["page_label"], "urls": c.get("urls", []), "images": c.get("images", [])} for c in enhanced_chunks]
             # -----------------------------------------
        with st.spinner("Creating FAISS index..."):
             vector_store = FAISS.from_texts(texts, embeddings_model, metadatas=metadatas)
        logger.info("PDF processed successfully with FAISS and enhanced metadata")
        return vector_store
    except Exception as e:
        logger.error(f"Error processing PDF with FAISS: {e}", exc_info=True)
        st.error(f"Error processing PDF with FAISS: {e}")
        raise e


# --- search_chunks ---
# Filters by "page_label" string (should be correct from previous step)
def search_chunks(collection_name, query, limit=5):
    """Search for chunks similar to the query, with optional page label filtering."""
    try:
        logger.info(f"Searching collection {collection_name} for: {query} (using page_label filter)")
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        query_embedding = model.encode([query])[0]

        page_filter = None
        match = re.search(r'(?:page|label)\s+([\w-]+)', query, re.IGNORECASE)
        if match:
            page_label_query = match.group(1)
            logger.info(f"Detected request for page label '{page_label_query}'. Applying filter.")
            page_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="page_label", # Filter key
                        match=models.MatchValue(value=page_label_query) # String match
                    )
                ]
            )

        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=page_filter,  # Pass the filter
            limit=limit,
            with_payload=True
        )
        logger.info(f"Found {len(search_results)} relevant chunks (Page label filter applied: {page_filter is not None})")
        if not search_results and page_filter:
             logger.info(f"No relevant chunks found specifically with label '{page_label_query}' matching the query semantically.")

        return search_results
    except Exception as e:
        logger.error(f"Error searching chunks in {collection_name}: {e}", exc_info=True)
        st.error(f"Error searching chunks: {e}")
        return []

# --- image_response ---
# Ensure this function matches how you call it in process_pdf_with_fitz
# If you call it with the path:
def image_response(image_path_str, prompt="Describe this image."):
    """Placeholder: Takes image path string, simulates explanation."""
    logger.info(f"Simulating image explanation for path: {image_path_str} | Prompt: {prompt}")
    time.sleep(0.1)
    try:
        image_path = Path(image_path_str)
        if image_path.is_file():
            image_bytes = image_path.read_bytes() # Read bytes if needed for hash/analysis
            image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
            return f"Placeholder explanation for image at '{image_path.name}' (hash: {image_hash}). Relevant to '{prompt[:30]}...'."
        else:
             logger.error(f"Image file not found at path during simulation: {image_path_str}")
             return f"[Placeholder error: Image not found at {image_path.name}]"
    except Exception as e:
        logger.error(f"Error processing image file {image_path_str} in placeholder: {e}")
        return f"[Placeholder error processing image at {Path(image_path_str).name}]"

# If you changed the call back to use bytes:
# def image_response(image_bytes, prompt="Describe this image."):
#     """Placeholder: Takes image bytes, simulates explanation."""
#     logger.info(f"Simulating image explanation from bytes for prompt: {prompt}")
#     time.sleep(0.1)
#     image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
#     return f"Placeholder explanation for image (hash: {image_hash}). Appears relevant to '{prompt[:30]}...'."

# --- Other Functions ---
# init_feedback_csv, log_feedback, validate_email, check_login, show_login
# display_image, abc_response, load_sentence_transformer, setup_qdrant_client
# main_app (ensure display/logging uses page_label if desired)
# Assume these are mostly unchanged, BUT double-check main_app's source expander
# and feedback logging logic for "page_num" vs "page_label".

# --- Example modification for main_app source display/log ---
# Inside main_app:
#  Sources Expander:
#   st.markdown(f"... Page Label: {source.get('page_label', 'Unknown')})**") # Changed Page: to Page Label: and key to page_label
#  Context/Prompt Prep:
#   item = f"Source {i+1} (Lbl:{src.get('page_label','N/A')}, Chunk:{src.get('chunk_id','N/A')}):\n" # Changed Page: to Lbl: and key to page_label
#   sources_info_log_list.append(f"Chunk:{src.get('chunk_id','N/A')}|Lbl:{src.get('page_label','N/A')}") # Changed Pg: to Lbl: and key to page_label

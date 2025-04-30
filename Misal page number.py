# --- Add import for Qdrant models if not already at the top ---
from qdrant_client.http import models # Ensure this is imported

# --- Modified Search Chunks Function ---
def search_chunks(collection_name, query, limit=5):
    """Search for chunks similar to the query, with optional page filtering."""
    try:
        logger.info(f"Searching collection {collection_name} for: {query}")
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        query_embedding = model.encode([query])[0]

        # --- START: Add Page Number Filtering Logic ---
        page_filter = None
        # Use regex to find patterns like "page 4", "on page 10", etc.
        match = re.search(r'page\s+(\d+)', query, re.IGNORECASE)
        if match:
            try:
                page_number = int(match.group(1))
                logger.info(f"Detected request for page {page_number}. Applying filter.")
                # Create a Qdrant filter condition
                page_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="page_num", # The metadata field storing the page number
                            match=models.MatchValue(value=page_number)
                        )
                    ]
                )
            except ValueError:
                logger.warning(f"Could not parse page number from query: {query}")
        # --- END: Add Page Number Filtering Logic ---

        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=page_filter,  # <--- PASS THE FILTER HERE (will be None if no page number detected)
            limit=limit,
            with_payload=True # Ensure payload is retrieved
        )
        logger.info(f"Found {len(search_results)} relevant chunks (Page filter applied: {page_filter is not None})")
        # If results are empty specifically due to filtering, maybe add a specific log/message?
        if not search_results and page_filter:
             logger.info(f"No relevant chunks found specifically on page {page_number} matching the query semantically.")

        return search_results
    except Exception as e:
        logger.error(f"Error searching chunks in {collection_name}: {e}", exc_info=True)
        st.error(f"Error searching chunks: {e}")
        return []



import logging # Ensure logger is available

# Assume logger is configured elsewhere as logger = logging.getLogger(...)
# Assume RecursiveCharacterTextSplitter is imported

logger = logging.getLogger("rag_pdf_chat") # Using the logger name from your original code

# --- Modified Create Enhanced Chunks Function ---

def create_enhanced_chunks(pages_content, chunk_size=1000, chunk_overlap=200):
    """
    Create text chunks with metadata (page_label, images[path,expl], urls).
    Handles pages with only images by creating a placeholder chunk using the
    first image's explanation as its text content.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    enhanced_chunks = []
    current_chunk_id = 0 # Initialize chunk ID counter

    for page_data in pages_content:
        page_text = page_data.get("text", "") # Get text, default to empty string
        page_images = page_data.get("images", []) # Get images list
        page_label = page_data.get("page_label", "Unknown") # Get page label
        page_urls = page_data.get("urls", []) # Get URLs

        # Attempt to split the text from the page
        text_chunks = text_splitter.split_text(page_text)

        # --- START MINIMAL CHANGE LOGIC ---
        # Check if text splitting yielded no chunks BUT there are images on the page
        if not text_chunks and page_images:
            logger.info(f"Page '{page_label}' has no text but contains {len(page_images)} image(s). Creating placeholder chunk.")

            # Use the explanation of the first image as representative text content
            # Add context to make the placeholder text more meaningful
            first_image_explanation = page_images[0].get("explanation", "[Image Explanation Unavailable]")
            placeholder_text = (
                f"Content for page '{page_label}' consists primarily of image(s). "
                f"Description of the first image: {first_image_explanation}"
            )

            # Create a single placeholder chunk for this image-only page
            enhanced_chunk = {
                "chunk_id": current_chunk_id,
                "text": placeholder_text, # Use the generated placeholder text
                "page_label": page_label,
                "urls": page_urls, # Include URLs if any were found on the page
                "images": page_images # Attach metadata for ALL images on the page
            }
            enhanced_chunks.append(enhanced_chunk)
            current_chunk_id += 1 # Increment chunk ID

        # --- ELSE: Process normally if text chunks were found ---
        else:
            # Original logic: Create chunks from the actual text found
            for chunk_text in text_chunks:
                enhanced_chunk = {
                    "chunk_id": current_chunk_id,
                    "text": chunk_text, # Use the actual text chunk
                    "page_label": page_label,
                    "urls": page_urls,
                    "images": page_images # Attach metadata for ALL images on the page
                }
                enhanced_chunks.append(enhanced_chunk)
                current_chunk_id += 1 # Increment chunk ID
        # --- END MINIMAL CHANGE LOGIC ---

    logger.info(f"Created {len(enhanced_chunks)} enhanced chunks (including placeholders if any)")
    return enhanced_chunks

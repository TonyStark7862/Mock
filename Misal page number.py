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

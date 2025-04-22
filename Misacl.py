# Assumes all necessary imports and helper functions are defined above this function

def main_app():
    # --- Initial Setup ---
    st.set_page_config(page_title="RAG PDF Chat Pro", layout="wide")
    init_feedback_csv() # Ensure feedback file is ready

    # --- Initialize Session State ---
    # Ensure all necessary keys have default values
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("user_email", "Unknown")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("processing_active", False) # Flag for ongoing processing
    st.session_state.setdefault("initial_message_sent", False) # Flag for the first AI message
    st.session_state.setdefault("retriever", None) # For FAISS
    st.session_state.setdefault("faiss_cache_path", None) # For FAISS
    st.session_state.setdefault("collection_name", None) # For Qdrant

    # --- Login Check ---
    if not check_login(): # Uses st.session_state.get("logged_in", False)
        show_login()
        return # Stop execution if not logged in

    # --- Main App UI ---
    st.title("Chat with PDF (Image/URL Integration + Download)")
    st.sidebar.text(f"Logged in as: {st.session_state.user_email}")

    # --- Sidebar Elements ---
    if st.sidebar.button("Logout"):
        logger.info(f"User logged out: {st.session_state.user_email}")
        # Clear relevant session state keys upon logout
        keys_to_clear = ["retriever", "faiss_cache_path", "collection_name", "messages",
                         "processing_active", "initial_message_sent", "user_email"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.logged_in = False # Explicitly set logged_in to False
        st.rerun() # Rerun to go back to login page

    st.sidebar.markdown("---")
    st.sidebar.header("Support")
    st.sidebar.markdown("support@example.com") # Replace with actual support info

    # --- PDF Upload and Processing Trigger ---
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose PDF", type="pdf", key="pdf_uploader")

        # Define these variables here to have them available in the processing block if needed
        file_bytes = None
        file_hash = None
        target_collection_name = None
        target_faiss_cache = None

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue() # Read bytes early
            file_hash = hashlib.md5(file_bytes).hexdigest()
            process_this_file = False
            target_collection_name = f"pdf_{file_hash}" # Assign here
            target_faiss_cache = VECTORDB_DIR / f"{file_hash}.pkl" # Assign here

            # Determine if the current file needs processing
            if USE_FAISS:
                if st.session_state.retriever is None or st.session_state.faiss_cache_path != str(target_faiss_cache):
                    process_this_file = True
            else: # Qdrant
                if st.session_state.collection_name is None or st.session_state.collection_name != target_collection_name:
                    process_this_file = True

            # --- Trigger Processing ---
            # Only start if needed AND not already processing
            if process_this_file and not st.session_state.processing_active:
                logger.info(f"Triggering processing for file: {uploaded_file.name} (Hash: {file_hash})")
                st.session_state.messages = [] # Clear chat for new file
                st.session_state.processing_active = True # Set flag: processing starts
                st.session_state.initial_message_sent = False # Reset flag for initial AI message
                # Clear previous vector store state explicitly before starting new processing
                st.session_state.retriever = None
                st.session_state.faiss_cache_path = None
                st.session_state.collection_name = None
                st.rerun() # Rerun immediately to reflect the processing state (show spinners, disable chat)

    # --- Processing Block ---
    # This section executes only during the rerun triggered above if processing_active is True
    if st.session_state.processing_active:
        processing_successful = False # Track outcome
        # Ensure file info is available (it should be if processing_active is True)
        if uploaded_file is None or file_bytes is None or file_hash is None:
             logger.error("Processing active but file info missing. Resetting.")
             st.error("An internal error occurred (file info missing during processing). Please re-upload.")
             st.session_state.processing_active = False
             st.rerun()
        else:
            logger.info(f"Now actively processing file: {uploaded_file.name}")
            try:
                if USE_FAISS:
                    # --- FAISS Processing ---
                    # (FAISS logic already skips parsing if cache exists)
                    if target_faiss_cache.exists():
                        with st.spinner(f"Loading cached FAISS data..."):
                            logger.info(f"Loading FAISS from cache: {target_faiss_cache}")
                            with open(target_faiss_cache, "rb") as f: vector_store = pickle.load(f)
                        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                        st.session_state.faiss_cache_path = str(target_faiss_cache)
                        processing_successful = True
                        st.success("✅ Loaded data from cache (FAISS).")
                    else:
                        with st.spinner("Processing PDF for FAISS..."):
                            vector_store = process_pdf_faiss(file_bytes, file_hash)
                        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                        st.session_state.faiss_cache_path = str(target_faiss_cache)
                        logger.info(f"Saving FAISS cache: {target_faiss_cache}")
                        with open(target_faiss_cache, "wb") as f: pickle.dump(vector_store, f)
                        processing_successful = True
                        st.success("✅ PDF processed & cached (FAISS).")

                else:
                    # --- Qdrant Processing (with Optimization) ---
                    qdrant_client = setup_qdrant_client()
                    model = load_sentence_transformer()

                    if not qdrant_client or not model:
                         raise Exception("Qdrant client or embedding model not available for processing.")

                    vector_size = model.get_sentence_embedding_dimension()
                    collection_exists = False
                    collection_has_data = False

                    # 1. Check if collection exists and has data *before* parsing
                    try:
                        with st.spinner("Checking vector database..."):
                            collection_info = qdrant_client.get_collection(collection_name=target_collection_name)
                        logger.info(f"Qdrant collection '{target_collection_name}' already exists.")
                        collection_exists = True
                        if collection_info.points_count > 0:
                            logger.info(f"Collection has {collection_info.points_count} points. Skipping PDF parsing and data upload.")
                            collection_has_data = True
                            # Set the state to use this existing collection
                            st.session_state.collection_name = target_collection_name
                            processing_successful = True
                            st.success(f"✅ Found existing data for this PDF in collection '{target_collection_name}'.") # User feedback

                    except Exception as e:
                        # Handle specific 'Not Found' case if possible, otherwise log general error
                        if "404" in str(e) or "not found" in str(e).lower():
                            logger.info(f"Qdrant collection '{target_collection_name}' not found. Will create.")
                        else:
                            logger.warning(f"Could not get collection info for {target_collection_name} (may be created): {e}")
                        collection_exists = False
                        collection_has_data = False

                    # 2. Process PDF *only if* data wasn't already found
                    if not collection_has_data:
                        # Create collection if it didn't exist
                        if not collection_exists:
                            logger.info(f"Creating Qdrant collection: {target_collection_name}")
                            with st.spinner("Setting up vector database collection..."):
                                qdrant_client.create_collection(
                                    collection_name=target_collection_name,
                                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                                )
                            logger.info(f"Collection '{target_collection_name}' created.")

                        # Now, process the PDF (parse, embed, upload)
                        # process_pdf_qdrant internally handles spinners and success messages for this part
                        logger.info(f"Processing PDF content for collection '{target_collection_name}'...")
                        process_pdf_qdrant(file_bytes, target_collection_name) # This performs parsing & upload

                        # If process_pdf_qdrant completed without error, set state
                        st.session_state.collection_name = target_collection_name
                        processing_successful = True

                # --- Post-processing Actions (if successful) ---
                if processing_successful:
                    logger.info(f"Processing successful for file hash {file_hash}. Ready state set.")
                    if not st.session_state.initial_message_sent:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "I have read and understood your document. How can I help you?",
                            "sources": []
                        })
                        st.session_state.initial_message_sent = True
                        logger.info("Initial 'ready' message added to chat.")

            except Exception as e:
                # --- Handle Processing Failure ---
                logger.error(f"Fatal error during file processing ({uploaded_file.name}): {e}", exc_info=True)
                st.error(f"Error processing PDF: {e}")
                # Explicitly reset state variables to ensure 'is_ready' becomes false
                st.session_state.retriever = None
                st.session_state.faiss_cache_path = None
                st.session_state.collection_name = None
                st.session_state.initial_message_sent = False # Reset flag on error

            finally:
                # --- Mark Processing as Finished ---
                st.session_state.processing_active = False
                logger.info("Processing marked as inactive.")
                st.rerun() # Rerun one final time to update UI

    # --- Define Cached Resource Loaders ---
    @st.cache_resource
    def load_sentence_transformer():
        """Loads/downloads Sentence Transformer model."""
        model_name = 'all-MiniLM-L6-v2'
        try:
            with st.spinner("Loading embedding model (once)..."): # Spinner only shows on first load
                 local_model_full_path = LOCAL_MODEL_PATH
                 if local_model_full_path.exists():
                     model = SentenceTransformer(str(local_model_full_path))
                     logger.info(f"Loaded embedding model from local path: {local_model_full_path}")
                 else:
                     logger.warning(f"Downloading {model_name} to {local_model_full_path}...")
                     model = SentenceTransformer(model_name)
                     model.save(str(local_model_full_path))
                     logger.info(f"Downloaded and saved model to {local_model_full_path}")
            return model
        except Exception as e:
            logger.error(f"Fatal: Could not load/download embedding model: {e}", exc_info=True)
            st.error(f"Fatal Error: Could not load embedding model: {e}")
            st.stop()

    @st.cache_resource
    def setup_qdrant_client():
        """Sets up Qdrant client."""
        # Decide here whether to use path or URL based on deployment needs
        # Option 1: Local Path (causes locking issues with multiple processes/unclean shutdowns)
        use_local_path = True # Set to False if using Qdrant Server
        qdrant_path = VECTORDB_DIR / "qdrant_db"
        qdrant_url = "http://localhost:6333" # Standard Qdrant server URL

        if use_local_path:
            logger.info(f"Setting up Qdrant client using local path: {qdrant_path}")
            try:
                qdrant_path.mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=str(qdrant_path), timeout=60)
                client.get_collections() # Verify connection
                logger.info("Qdrant client (local path) setup verified.")
                return client
            except Exception as e:
                 # Specific check for the locking error
                 if "already accessed" in str(e).lower():
                     logger.error(f"Qdrant lock error on path {qdrant_path}: {e}", exc_info=True)
                     st.error(f"Fatal Error: Vector DB directory is locked by another process.\nDetails: {e}\nSuggestion: Stop other Qdrant instances or use Qdrant server mode.")
                 else:
                     logger.error(f"Fatal: Error setting up Qdrant (local path): {e}", exc_info=True)
                     st.error(f"Fatal Error: Could not connect to Vector DB (local path): {e}")
                 st.stop()
        else:
            # Option 2: Connect to Qdrant Server
            logger.info(f"Setting up Qdrant client to connect to server: {qdrant_url}")
            try:
                # Connect to the server using URL
                client = QdrantClient(url=qdrant_url, timeout=60)
                client.get_collections() # Verify connection
                logger.info("Qdrant client (server) connection verified.")
                return client
            except Exception as e:
                logger.error(f"Fatal: Error connecting to Qdrant server at {qdrant_url}: {e}", exc_info=True)
                st.error(f"Fatal Error: Could not connect to Vector DB Server: {e}")
                st.info(f"Ensure the Qdrant server is running and accessible at {qdrant_url}.")
                st.stop()

    # --- Determine Readiness for Chatting ---
    is_ready = False
    if not st.session_state.processing_active:
        if USE_FAISS:
            is_ready = st.session_state.retriever is not None
        else: # Qdrant
            is_ready = st.session_state.collection_name is not None

    # --- Display Status Info / Warnings ---
    if not is_ready and uploaded_file is not None and not st.session_state.processing_active:
        st.warning("PDF processing may have failed or is initializing. Please check logs or re-upload if needed.")
    elif not is_ready and uploaded_file is None and not st.session_state.processing_active:
        st.info("Please upload a PDF file using the sidebar to begin chatting.")

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # (Keep the user/assistant message rendering logic exactly as before,
            # including image display, download link, and sources expander)
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "assistant":
                response_content = message["content"]
                # --- Assistant Message Rendering ---
                logger.debug(f"Rendering assistant message: {response_content[:100]}...")
                last_end = 0
                try:
                    image_pattern = r'\[Image:\s*(.*?)\s*\]'
                    for match in re.finditer(image_pattern, response_content):
                        start, end = match.span()
                        image_path_str = match.group(1).strip()
                        image_path = Path(image_path_str)
                        if start > last_end:
                            st.markdown(response_content[last_end:start], unsafe_allow_html=True)
                        logger.debug(f"Rendering image marker: {image_path}")
                        if image_path.is_file():
                            st.image(str(image_path), width=400) # Inline image preview
                            # Download link generation
                            try:
                                filename = image_path.name
                                ext = image_path.suffix.lstrip('.').lower()
                                known_image_types = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'svg']
                                mime_type = f"image/{ext}" if ext in known_image_types else 'application/octet-stream'
                                if ext == 'svg': mime_type = 'image/svg+xml'
                                with open(image_path, "rb") as f: img_bytes = f.read()
                                b64_encoded = base64.b64encode(img_bytes).decode()
                                data_url = f"data:{mime_type};base64,{b64_encoded}"
                                link_text = "Download Image"
                                link_html = f'<a href="{data_url}" download="{filename}" style="display: inline-block; margin-top: 5px; text-decoration: underline; color: #1E88E5; font-size: 0.9em;">{link_text}</a>'
                                st.markdown(link_html, unsafe_allow_html=True)
                            except Exception as dl_err:
                                logger.error(f"Error creating download link for {image_path}: {dl_err}", exc_info=True)
                                st.caption("_(Could not create download link)_")
                        elif image_path_str:
                            st.error(f"Image file not found: {image_path_str}")
                            logger.error(f"Render Error: Image file not found at {image_path_str}")
                        else:
                            st.error("Invalid image marker format.")
                            logger.error("Render Error: Invalid image marker format.")
                        last_end = end
                    if last_end < len(response_content):
                        st.markdown(response_content[last_end:], unsafe_allow_html=True)
                except Exception as render_err:
                    logger.error(f"Critical error parsing/rendering assistant response: {render_err}", exc_info=True)
                    st.markdown(response_content, unsafe_allow_html=True) # Fallback display
                    st.warning("Could not fully parse response for inline images/links.")
                # --- End Assistant Message Rendering ---

                # --- Sources Expander ---
                if "sources" in message and message["sources"]:
                     with st.expander("View Raw Sources Used"):
                         for i, source in enumerate(message["sources"]):
                             st.markdown(f"**Source Context {i+1} (Chunk ID: {source.get('chunk_id', 'N/A')}, Page: {source.get('page_num', 'Unknown')})**")
                             st.markdown(f"> {source.get('text', '[No text found]')}")
                             urls = source.get("urls", [])
                             if urls: st.markdown(f"**URLs:** {', '.join(urls)}")
                             images_in_source = source.get("images", [])
                             if images_in_source:
                                 st.markdown("**Associated Images (from source chunk):**")
                                 cols = st.columns(min(len(images_in_source), 3))
                                 for j, img_data in enumerate(images_in_source):
                                     with cols[j % 3]:
                                         st.markdown(f"_{img_data.get('filename', 'N/A')}_")
                                         img_path_exp = img_data.get("path")
                                         if img_path_exp and Path(img_path_exp).is_file(): # Check if path exists before displaying
                                            st.markdown(display_image(img_path_exp), unsafe_allow_html=True) # Uses helper
                                         elif img_path_exp:
                                            st.caption(f"_[Image not found: {Path(img_path_exp).name}]_")
                                         st.caption(f"Expl: {img_data.get('explanation', '')[:100]}...") # Show explanation regardless
                             st.divider()


    # --- Chat Input Area ---
    chat_disabled = st.session_state.processing_active or not is_ready
    if st.session_state.processing_active:
        prompt_placeholder = "Processing PDF, please wait..."
    elif not is_ready:
        prompt_placeholder = "Please upload a PDF to enable chat"
    else:
        prompt_placeholder = "Ask a question about your PDF"

    if prompt := st.chat_input(prompt_placeholder, disabled=chat_disabled):
        # --- Handle User Query ---
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt) # Display user message

        # --- Retrieval ---
        current_file_name = uploaded_file.name if uploaded_file else "No file uploaded"
        retrieved_sources = []
        with st.spinner("Searching relevant parts in PDF..."):
            try:
                if USE_FAISS:
                     # (Keep FAISS retrieval logic)
                     if st.session_state.retriever:
                         docs = st.session_state.retriever.get_relevant_documents(prompt)
                         for doc in docs:
                             meta = doc.metadata if hasattr(doc, 'metadata') else {}
                             retrieved_sources.append({"text": doc.page_content, **meta})
                         logger.info(f"Retrieved {len(retrieved_sources)} chunks (FAISS)")
                     else: logger.error("FAISS retriever not available for search.")
                else: # Qdrant
                     # (Keep Qdrant retrieval logic)
                     if st.session_state.collection_name:
                         qdrant_client = setup_qdrant_client()
                         if not qdrant_client: raise Exception("Qdrant client not available for search")
                         search_results = search_chunks(st.session_state.collection_name, prompt)
                         for result in search_results:
                             payload = result.payload if hasattr(result, 'payload') else {}
                             payload_with_id = {**payload, "chunk_id": payload.get("chunk_id", result.id)}
                             retrieved_sources.append(payload_with_id)
                         logger.info(f"Retrieved {len(retrieved_sources)} chunks (Qdrant)")
                     else: logger.error("Qdrant collection name not available for search.")
            except Exception as search_err:
                 logger.error(f"Error during retrieval: {search_err}", exc_info=True)
                 st.error(f"Error searching document: {search_err}")
                 retrieved_sources = []

        # --- Context/Prompt Prep & LLM Call ---
        response = ""
        sources_info_log = "N/A"
        if not retrieved_sources:
            response = "I couldn't find specific information related to your query in the document based on my search."
            sources_info_log = "No sources found"
        else:
            # (Keep Context/Prompt Prep logic)
            context_items = []
            sources_info_log_list = []
            for i, src in enumerate(retrieved_sources):
                item = f"Source {i+1} (Page:{src.get('page_num','N/A')}, Chunk:{src.get('chunk_id','N/A')}):\n"
                item += f"Text: \"{src.get('text', '')}\"\n"
                if src.get('urls'): item += f"URLs: {', '.join(src['urls'])}\n"
                if src.get('images'):
                    item += "Images:\n"
                    for img in src['images']:
                        item += f" - Path: {img.get('path','N/A')} | Explanation: {img.get('explanation','N/A')}\n"
                item += "---\n"
                context_items.append(item)
                sources_info_log_list.append(f"Chunk:{src.get('chunk_id','N/A')}|Pg:{src.get('page_num','N/A')}")
            context_for_llm = "\n".join(context_items)
            sources_info_log = "; ".join(sources_info_log_list)

            # (Keep Prompt Definition)
            full_prompt = f"""You are an AI assistant analyzing a PDF based ONLY on the provided context chunks.
Answer the user's question accurately using information solely from the context.
Integrate relevant details directly into your response:
- **URLs:** If a URL from the context supports your point, include it in Markdown: [descriptive text](URL).
- **Images:** If an image (identified by its absolute Path) from the context is directly relevant, include its path using this exact format: [Image: /absolute/path/to/image.ext]. You can refer to its Explanation if useful. Only include images mentioned in the provided context sources.

Do not make up information or refer to external knowledge. If the context doesn't contain the answer, state that clearly.

--- Start of Context ---
{context_for_llm}
--- End of Context ---

Question: {prompt}

Answer:"""

            # (Keep LLM Call)
            with st.spinner("Generating response..."):
                try:
                    response = abc_response(full_prompt) # Using placeholder
                    logger.info("Received response from placeholder LLM.")
                except Exception as e:
                    logger.error(f"LLM placeholder error: {e}", exc_info=True)
                    response = f"Sorry, an error occurred while generating the response: {str(e)}"

        # --- Add Assistant Message to History ---
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": retrieved_sources
        })

        # --- Log Feedback ---
        logger.info(f"Logging feedback for prompt: {prompt}")
        response_for_log = response.replace('\n', '\\n').replace('\r', '')
        log_feedback(st.session_state.user_email, prompt, current_file_name, response_for_log, sources_info_log)

        # --- Trigger Re-run to Display New Assistant Message ---
        st.rerun()

# --- Entry point ---
# (Keep the if __name__ == "__main__": block exactly as before)
# if __name__ == "__main__":
#     # ... (try/except block calling main_app) ...

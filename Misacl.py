# --- Processing Block ---
            # This section executes only during the rerun triggered above if processing_active is True
            if st.session_state.processing_active:
                processing_successful = False # Track outcome
                logger.info(f"Now actively processing file: {uploaded_file.name}")
                try:
                    if USE_FAISS:
                        # --- FAISS Processing ---
                        # (Keep the existing FAISS logic - it already checks for the cache file first)
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
                        # --- Qdrant Processing ---
                        qdrant_client = setup_qdrant_client() # Ensure client is ready
                        model = load_sentence_transformer() # Ensure model is ready (needed for vector size)

                        if not qdrant_client or not model:
                             raise Exception("Qdrant client or embedding model not available for processing.")

                        vector_size = model.get_sentence_embedding_dimension()
                        collection_exists = False
                        collection_has_data = False

                        # 1. Check if collection exists and has data
                        try:
                            with st.spinner("Checking vector database..."):
                                collection_info = qdrant_client.get_collection(collection_name=target_collection_name)
                            logger.info(f"Qdrant collection '{target_collection_name}' already exists.")
                            collection_exists = True
                            if collection_info.points_count > 0:
                                logger.info(f"Collection has {collection_info.points_count} points. Skipping processing and upload.")
                                collection_has_data = True
                                # Set the state to use this existing collection
                                st.session_state.collection_name = target_collection_name
                                processing_successful = True
                                st.success(f"✅ Found existing data for this PDF in collection '{target_collection_name}'.")

                        except Exception as e:
                            # Collection does not exist (or other error fetching it)
                            # Handle specific 'Not Found' case if possible, otherwise log general error
                            if "404" in str(e) or "not found" in str(e).lower():
                                logger.info(f"Qdrant collection '{target_collection_name}' not found. Will create.")
                            else:
                                logger.warning(f"Could not get collection info for {target_collection_name}: {e}")
                            collection_exists = False
                            collection_has_data = False # Cannot have data if collection doesn't exist

                        # 2. Process only if needed
                        if not collection_has_data:
                            # Create collection if it doesn't exist
                            if not collection_exists:
                                logger.info(f"Creating Qdrant collection: {target_collection_name}")
                                with st.spinner("Setting up vector database collection..."):
                                    qdrant_client.create_collection(
                                        collection_name=target_collection_name,
                                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                                    )
                                logger.info(f"Collection '{target_collection_name}' created.")

                            # Now, process the PDF (parse, embed, upload)
                            # This function *also* has an internal check to skip upload if points somehow exist now, but the main check is above.
                            logger.info(f"Processing PDF content for collection '{target_collection_name}'...")
                            process_pdf_qdrant(file_bytes, target_collection_name)

                            # If process_pdf_qdrant completed without error
                            st.session_state.collection_name = target_collection_name
                            processing_successful = True
                            # Success message is inside process_pdf_qdrant

                    # --- Post-processing Actions (if successful) ---
                    if processing_successful:
                        # (Keep the logic for the initial AI message exactly as before)
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
                    # (Keep the existing error handling logic)
                    logger.error(f"Fatal error during file processing ({uploaded_file.name}): {e}", exc_info=True)
                    st.error(f"Error processing PDF: {e}")
                    st.session_state.retriever = None
                    st.session_state.faiss_cache_path = None
                    st.session_state.collection_name = None
                    st.session_state.initial_message_sent = False

                finally:
                    # --- Mark Processing as Finished ---
                    # (Keep the existing finally block logic)
                    st.session_state.processing_active = False
                    logger.info("Processing marked as inactive.")
                    st.rerun()

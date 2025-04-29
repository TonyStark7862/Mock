# Inside the create_enhanced_nodes function:

# ... inside the loop iterating through chunk_text ...
for chunk_text in text_chunks:
    # Generate a unique UUID for the node ID required by Qdrant
    node_id_uuid = str(uuid.uuid4())
    # Keep the original semantic ID format for reference in metadata
    original_semantic_id = f"{file_hash}_p{page_num}_c{node_counter}"

    metadata = {
        "file_name": file_name,
        "file_hash": file_hash,
        "page_num": page_num,
        "semantic_id": original_semantic_id, # Store original ID here
        "urls": page_urls,
        "images": page_images
    }
    node = TextNode(
        id_=node_id_uuid, # <--- Use the generated UUID here
        text=chunk_text,
        metadata=metadata,
        # Ensure new metadata keys are excluded if necessary
        excluded_embed_metadata_keys=["file_name", "file_hash", "page_num", "urls", "images", "semantic_id"], # Exclude semantic_id too
        excluded_llm_metadata_keys=["file_hash", "semantic_id"] # LLM likely doesn't need these IDs
    )
    nodes.append(node)
    node_counter += 1
# ... rest of the function

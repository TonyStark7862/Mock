import streamlit as st
from PyPDF2 import PdfReader
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
import re
import csv
import logging
import datetime
from logging.handlers import RotatingFileHandler
import fitz  # PyMuPDF
import base64
from PIL import Image
import tempfile

# Configure logging
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "app.log"

# Configure logger
logger = logging.getLogger("rag_pdf_chat")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE)  # Changed to normal FileHandler
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Add console handler for development
console_handler = logging.StreamHandler()
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)

# Feedback file path
FEEDBACK_DIR = Path("./feedback")
FEEDBACK_DIR.mkdir(exist_ok=True, parents=True)
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.csv"

# Images and assets directory
IMAGES_DIR = Path("./images")
IMAGES_DIR.mkdir(exist_ok=True, parents=True)

# Flag to determine which vector database to use
USE_FAISS = False  # Set to False to use Qdrant instead

# Create directories
VECTORDB_DIR = Path("./vectordb")
VECTORDB_DIR.mkdir(exist_ok=True, parents=True)
LOCAL_MODEL_PATH = "./models/all-MiniLM-L6-v2"
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

# Initialize feedback CSV if it doesn't exist
def init_feedback_csv():
    if not FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'email', 'question', 'file_name', 'response'])
        logger.info(f"Created new feedback file at {FEEDBACK_FILE}")

# Log feedback to CSV
def log_feedback(email, question, file_name, response):
    try:
        with open(FEEDBACK_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, email, question, file_name, response])
        logger.info(f"Logged feedback from {email}")
    except Exception as e:
        logger.error(f"Error logging feedback: {e}")

# Validate email format
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@xyz\.com$'
    return re.match(pattern, email) is not None

# Check login state
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in

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

# Process PDF with PyMuPDF to extract text, images, and URLs
def process_pdf_with_fitz(file_bytes, file_hash):
    """Extract text, images, URLs, and other metadata from PDF using PyMuPDF (fitz)."""
    try:
        logger.info("Processing PDF with PyMuPDF to extract text, images, and URLs")
        
        # Create a directory for this PDF's assets
        pdf_assets_dir = IMAGES_DIR / file_hash
        pdf_assets_dir.mkdir(exist_ok=True)
        
        # Open the PDF document
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        
        # Initialize containers for extracted content
        pages_content = []
        
        # Process each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Extract URLs from the page
            links = page.get_links()
            urls = []
            for link in links:
                if "uri" in link:
                    urls.append(link["uri"])
            
            # Extract images from the page
            image_list = page.get_images()
            images_data = []
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = f'image_p{page_num + 1}_{img_index + 1}.{image_ext}'
                    image_path = pdf_assets_dir / image_filename
                    
                    # Save the image
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_bytes)
                    
                    # Add image info to our data
                    images_data.append({
                        "path": str(image_path),
                        "filename": image_filename,
                        "ext": image_ext
                    })
            
            # Store all data for this page
            page_data = {
                "page_num": page_num + 1,
                "text": text,
                "urls": urls,
                "images": images_data
            }
            
            pages_content.append(page_data)
        
        logger.info(f"Extracted content from {len(pages_content)} pages")
        return pages_content
        
    except Exception as e:
        logger.error(f"Error processing PDF with PyMuPDF: {e}")
        raise e

# Create enhanced chunks with metadata
def create_enhanced_chunks(pages_content, chunk_size=1000, chunk_overlap=200):
    """Create text chunks with metadata about images and URLs."""
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    enhanced_chunks = []
    
    for page_data in pages_content:
        # Split text into chunks
        page_text = page_data["text"]
        text_chunks = text_splitter.split_text(page_text)
        
        # For each chunk from this page, add metadata
        for chunk in text_chunks:
            enhanced_chunk = {
                "text": chunk,
                "page_num": page_data["page_num"],
                "urls": page_data["urls"],
                "images": page_data["images"]
            }
            enhanced_chunks.append(enhanced_chunk)
    
    logger.info(f"Created {len(enhanced_chunks)} enhanced chunks with metadata")
    return enhanced_chunks

# Process PDF and add to Qdrant with enhanced metadata
def process_pdf_qdrant(file_bytes, collection_name):
    try:
        logger.info(f"Processing PDF for Qdrant collection: {collection_name}")
        
        # Calculate file hash for image storage
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        # Extract text, images, and URLs using PyMuPDF
        pages_content = process_pdf_with_fitz(file_bytes, file_hash)
        
        # Create enhanced chunks with metadata
        enhanced_chunks = create_enhanced_chunks(pages_content)
        
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Check if collection exists and has points
        collection_info = qdrant_client.get_collection(collection_name)
        existing_count = collection_info.points_count
        
        # Skip if chunks are already added
        if existing_count > 0:
            logger.info(f"Document chunks already present in collection (found {existing_count} points)")
            st.info(f"Document chunks already added to collection (found {existing_count} points)")
            return
        
        # Create embeddings for chunk text
        logger.info("Creating embeddings for chunks")
        text_chunks = [chunk["text"] for chunk in enhanced_chunks]
        embeddings = model.encode(text_chunks)
        
        # Prepare points for upload
        logger.info("Preparing points for upload to Qdrant with enhanced metadata")
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "chunk_id": idx,
                    "page_num": chunk["page_num"],
                    "urls": chunk["urls"],
                    "images": chunk["images"]
                }
            )
            for idx, (embedding, chunk) in enumerate(zip(embeddings, enhanced_chunks))
        ]
        
        # Upload to collection
        logger.info(f"Uploading {len(points)} points to Qdrant collection")
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        logger.info(f"Successfully added {len(points)} enhanced chunks to collection {collection_name}")
        st.success(f"✅ Added {len(points)} enhanced chunks to collection")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"Error processing PDF: {e}")
        raise e

# Process PDF and add to FAISS with enhanced metadata
def process_pdf_faiss(file_bytes, file_hash):
    """Process PDF for FAISS with enhanced metadata."""
    try:
        logger.info("Processing PDF with PyMuPDF and creating FAISS index")
        
        # Extract text, images, and URLs using PyMuPDF
        pages_content = process_pdf_with_fitz(file_bytes, file_hash)
        
        # Create enhanced chunks with metadata
        enhanced_chunks = create_enhanced_chunks(pages_content)
        
        # Create embeddings
        logger.info("Creating embeddings with HuggingFaceEmbeddings")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Extract just the text for FAISS
        texts = [chunk["text"] for chunk in enhanced_chunks]
        
        # Create metadatas list
        metadatas = [
            {
                "chunk_id": idx,
                "page_num": chunk["page_num"],
                "urls": chunk["urls"],
                "images": chunk["images"]
            }
            for idx, chunk in enumerate(enhanced_chunks)
        ]
        
        # Create FAISS vector store with metadata
        logger.info("Creating FAISS vector store with enhanced metadata")
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        logger.info("PDF processed successfully with FAISS and enhanced metadata")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error processing PDF with FAISS: {e}")
        raise e

# Function to display image in Streamlit
def display_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
            encoded_image = base64.b64encode(image_data).decode()
            return f"<img src='data:image/png;base64,{encoded_image}' style='max-width:100%;'>"
    except Exception as e:
        logger.error(f"Error displaying image: {e}")
        return f"[Image could not be displayed: {image_path}]"

# Search for relevant chunks in Qdrant
def search_chunks(collection_name, query, limit=4):
    """Search for chunks similar to the query."""
    try:
        logger.info(f"Searching for chunks in collection {collection_name} for query: {query}")
        # Get model and client
        model = load_sentence_transformer()
        qdrant_client = setup_qdrant_client()
        
        # Generate embedding for query
        query_embedding = model.encode([query])[0]
        
        # Search in collection
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )
        
        logger.info(f"Found {len(search_results)} relevant chunks")
        return search_results
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        st.error(f"Error searching chunks: {e}")
        return []

# Main application
def main_app():
    # Page configuration
    st.set_page_config(page_title="RAG PDF Chat", layout="wide")
    
    # Initialize feedback CSV file
    init_feedback_csv()
    
    # Check if user is logged in
    if not check_login():
        show_login()
        return
    
    # Display main app
    st.title("Chat with PDF using Local LLM")
    st.sidebar.text(f"Logged in as: {st.session_state.user_email}")
    
    # Logout button
    if st.sidebar.button("Logout"):
        logger.info(f"User logged out: {st.session_state.user_email}")
        st.session_state.logged_in = False
        if "user_email" in st.session_state:
            del st.session_state.user_email
        st.rerun()
        
    # Write to us section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Write to Us")
    st.sidebar.markdown("Have questions or feedback? Contact us at:")
    st.sidebar.markdown("support@example.com")  # Placeholder email to be replaced
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            logger.info(f"User {st.session_state.user_email} uploaded file: {uploaded_file.name}")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            
            if USE_FAISS:
                st.session_state.retriever = None
            else:
                st.session_state.collection_name = None
                
            logger.info(f"User {st.session_state.user_email} cleared chat")
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if USE_FAISS and "retriever" not in st.session_state:
        st.session_state.retriever = None
        
    if not USE_FAISS and "collection_name" not in st.session_state:
        st.session_state.collection_name = None

    # Initialize the embedding model for Qdrant
    @st.cache_resource
    def load_sentence_transformer():
        """Initialize the embedding model from local path or download if not present."""
        try:
            with st.spinner("Loading embedding model..."):
                logger.info("Attempting to load embedding model from local path")
                model = SentenceTransformer(LOCAL_MODEL_PATH)
                st.success("✅ Model loaded from local path")
                logger.info("Successfully loaded embedding model from local path")
        except Exception as e:
            with st.spinner(f"Model not found locally or error loading. Downloading model (this may take a moment)..."):
                logger.warning(f"Error loading local model, downloading: {e}")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                # Save the model for future use
                os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
                model.save(LOCAL_MODEL_PATH)
                st.success("✅ Model downloaded and saved locally")
                logger.info("Downloaded and saved embedding model locally")
        return model

    # Setup Qdrant client
    @st.cache_resource
    def setup_qdrant_client():
        """Setup Qdrant client with local persistence."""
        try:
            logger.info("Setting up Qdrant client")
            client = QdrantClient(path=str(VECTORDB_DIR / "qdrant_db"))
            logger.info("Qdrant client setup complete")
            return client
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            st.error(f"Error connecting to Qdrant: {e}")
            return None

    # Create collection for the PDF if it doesn't exist (Qdrant)
    def create_collection(client, collection_name, vector_size=384):
        """Create a new collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_name not in collection_names:
                logger.info(f"Creating new collection: {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                st.success(f"✅ Collection '{collection_name}' created")
                logger.info(f"Collection '{collection_name}' created successfully")
            else:
                logger.info(f"Using existing collection: {collection_name}")
                st.info(f"Using existing collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            st.error(f"Error creating collection: {e}")

    # Process PDF file
    if uploaded_file is not None:
        # Calculate file hash for caching/collection
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        
        if USE_FAISS:
            # FAISS Implementation
            cache_file = VECTORDB_DIR / f"{file_hash}.pkl"
            
            # Check if we need to process the file
            if st.session_state.retriever is None:
                with st.spinner("Processing PDF with FAISS..."):
                    logger.info(f"Processing PDF with FAISS: {uploaded_file.name}")
                    # Load from cache if exists
                    if cache_file.exists():
                        try:
                            logger.info(f"Attempting to load from cache: {cache_file}")
                            with open(cache_file, "rb") as f:
                                vector_store = pickle.load(f)
                            st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                            st.success("Loaded from cache")
                            logger.info("Successfully loaded vector store from cache")
                        except Exception as e:
                            logger.error(f"Error loading cache: {e}")
                            st.error(f"Error loading cache: {e}")
                            # Will proceed to regenerate
                    
                    # Process if not in cache or failed to load
                    if st.session_state.retriever is None:
                        try:
                            # Process PDF with enhanced metadata
                            vector_store = process_pdf_faiss(file_bytes, file_hash)
                            
                            # Save to cache
                            logger.info(f"Saving vector store to cache: {cache_file}")
                            with open(cache_file, "wb") as f:
                                pickle.dump(vector_store, f)
                            
                            # Create retriever
                            st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                            
                            logger.info("PDF processed successfully with FAISS")
                            st.success("PDF processed successfully with FAISS!")
                        except Exception as e:
                            logger.error(f"Error processing PDF with FAISS: {e}")
                            st.error(f"Error processing PDF with FAISS: {e}")
        else:
            # Qdrant Implementation
            collection_name = f"pdf_{file_hash}"
            
            # Check if we need to process the file
            if st.session_state.collection_name != collection_name:
                with st.spinner("Processing PDF with Qdrant..."):
                    logger.info(f"Processing PDF with Qdrant: {uploaded_file.name}")
                    # Setup Qdrant client and model
                    qdrant_client = setup_qdrant_client()
                    model = load_sentence_transformer()
                    
                    if qdrant_client:
                        # Create collection with appropriate vector size
                        vector_size = model.get_sentence_embedding_dimension()
                        create_collection(qdrant_client, collection_name, vector_size)
                        
                        # Process PDF and add to collection
                        process_pdf_qdrant(file_bytes, collection_name)
                        
                        # Update session state
                        st.session_state.collection_name = collection_name
                        logger.info("PDF processed successfully with Qdrant")
                        st.success("PDF processed successfully with Qdrant!")
                    else:
                        logger.error("Failed to initialize Qdrant client")
                        st.error("Failed to initialize Qdrant client")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1} (Page {source.get('page_num', 'Unknown')}):**")
                        st.markdown(source.get("text", ""))
                        
                        # Display URLs if available
                        urls = source.get("urls", [])
                        if urls:
                            st.markdown("**URLs on this page:**")
                            for url in urls:
                                st.markdown(f"- [{url}]({url})")
                        
                        # Display images if available
                        images = source.get("images", [])
                        if images:
                            st.markdown("**Images on this page:**")
                            for img in images:
                                img_path = img.get("path")
                                if img_path:
                                    st.markdown(display_image(img_path), unsafe_allow_html=True)
                        
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF"):
        logger.info(f"User {st.session_state.user_email} asked: {prompt}")
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get uploaded filename for logging
        current_file = uploaded_file.name if uploaded_file else "No file uploaded"
        
        if USE_FAISS:
            # FAISS implementation for retrieval
            # Check if retriever is available
            if st.session_state.retriever is None:
                response = "Please upload a PDF file first."
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                logger.warning(f"User tried to ask question without uploading PDF first: {prompt}")
                log_feedback(st.session_state.user_email, prompt, current_file, response)
            else:
                # Get relevant documents
                with st.spinner("Thinking..."):
                    try:
                        logger.info("Retrieving relevant chunks with FAISS")
                        # Retrieve relevant chunks
                        docs = st.session_state.retriever.get_relevant_documents(prompt)
                        
                        # Extract text and metadata from docs
                        sources = []
                        for doc in docs:
                            source_data = {
                                "text": doc.page_content,
                                "page_num": doc.metadata.get("page_num", "Unknown"),
                                "urls": doc.metadata.get("urls", []),
                                "images": doc.metadata.get("images", [])
                            }
                            sources.append(source_data)
                            
                        logger.info(f"Retrieved {len(sources)} relevant chunks")
                        
                        # Prepare context
                        context_texts = [source["text"] for source in sources]
                        context = "\n\n".join(context_texts)
                        
                        # Prepare prompt for LLM
                        full_prompt = f"""
                        Answer the following question based on the provided context.
                        
                        Context:
                        {context}
                        
                        Question: {prompt}
                        
                        Answer:
                        """
                        
                        logger.info("Sending prompt to LLM")
                        # Get response from your custom LLM function
                        response = abc_response(full_prompt)  # Your custom LLM function
                        
                        # Fallback if abc_response is not defined
                        if 'abc_response' not in globals():
                            logger.warning("Custom LLM function not found, using fallback response")
                            response = f"Using local LLM to answer: {prompt}\n\nBased on the document, I found relevant information that would help answer this question."
                        
                        # Display assistant response
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            
                            # Show sources
                            with st.expander("View Sources"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1} (Page {source.get('page_num', 'Unknown')}):**")
                                    st.markdown(source.get("text", ""))
                                    
                                    # Display URLs if available
                                    urls = source.get("urls", [])
                                    if urls:
                                        st.markdown("**URLs on this page:**")
                                        for url in urls:
                                            st.markdown(f"- [{url}]({url})")
                                    
                                    # Display images if available
                                    images = source.get("images", [])
                                    if images:
                                        st.markdown("**Images on this page:**")
                                        for img in images:
                                            img_path = img.get("path")
                                            if img_path:
                                                st.markdown(display_image(img_path), unsafe_allow_html=True)
                                    
                                    st.divider()
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                        # Log feedback
                        logger.info(f"Logged response for user {st.session_state.user_email}")
                        log_feedback(st.session_state.user_email, prompt, current_file, response)
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        error_message = f"Error generating response: {str(e)}"
                        with st.chat_message("assistant"):
                            st.markdown(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        log_feedback(st.session_state.user_email, prompt, current_file, error_message)
        else:
            # Qdrant implementation for retrieval
            # Check if collection is available
            if st.session_state.collection_name is None:
                response = "Please upload a PDF file first."
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                logger.warning(f"User tried to ask question without uploading PDF first: {prompt}")
                log_feedback(st.session_state.user_email, prompt, current_file, response)
            else:
                # Get relevant documents
                with st.spinner("Thinking..."):
                    try:
                        logger.info("Retrieving relevant chunks with Qdrant")
                        # Retrieve relevant chunks
                        search_results = search_chunks(st.session_state.collection_name, prompt)
                        
                        if not search_results:
                            logger.warning("No relevant chunks found for query")
                            response = "I couldn't find relevant information in the document to answer your question. Please try a different question or upload a different PDF."
                            with st.chat_message("assistant"):
                                st.markdown(response)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
                            log_feedback(st.session_state.user_email, prompt, current_file, response)
                        else:
                            # Extract text and metadata from search results
                            sources = []
                            for result in search_results:
                                source_data = {
                                    "text": result.payload["text"],
                                    "page_num": result.payload.get("page_num", "Unknown"),
                                    "urls": result.payload.get("urls", []),
                                    "images": result.payload.get("images", [])
                                }
                                sources.append(source_data)
                                
                            logger.info(f"Retrieved {len(sources)} relevant chunks")
                            
                            # Prepare context
                            context_texts = [source["text"] for source in sources]
                            context = "\n\n".join(context_texts)
                            
                            # Prepare prompt for LLM
                            full_prompt = f"""
                            Answer the following question based on the provided context.
                            
                            Context:
                            {context}
                            
                            Question: {prompt}
                            
                            Answer:
                            """
                            
                            logger.info("Sending prompt to LLM")
                            # Get response from your custom LLM function
                            response = abc_response(full_prompt)  # Your custom LLM function
                            
                            # Fallback if abc_response is not defined
                            if 'abc_response' not in globals():
                                logger.warning("Custom LLM function not found, using fallback response")
                                response = f"Using local LLM to answer: {prompt}\n\nBased on the document, I found relevant information that would help answer this question."
                            
                            # Display assistant response
                            with st.chat_message("assistant"):
                                st.markdown(response)
                                
                                # Show sources
                                with st.expander("View Sources"):
                                    for i, source in enumerate(sources):
                                        st.markdown(f"**Source {i+1} (Page {source.get('page_num', 'Unknown')}):**")
                                        st.markdown(source.get("text", ""))
                                        
                                        # Display URLs if available
                                        urls = source.get("urls", [])
                                        if urls:
                                            st.markdown("**URLs on this page:**")
                                            for url in urls:
                                                st.markdown(f"- [{url}]({url})")
                                        
                                        # Display images if available
                                        images = source.get("images", [])
                                        if images:
                                            st.markdown("**Images on this page:**")
                                            for img in images:
                                                img_path = img.get("path")
                                                if img_path:
                                                    st.markdown(display_image(img_path), unsafe_allow_html=True)
                                        
                                        st.divider()
                            
                            # Add assistant message to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "sources": sources
                            })
                            
                            # Log feedback
                            logger.info(f"Logged response for user {st.session_state.user_email}")
                            log_feedback(st.session_state.user_email, prompt, current_file, response)
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        error_message = f"Error generating response: {str(e)}"
                        with st.chat_message("assistant"):
                            st.markdown(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        log_feedback(st.session_state.user_email, prompt, current_file, error_message)

# Define a placeholder for abc_response if not provided
def abc_response(prompt):
    """Placeholder function for LLM response."""
    return f"This is a placeholder response for: {prompt}\n\nIn a production environment, this would use your custom LLM."

# Run the main function
if __name__ == "__main__":
    try:
        logger.info("Starting application")
        main_app()
    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)

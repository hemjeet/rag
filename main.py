import os
import streamlit as st
import tempfile
from rag_system import MyFirstRag
from utils import stream_data
from google.cloud import storage
import uuid
from utils import *

# Set up the page
st.set_page_config(
    page_title="RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LLM Provider Options
LLM_PROVIDERS = {
    "OpenAI": "openai",
    "DeepSeek": "deepseek"
}

# Model options for each provider
MODEL_OPTIONS = {
    "openai": {
        "GPT-5": "gpt-5",
        "GPT-4": "gpt-4",
        "GPT-4o": "gpt-4o",
        "GPT-4o Mini": "gpt-4o-mini",
    },
    "deepseek": {
        "DeepSeek Chat": "deepseek-chat",
        "DeepSeek Reasoning": "deepseek-reasoning"
    }
}

EMBEDDING_OPTIONS = {
   "Small (fastest, lower quality)":  "text-embedding-3-small",
    "Large (slower, higher quality)": "text-embedding-3-large",
    "Ada-002 (balanced)": "text-embedding-ada-002"
}



# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "vector_store_path" not in st.session_state:
    st.session_state.vector_store_path = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]  # Unique session ID for GCS paths


def clear_query():
    """Clear the query input"""
    st.session_state.query = ""

# Sidebar for API keys and settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.subheader("GCS Configuration")
    st.info(f"Bucket: {GCS_BUCKET_NAME}")
    st.info(f"Session ID: {st.session_state.session_id}")
    
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.document_processed = False
        st.session_state.rag = None
        st.rerun()
    
    st.subheader("LLM Provider Selection")
    llm_provider = st.selectbox(
        "Choose LLM Provider",
        options = list(LLM_PROVIDERS.keys()),
        index=0,
        help="Select which provider to use for the language model"
    )
    
    st.subheader("API Keys")
    
    # Show appropriate API key input based on provider
    if llm_provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Get your API key from https://platform.openai.com/")
        os.environ["OPENAI_API_KEY"] = api_key if api_key else ""
    else:  # DeepSeek
        api_key = st.text_input("DeepSeek API Key", type="password",
                               help="Get your API key from https://platform.deepseek.com/")
        os.environ["DEEPSEEK_API_KEY"] = api_key if api_key else ""
    
    
    st.subheader("Model Selection")
    
    # Show model options based on selected provider
    provider_key = LLM_PROVIDERS[llm_provider]
    
    selected_model_name = st.selectbox(
        "Chat Model",
        options = list(MODEL_OPTIONS[provider_key].keys()),
        index=0,
        help="Select which model to use for generating answers"
    )
    
    if llm_provider == 'DeepSeek':
        openai_embedding_key = st.text_input("OpenAI API Key for embeddings", type="password", 
                               help="Get your API key from https://platform.openai.com/")
        
        os.environ['OPENAI_API_KEY_FOR_EMBEDDINGS'] = openai_embedding_key

    selected_embedding_model = st.selectbox(
        "Embedding Model",
        options = list(EMBEDDING_OPTIONS.keys()),
        index = 0,
        help = "Select which OpenAI model to use for creating document embeddings"
    )

    st.subheader('Set temperature for model')
    
    temp = st.slider(
        "Temperature", 
        min_value = 0.0, 
        max_value = 1.0, 
        value = 0.0, 
        step = 0.1,
        help = "Controls randomness: 0 = deterministic, 1 = more creative"
    )
    
    st.subheader("Processing Options")
    use_reranking = st.checkbox("Use Reranking", value=False, 
                               help = "Enable NVIDIA reranking for better results (requires NVIDIA API key)")
    
    if use_reranking:
        nvidia_key = st.text_input("NVIDIA API Key", type = "password",
                                  help = "Required for reranking functionality")
        if nvidia_key:
            os.environ["NVIDIA_API_KEY"] = nvidia_key
        else:
            # Clear the environment variable if no key is provided
            os.environ.pop("NVIDIA_API_KEY", None)
    else:
        # Clear the environment variable when reranking is disabled
        os.environ.pop("NVIDIA_API_KEY", None)

    st.subheader("Document Processing")
    uploaded_file = st.file_uploader("Upload a text file", type = ["md", "txt"])
    
    if uploaded_file is not None:
        # Store file content in session state
        if st.session_state.uploaded_file_content is None:
            gcs_path, file_content = save_uploaded_file(uploaded_file)
            
            if gcs_path:
                st.session_state.uploaded_gcs_path = gcs_path
                st.session_state.uploaded_file_content = file_content
                st.session_state.uploaded_file_name = uploaded_file.name
        
        st.info(f"File uploaded: {uploaded_file.name}")
        
        # Initialize RAG system
        if st.button("Process Document"):
            if not api_key:
                st.error(f"Please provide an {llm_provider} API key")
            else:
                with st.spinner("Processing document and building indexes..."):
                    try:
                        # Create a temporary file for processing
                        local_file_path = download_from_gcs(st.session_state.uploaded_gcs_path)
                        
                        provider = LLM_PROVIDERS[llm_provider]
                        model_name = MODEL_OPTIONS[provider][selected_model_name]
                        embedding_model = EMBEDDING_OPTIONS[selected_embedding_model]
                        
                        # Use GCS path for vector store
                        vector_store_path = f"gs://{GCS_BUCKET_NAME}/{GCS_BASE_PATH}/{st.session_state.session_id}/vector_store"
                        
                        if local_file_path:
                            # Create temporary directory for vector store
                            temp_vector_dir = tempfile.mkdtemp()
                            st.session_state.temp_vector_store_dir = temp_vector_dir
                            st.session_state.local_file_path = local_file_path
                        
                            # Initialize RAG with LOCAL paths only
                            st.session_state.rag = MyFirstRag(
                                document_path = local_file_path,  # Local file path
                                vector_store_path = temp_vector_dir,  # Local directory path
                                model_name = model_name,
                                embedding_model = embedding_model,
                                temperature = temp,
                                llm_provider = provider
                            )
                            st.session_state.document_processed = True
                            st.success("Document processed successfully!")
                        else:
                            st.error("Failed to download file from GCS")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {e}")

#==============================================================
# MAIN CHAT SECTION 
#===============================================================

st.title("📚 RAG System")
st.markdown("Upload a text file and ask questions about its content.")

if not st.session_state.document_processed:
    st.info("👈 Please upload a text file and process it using the sidebar.")
else:
    # Display current model selection
    if st.session_state.rag:
        st.session_state.rag.vector_store.load_local(st.session_state.temp_vector_store_dir)
        st.sidebar.info(f"Using: {llm_provider} - {selected_model_name} with {selected_embedding_model} embeddings")
        st.sidebar.info(f"Vector Store: {st.session_state.vector_store_path}")
    
    # Display file info
    if st.session_state.uploaded_file_name:
        st.subheader(f"Document: {st.session_state.uploaded_file_name}")
    
    # Query section
    st.subheader("Ask a question about the document")
    
    # query form
    with st.form(key = "query_form", clear_on_submit = True):
        query = st.text_input(
            "Enter your question:", 
            placeholder = "What would you like to know about the document?",
            key = "query_input",
            value = st.session_state.query
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            
            submit_button = st.form_submit_button("Ask Question")
        
        with col2:
            
            clear_button = st.form_submit_button("Clear", on_click=clear_query)
    
    if submit_button and query:
        with st.spinner("Searching for answers..."):
            try:
                answer = st.session_state.rag.create_rag_chain(query, rerank = use_reranking)
                
                if st.session_state.rag.llm_provider == "deepseek":
                    st.write_stream(stream_data(answer, query))
                else:
                    st.session_state.answer = answer
                    st.session_state.query = ''
            
            except Exception as e:
                st.error(f"Error generating answer: {e}")
    
    # Display the answer if available
    if st.session_state.answer:
        st.subheader("Answer:")
        st.code(st.session_state.answer, language=None)
        
        # Add a button to clear the answer
        if st.button("Clear Answer"):
            st.session_state.answer = None
            st.rerun()
    
    # Display document info
    if st.checkbox("Show document preview"):
        if st.session_state.uploaded_file_content:
            st.subheader("Document Preview (first 1000 characters)")
            st.text_area("Document content", 
                        st.session_state.uploaded_file_content[:1000] + "..." 
                        if len(st.session_state.uploaded_file_content) > 1000 else st.session_state.uploaded_file_content,
                        height=200,
                        disabled=True)

# Instructions section
with st.expander("How to use this app"):
    st.markdown("""
    1. **GCS Setup**: Files and vector stores are automatically saved to Google Cloud Storage
    2. **Select Provider**: Choose between OpenAI or DeepSeek for the language model
    3. **Set API Keys**: Enter your API key for the selected provider
    4. **Select Models**: Choose chat and embedding models
    5. **Upload Document**: Upload a text file containing the content you want to query
    6. **Process Document**: Click the 'Process Document' button to build the search indexes
    7. **Ask Questions**: Enter questions about the document content in the main panel
    8. **New Session**: Use 'New Session' button to start fresh with different documents
    
    **GCP Features:**
    - All files stored in Google Cloud Storage
    - Vector indexes persisted in GCS for reuse
    - Session-based file management
    - Automatic cleanup of temporary files
    """)

# Footer
st.markdown("---")
st.caption("Built with LangChain, OpenAI/DeepSeek, FAISS, Streamlit, and Google Cloud Storage")

# Cleanup function for GCS (optional - you might want to keep files for persistence)
def cleanup_gcs():
    """Clean up GCS files for this session when app closes"""
    try:
        if hasattr(st.session_state, 'session_id'):
            prefix = f"{GCS_BASE_PATH}/{st.session_state.session_id}/"
            files = list_gcs_files(prefix)
            for file_path in files:
                delete_from_gcs(file_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

# Register cleanup function
import atexit
atexit.register(cleanup_gcs)
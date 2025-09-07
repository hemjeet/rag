import os
from google.cloud import storage
import streamlit as st
import tempfile


# GCS Configuration
def stream_data(chain, query):
    for i in chain.stream({'input': query}):
        if 'answer' in i:
            yield i['answer']


#===============================
# GCP CONFIG
#===============================
# GCS_BUCKET_NAME = os.getenv('bucket_name')
# GCS_BASE_PATH = os.getenv('gcs_base_path')
GCS_BUCKET_NAME = 'rag-file'
GCS_BASE_PATH = 'files'


def get_gcs_client():
    """Initialize and return GCS client"""
    try:
        # For GCP deployment, credentials are automatically handled
        return storage.Client()
    except Exception as e:
        st.error(f"Error initializing GCS client: {e}")
        return None

# GCS Utility functions
def upload_to_gcs(file_content, file_name, content_type = "text/plain"):
    """Upload file to GCS and return the GCS path"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        bucket = client.bucket(GCS_BUCKET_NAME)
        gcs_path = f"{GCS_BASE_PATH}/{st.session_state.session_id}/{file_name}"
        blob = bucket.blob(gcs_path)
        
        blob.upload_from_string(file_content, content_type=content_type)
        return gcs_path
    except Exception as e:
        st.error(f"Error uploading to GCS: {e}")
        return None

def download_from_gcs(gcs_path):
    """Download file content from GCS"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        return blob.download_as_text()
    except Exception as e:
        st.error(f"Error downloading from GCS: {e}")
        return None

def list_gcs_files(prefix):
    """List files in GCS with given prefix"""
    try:
        client = get_gcs_client()
        if not client:
            return []
            
        bucket = client.bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix = prefix)
        return [blob.name for blob in blobs]
    except Exception as e:
        st.error(f"Error listing GCS files: {e}")
        return []

def delete_from_gcs(gcs_path):
    """Delete file from GCS"""
    try:
        client = get_gcs_client()
        if not client:
            return False
            
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.delete()
        return True
    except Exception as e:
        st.error(f"Error deleting from GCS: {e}")
        return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file content and upload to GCS"""
    try:
        content = uploaded_file.read().decode('utf-8')
        file_name = uploaded_file.name
        
        # Upload to GCS
        gcs_path = upload_to_gcs(content, file_name)
        if gcs_path:
            return gcs_path, content
        return None, None
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None, None
def get_gcs_file_path(file_name):
    """Get the full GCS path for a file"""
    return f"gs://{GCS_BUCKET_NAME}/{GCS_BASE_PATH}/{st.session_state.session_id}/{file_name}"

def get_gcs_vector_store_path():
    """Get the GCS path for vector store"""
    return f"gs://{GCS_BUCKET_NAME}/{GCS_BASE_PATH}/{st.session_state.session_id}/vector_store/"


def download_from_gcs(gcs_path):
    """Download file content from GCS to a temporary file"""
    try:
        client = get_gcs_client()
        if not client:
            return None
            
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete = False, suffix=".md", mode='w', encoding='utf-8') as tmp_file:
            content = blob.download_as_text()
            tmp_file.write(content)
            return tmp_file.name
            
    except Exception as e:
        st.error(f"Error downloading from GCS: {e}")
        return None



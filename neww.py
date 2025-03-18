# Install required dependencies

import os
import streamlit as st
import os
import sys
# Force ChromaDB to use pysqlite3 instead of system SQLite
os.environ["SQLITE_LIBRARY_PATH"] = sys.prefix + "/lib"
os.environ["LD_LIBRARY_PATH"] = sys.prefix + "/lib"
os.environ["PATH"] += os.pathsep + sys.prefix + "/bin"

import chromadb
import logging
import torch
import re
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gradio_client import Client
from dotenv import load_dotenv
import pdfplumber

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LegalRAG:
    def __init__(self, embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                 llm_model='LEGALSUMMAI',
                 db_path='./chroma_db', collection_name='legal_docs'):
        load_dotenv()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.llm = Client("LEGALSUMMAI")
        
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def extract_text(self, uploaded_file) -> str:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                return '\n\n'.join([page.extract_text() or '' for page in pdf.pages])
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""

    def process_document(self, text: str) -> list:
        try:
            chunks = self.text_splitter.split_text(text)
            embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
            return [{"id": str(hash(chunk)), "chunk": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
        except Exception as e:
            logger.error(f"Chunking & embedding error: {e}")
            return []

    def upload_document(self, document_data: list) -> bool:
        try:
            existing_ids = set(self.collection.get(ids=[d['id'] for d in document_data])['ids'])
            new_data = [d for d in document_data if d['id'] not in existing_ids]
            
            if new_data:
                self.collection.add(ids=[d['id'] for d in new_data],
                                    embeddings=[d['embedding'] for d in new_data],
                                    documents=[d['chunk'] for d in new_data])
                logger.info(f"Uploaded {len(new_data)} new chunks.")
            return True
        except Exception as e:
            logger.error(f"ChromaDB upload error: {e}")
            return False

    def retrieve_and_summarize(self, query: str) -> str:
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False).tolist()
            results = self.collection.query(query_embeddings=[query_embedding], n_results=3)
            
            if not results["documents"]:
                return "No relevant information found."
            context = '\n\n'.join(results["documents"][0])
            
            template = f'''
            You are a Legal AI Assistant specializing in **Indian Laws and the Indian Constitution**. Answer only based on Indian legal context:
            
            **Context:** {context}
            
            **User Query:** {query}
            
            **Important Guidelines:**
            - Only provide answers related to Indian laws and the Indian Constitution.
            - If the query is unrelated to Indian legal matters, respond with: "I specialize in Indian laws and the Indian Constitution. I cannot provide information on this topic."
            - Ensure your response is structured and references applicable Indian legal provisions.
            '''
            
            response = self.llm.predict(Query=template, api_name="/predict")
            
            # Remove unwanted metadata from response
            cleaned_response = re.sub(r"<.*?>", "", response)
            
            return cleaned_response.strip()
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return "Error generating summary."

# Streamlit Deployment
def main():
    st.set_page_config(page_title="🏛️ LegalBot: Indian Law Expert", page_icon="⚖️", layout="wide")
    st.title("🏛️ LegalBot: Indian Law Expert")

    rag_system = LegalRAG()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], help="Upload a legal document for analysis.")
    if uploaded_file:
        with st.spinner("Processing..."):
            text = rag_system.extract_text(uploaded_file)
            document_data = rag_system.process_document(text)
            if rag_system.upload_document(document_data):
                st.success("Document processed successfully!")

    query = st.text_input("Enter your legal query:", placeholder="Ask a question about Indian laws...")
    if query:
        with st.spinner("Generating Summary..."):
            summary = rag_system.retrieve_and_summarize(query)
        st.markdown("### 📄 Summary")
        st.markdown(summary)

if __name__ == "__main__":
    main()

from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
import os
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
from langchain.schema import BaseRetriever, Document as LangChainDocument
from langchain_community.vectorstores import Chroma
import faiss
import numpy as np
import torch

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, str]

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            # Set device to CPU for better compatibility
            self.device = "cpu"
            torch.set_num_threads(4)  # Limit CPU threads
            
            # Initialize the model with error handling
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Initialize FAISS index
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Store documents
            self.documents = []
            
            # Setup logging
            self.logger = logging.getLogger(__name__)
            
        except Exception as e:
            print(f"Error initializing VectorStore: {str(e)}")
            raise

    def add_documents(self, texts: List[str]) -> None:
        try:
            if not texts:
                return
                
            # Generate embeddings in batches for better performance
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.model.encode(batch, convert_to_numpy=True)
                self.index.add(embeddings)
                self.documents.extend(batch)
                
            self.logger.info(f"Added {len(texts)} documents to vector store")
                
        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        try:
            if not self.documents:
                self.logger.warning("No documents in vector store")
                return []
                
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search in FAISS index
            k = min(k, len(self.documents))  # Don't request more results than we have documents
            if k == 0:
                return []
                
            distances, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.documents):  # Ensure index is valid
                    results.append({
                        "text": self.documents[idx],
                        "score": float(distances[0][i])
                    })
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []

    def save(self, path: str) -> None:
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
            
            # Save documents
            with open(os.path.join(path, "documents.txt"), "w", encoding="utf-8") as f:
                for doc in self.documents:
                    f.write(doc + "\n")
                    
            self.logger.info(f"Saved vector store to {path}")
                    
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
            raise

    def load(self, path: str) -> None:
        try:
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(path, "index.faiss"))
            
            # Load documents
            with open(os.path.join(path, "documents.txt"), "r", encoding="utf-8") as f:
                self.documents = [line.strip() for line in f]
                
            self.logger.info(f"Loaded vector store from {path}")
                
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            raise

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> BaseRetriever:
        """Get the vector store as a LangChain retriever."""
        return self.db.as_retriever(
            search_kwargs=search_kwargs or {"k": 5}
        )

    def delete_all(self) -> None:
        """Delete all documents from the vector store."""
        try:
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            self.logger.info("Deleted all documents from vector store")
        except Exception as e:
            self.logger.error(f"Error deleting documents from vector store: {str(e)}")
            raise 
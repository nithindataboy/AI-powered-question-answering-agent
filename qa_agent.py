import os
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from crawler import WebCrawler, PageContent
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import hashlib
from datetime import datetime, timedelta
import re

# Load environment variables
load_dotenv()

class QAAgent:
    def __init__(self):
        try:
            # Set up logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            
            # Initialize sentence transformer for semantic search
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading sentence transformer: {str(e)}")
                self.model = None
            
            # Initialize FAISS index
            self.dimension = 768  # Dimension for all-mpnet-base-v2
            self.index = faiss.IndexFlatL2(self.dimension)
            
            # Store documents and their metadata
            self.documents = []
            self.document_metadata = []
            
            # Initialize answer cache
            self.answer_cache: Dict[str, Dict] = {}
            self.cache_ttl = timedelta(hours=24)
            
            self.knowledge_base = []  # Initialize knowledge base
            self.processed_urls = set()  # Track processed URLs
            self.embeddings = None  # Will be initialized when needed
            
        except Exception as e:
            self.logger.error(f"Error initializing QA Agent: {str(e)}")
            raise

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split text into smaller chunks for better context preservation."""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence.split())
            if current_size + sentence_size > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def process_documentation(self, pages: List[Dict]) -> None:
        """Process crawled pages and create vector store."""
        if not pages:
            self.logger.warning("No pages to process")
            return

        self.logger.info(f"Processing {len(pages)} pages...")
        
        # Process each page
        for page in pages:
            if isinstance(page, dict) and 'content' in page:
                # Extract content and metadata
                content = page['content']
                title = page.get('title', '')
                url = page.get('url', '')
                
                # Skip if content is too short
                if len(content.split()) < 10:
                    self.logger.warning(f"Skipping page {url} - insufficient content")
                    continue
                
                # Split content into chunks
                chunks = self._chunk_text(content)
                
                # Process each chunk
                for chunk in chunks:
                    # Skip if chunk is too short
                    if len(chunk.split()) < 5:
                        continue
                        
                    # Create text with context
                    text = f"Title: {title}\nURL: {url}\nContent: {chunk}"
                    self.documents.append(text)
                    self.document_metadata.append({
                        "url": url,
                        "title": title,
                        "chunk": chunk
                    })

        if not self.documents:
            self.logger.warning("No content found in any pages")
            return

        self.logger.info(f"Created {len(self.documents)} chunks from {len(pages)} pages")
        
        # Create embeddings for all chunks
        embeddings = self.model.encode(self.documents)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        self.logger.info(f"Documentation processing completed. Indexed {len(self.documents)} chunks.")

    def _get_cache_key(self, question: str) -> str:
        """Generate a cache key for the question."""
        return hashlib.md5(question.encode()).hexdigest()

    def _get_cached_answer(self, question: str) -> Optional[Dict]:
        """Get cached answer if available and not expired."""
        cache_key = self._get_cache_key(question)
        if cache_key in self.answer_cache:
            cached_data = self.answer_cache[cache_key]
            if datetime.now() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["answer"]
        return None

    def _cache_answer(self, question: str, answer: Dict) -> None:
        """Cache the answer with timestamp."""
        cache_key = self._get_cache_key(question)
        self.answer_cache[cache_key] = {
            "answer": answer,
            "timestamp": datetime.now()
        }

    def _format_answer(self, relevant_docs: List[str], relevant_metadata: List[Dict], confidence: float) -> Dict:
        """Format the answer with relevant information."""
        # Combine chunks from the same document
        doc_groups = {}
        for doc, meta in zip(relevant_docs, relevant_metadata):
            url = meta["url"]
            if url not in doc_groups:
                doc_groups[url] = {
                    "title": meta["title"],
                    "chunks": [],
                    "confidence": confidence
                }
            doc_groups[url]["chunks"].append(meta["chunk"])

        # Format the answer
        formatted_answer = []
        for url, info in doc_groups.items():
            # Skip if content is not relevant
            if len(info["chunks"]) == 0:
                continue
                
            formatted_answer.append(f"Title: {info['title']}")
            formatted_answer.append(f"Source: {url}")
            formatted_answer.append("Content:")
            
            # Combine chunks with better formatting
            content = " ".join(info["chunks"])
            # Clean up the content
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'([.!?])\s*', r'\1\n', content)
            
            # Add bullet points for lists
            content = re.sub(r'^\s*[-•*]\s*', '• ', content, flags=re.MULTILINE)
            
            # Remove any non-English content
            content = re.sub(r'[^\x00-\x7F]+', '', content)
            
            formatted_answer.append(content)
            formatted_answer.append("-" * 80)

        if not formatted_answer:
            return {
                "answer": "No relevant information found in the documentation.",
                "confidence": 0.0,
                "source_url": "",
                "num_sources": 0
            }

        return {
            "answer": "\n".join(formatted_answer),
            "confidence": round(confidence, 2),
            "source_url": url,
            "num_sources": len(doc_groups)
        }

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Sentence transformer model not loaded")
        return self.model.encode(texts).tolist()
        
    def _find_relevant_content(self, question: str, top_k: int = 3) -> List[Dict]:
        """Find relevant content using semantic search."""
        try:
            if not self.knowledge_base:
                return []
                
            # Get question embedding
            question_embedding = self._get_embeddings([question])[0]
            
            # Get content embeddings if not already computed
            if self.embeddings is None:
                self.logger.info("Computing embeddings for knowledge base...")
                content_texts = [doc["content"] for doc in self.knowledge_base]
                self.embeddings = self._get_embeddings(content_texts)
            
            # Calculate similarities
            similarities = []
            for i, embedding in enumerate(self.embeddings):
                similarity = self._cosine_similarity(question_embedding, embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity and get top k
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:top_k]]
            
            return [self.knowledge_base[i] for i in top_indices]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []
            
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def answer_question(self, question: str) -> Dict:
        """Answer a question based on the knowledge base using semantic search."""
        try:
            if not self.knowledge_base:
                return {
                    "error": "No content has been processed yet. Please process a website first.",
                    "confidence": 0,
                    "num_sources": 0
                }
            
            # Clean question
            question = self._clean_content(question)
            
            # Find relevant content using semantic search
            relevant_content = self._find_relevant_content(question)
            
            if not relevant_content:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "confidence": 0,
                    "num_sources": 0
                }
            
            # Generate answer
            answer = f"Based on the documentation, here's what I found:\n\n"
            for doc in relevant_content:
                answer += f"From {doc['title']}:\n{doc['content']}\n\n"
            
            # Calculate confidence based on number of relevant sources
            confidence = min(1.0, len(relevant_content) / 3)  # Cap at 1.0
            
            return {
                "answer": answer,
                "confidence": confidence,
                "num_sources": len(relevant_content),
                "sources": relevant_content
            }
            
        except Exception as e:
            self.logger.error(f"Error answering question: {str(e)}")
            return {
                "error": f"Failed to answer question: {str(e)}",
                "confidence": 0,
                "num_sources": 0
            }

    def crawl_and_process(self, url: str) -> None:
        """Crawl a website and process its content."""
        try:
            self.logger.info(f"Processing website: {url}")
            
            # Check if URL is already processed
            if url in self.processed_urls:
                self.logger.info(f"URL {url} already processed, skipping")
                return
            
            # Crawl website
            crawler = WebCrawler(url)
            pages = crawler.crawl()
            
            if not pages:
                self.logger.warning("No pages were crawled")
                return
            
            # Process each page
            for page in pages:
                try:
                    # Clean and chunk content
                    cleaned_content = self._clean_content(page["content"])
                    chunks = self._chunk_content(cleaned_content)
                    
                    # Add to knowledge base
                    for chunk in chunks:
                        self.knowledge_base.append({
                            "url": page["url"],
                            "title": page["title"],
                            "content": chunk
                        })
                except Exception as e:
                    self.logger.error(f"Error processing page {page['url']}: {str(e)}")
                    continue
            
            # Mark URL as processed
            self.processed_urls.add(url)
            self.logger.info(f"Successfully processed website: {url}")
            
        except Exception as e:
            self.logger.error(f"Error processing website {url}: {str(e)}")
            raise ValueError(f"Failed to process website: {str(e)}")
            
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        # Remove special characters
        content = re.sub(r'[^\w\s.,!?-]', '', content)
        return content
        
    def _chunk_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Split content into chunks."""
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize and run agent
    agent = QAAgent()
    
    try:
        # Process website
        url = input("Enter website URL (e.g., https://help.example.com): ").strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        print(f"\nStarting to process website: {url}")
        agent.crawl_and_process(url)
        
        # Interactive question answering
        print("\nWebsite processed successfully! You can now ask questions.")
        print("Type 'exit' to quit.\n")
        
        while True:
            question = input("Your question: ").strip()
            if question.lower() == 'exit':
                break
                
            answer = agent.answer_question(question)
            if "error" in answer:
                print(f"\nError: {answer['error']}\n")
            else:
                print(f"\nAnswer: {answer['answer']}")
                print(f"Confidence: {answer['confidence']*100:.1f}%")
                print(f"Number of sources: {answer['num_sources']}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
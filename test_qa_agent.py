import pytest
from qa_agent import QAAgent
from crawler import WebCrawler, PageContent
from vector_store import VectorStore, Document
import os
from unittest.mock import Mock, patch
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

@pytest.fixture
def mock_huggingface_pipeline():
    with patch('transformers.pipeline') as mock_pipeline:
        # Create a mock pipeline that returns a simple response
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "This is a test response about feature X. You can enable it in Settings."}]
        mock_pipeline.return_value = mock_pipe
        yield mock_pipeline

@pytest.fixture
def qa_agent(mock_huggingface_pipeline):
    with patch('qa_agent.HuggingFacePipeline') as mock_hf:
        # Create a mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = {"result": "This is a test response about feature X. You can enable it in Settings. Source: https://example.com/features"}
        mock_hf.return_value = mock_llm
        
        # Create a mock model and tokenizer
        with patch('qa_agent.AutoModelForCausalLM.from_pretrained') as mock_model, \
             patch('qa_agent.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_model.return_value = Mock()
            mock_tokenizer.return_value = Mock()
            
            agent = QAAgent()
            yield agent

@pytest.fixture
def mock_page_content():
    return PageContent(
        url="https://example.com/test",
        title="Test Page",
        content="This is a test page content. It contains some information about features.",
        links=["https://example.com/test2"]
    )

def test_process_website(qa_agent, mock_page_content):
    with patch('crawler.WebCrawler.crawl') as mock_crawl:
        mock_crawl.return_value = [mock_page_content]
        
        # Process website
        qa_agent.process_website("https://example.com")
        
        # Verify vector store was updated
        results = qa_agent.vector_store.search("features")
        assert len(results) > 0
        assert "features" in results[0]['content'].lower()

def test_answer_question(qa_agent):
    # Add some test documents to the vector store
    test_doc = Document(
        id="test1",
        content="The product supports feature X. To enable it, go to Settings > Features.",
        metadata={"url": "https://example.com/features", "title": "Features Guide"}
    )
    qa_agent.vector_store.add_documents([test_doc])
    
    # Test question answering
    answer = qa_agent.answer_question("How do I enable feature X?")
    assert "feature X" in answer.lower()
    assert "settings" in answer.lower()
    assert "https://example.com/features" in answer

def test_error_handling(qa_agent):
    # Test with invalid URL
    with pytest.raises(Exception):
        qa_agent.process_website("invalid-url")
    
    # Test with empty question
    answer = qa_agent.answer_question("")
    assert "error" in answer.lower()

def test_vector_store_search():
    vector_store = VectorStore()
    
    # Add test documents
    docs = [
        Document(
            id="1",
            content="This is about feature A",
            metadata={"url": "https://example.com/a", "title": "Feature A"}
        ),
        Document(
            id="2",
            content="This is about feature B",
            metadata={"url": "https://example.com/b", "title": "Feature B"}
        )
    ]
    vector_store.add_documents(docs)
    
    # Test search
    results = vector_store.search("feature A")
    assert len(results) > 0
    assert "feature A" in results[0]['content'].lower()

if __name__ == "__main__":
    pytest.main([__file__]) 
# Help Website QA Assistant

An AI-powered question-answering agent that processes documentation from help websites and provides accurate answers to user queries.

## Features

- 🕷️ Web crawling of help/documentation websites
- 🔍 Semantic search using sentence transformers
- 📊 Confidence scoring for answers
- 🔗 Source attribution with clickable links
- 💾 Answer caching for improved performance
- 🎯 Support for multiple documentation formats
- 🚀 Docker containerization support
- 📱 Streamlit web interface
- 💻 Command-line interface

## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/help-website-qa.git
cd help-website-qa
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup

```bash
docker build -t help-website-qa .
docker run -p 8501:8501 help-website-qa
```

## Usage

### Web Interface

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter a help website URL and start asking questions!

### Command Line Interface

```bash
python app.py --url https://help.example.com
```

## Technical Architecture

### Components

1. **Web Crawler**
   - Handles URL validation and content extraction
   - Implements rate limiting and parallel processing
   - Extracts structured content (headings, lists, tables)

2. **QA Agent**
   - Uses sentence transformers for semantic search
   - Implements answer caching
   - Provides confidence scoring
   - Handles multiple documentation sources

3. **User Interface**
   - Streamlit web interface
   - Command-line interface
   - Docker containerization

### Implementation Approach

- **Semantic Search**: Uses `sentence-transformers` for embedding generation
- **Content Processing**: Implements chunking and cleaning of documentation
- **Answer Generation**: Combines relevant content with confidence scoring
- **Caching**: Implements TTL-based caching for improved performance

## Testing

### Unit Tests

Run the test suite:
```bash
python -m pytest tests/
```

### Example Test Cases

1. Basic functionality:
```python
def test_basic_qa():
    agent = QAAgent()
    agent.crawl_and_process("https://help.example.com")
    answer = agent.answer_question("How do I get started?")
    assert answer["confidence"] > 0
```

2. Performance benchmarks:
```python
def test_performance():
    agent = QAAgent()
    start_time = time.time()
    agent.crawl_and_process("https://help.example.com")
    assert time.time() - start_time < 300  # Should complete within 5 minutes
```

## Design Decisions

1. **Semantic Search**: Chose sentence transformers over keyword matching for better answer relevance
2. **Content Chunking**: Implemented smart chunking to preserve context
3. **Caching**: Added TTL-based caching to improve response times
4. **UI**: Selected Streamlit for rapid development and user-friendly interface

## Known Limitations

1. **Content Extraction**: May struggle with complex JavaScript-rendered content
2. **Rate Limiting**: Some websites may block aggressive crawling
3. **Memory Usage**: Large documentation sets may require significant memory
4. **Language Support**: Currently optimized for English content

## Future Improvements

1. **Advanced Features**
   - Support for more documentation formats (PDF, Word)
   - Multi-language support
   - Advanced NLP techniques for better answer generation
   - Integration with more LLM APIs

2. **Technical Improvements**
   - Distributed crawling
   - Vector database integration
   - API rate limiting
   - Advanced caching strategies

## Dependencies

- streamlit
- beautifulsoup4
- requests
- sentence-transformers
- numpy
- faiss-cpu
- python-dotenv

## Assumptions

1. Documentation websites follow standard HTML structure
2. Content is primarily in English
3. Websites allow reasonable crawling rates
4. Documentation is publicly accessible

## License

MIT License 
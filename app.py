import streamlit as st
from qa_agent import QAAgent
from crawler import WebCrawler
import logging
from urllib.parse import urlparse
import sys
import markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'qa_agent' not in st.session_state:
    st.session_state.qa_agent = QAAgent()
if 'processed_url' not in st.session_state:
    st.session_state.processed_url = None

# Set page title and description
st.title("📚 Help Website QA Assistant")
st.markdown("""
This assistant helps you find answers from help and documentation websites.

**Core Functionality:**
- Accepts help website URLs (e.g., help.zluri.com, help.slack.com)
- Processes and indexes documentation content
- Answers questions about product features, integrations, and functionality
- Clearly indicates when information is not available
""")

# Create two columns for URL input and search
col1, col2 = st.columns([2, 1])

# URL input in first column
with col1:
    url = st.text_input(
        "Enter help website URL:",
        placeholder="e.g., help.zluri.com",
        help="Enter the URL of a help/documentation website"
    )

# Search bar in second column
with col2:
    search_query = st.text_input(
        "Search documentation:",
        placeholder="Type your question here",
        help="Ask questions about the documentation"
    )

# Process website when URL is submitted
if url and url != st.session_state.processed_url:
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    # Validate URL
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            st.error("Invalid URL. Please enter a valid help website URL.")
            st.stop()
            
        # Check if URL looks like a help/documentation site
        if not any(domain in parsed.netloc.lower() for domain in ['help.', 'docs.', 'documentation.', 'support.']):
            st.warning("This doesn't appear to be a help/documentation website. Results may be limited.")
            
    except Exception as e:
        st.error(f"Invalid URL: {str(e)}")
        st.stop()
        
    try:
        with st.spinner("Processing help documentation... This may take a minute..."):
            st.session_state.qa_agent.crawl_and_process(url)
            st.session_state.processed_url = url
        st.success("Documentation processed successfully!")
        
        # Show processed content in a structured format
        st.markdown("### 📚 Processed Documentation")
        for page in st.session_state.qa_agent.knowledge_base:
            with st.expander(f"📄 {page['title']}"):
                st.markdown(f"**URL:** [{page['url']}]({page['url']})")
                st.markdown("---")
                st.markdown(page['content'], unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        st.stop()

# Handle search queries
if search_query and st.session_state.processed_url:
    try:
        with st.spinner("Searching documentation..."):
            answer = st.session_state.qa_agent.answer_question(search_query)
            
        if "error" in answer:
            st.error(answer["error"])
        else:
            # Display answer
            st.markdown("### Answer")
            st.markdown(answer["answer"], unsafe_allow_html=True)
            
            # Show confidence and sources
            if answer["confidence"] > 0:
                st.markdown(f"**Confidence:** {answer['confidence']*100:.1f}%")
                st.markdown(f"**Number of sources:** {answer['num_sources']}")
                
                # Show source documents
                if answer.get('sources'):
                    st.markdown("### Source Documents")
                    for source in answer['sources']:
                        with st.expander(f"📄 {source['title']}"):
                            st.markdown(f"**URL:** [{source['url']}]({source['url']})")
                            st.markdown("---")
                            st.markdown(source['content'], unsafe_allow_html=True)
            else:
                st.info("No relevant information found in the documentation.")
            
    except Exception as e:
        st.error(f"Error searching documentation: {str(e)}")
elif search_query and not st.session_state.processed_url:
    st.warning("Please process a documentation website first before searching.")

# Add command line interface support
if len(sys.argv) > 1:
    if sys.argv[1] == '--url' and len(sys.argv) > 2:
        url = sys.argv[2]
        try:
            st.session_state.qa_agent.crawl_and_process(url)
            print("Documentation processed successfully!")
            while True:
                question = input("\nEnter your question (or 'quit' to exit): ")
                if question.lower() == 'quit':
                    break
                answer = st.session_state.qa_agent.answer_question(question)
                if "error" in answer:
                    print(f"Error: {answer['error']}")
                else:
                    print("\nAnswer:")
                    print(answer["answer"])
                    if answer["confidence"] > 0:
                        print(f"\nConfidence: {answer['confidence']*100:.1f}%")
                        print(f"Number of sources: {answer['num_sources']}")
                        if answer.get('sources'):
                            print("\nSource Documents:")
                            for source in answer['sources']:
                                print(f"\nTitle: {source['title']}")
                                print(f"URL: {source['url']}")
                                print(f"Content: {source['content'][:200]}...")
                    else:
                        print("No relevant information found in the documentation.")
        except Exception as e:
            print(f"Error: {str(e)}") 
from crewai import Agent, Task, Crew
from langchain.tools import tool
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class DocumentationAgents:
    def __init__(self):
        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            model_name='gemini-1.0-pro-001',
            generation_config={
                "temperature": 0.1,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        # Initialize LangChain Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.0-pro-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
    @tool("Crawl Documentation")
    def crawl_documentation(self, url: str) -> List[Dict]:
        """Crawl a documentation website and extract content."""
        from crawler import WebCrawler
        crawler = WebCrawler(url, max_pages=10, timeout=30)
        return crawler.crawl()
    
    @tool("Process Content")
    def process_content(self, pages: List[Dict]) -> Dict:
        """Process crawled content and create vector store."""
        from qa_agent import QAAgent
        agent = QAAgent()
        agent.process_documentation(pages)
        return {"status": "success", "message": "Content processed successfully"}
    
    @tool("Answer Question")
    def answer_question(self, question: str) -> Dict:
        """Answer a question based on processed documentation."""
        from qa_agent import QAAgent
        agent = QAAgent()
        return agent.answer_question(question)

    def create_agents(self):
        """Create and return the agents for the documentation QA system."""
        # Content Crawler Agent
        crawler_agent = Agent(
            role='Content Crawler',
            goal='Crawl documentation websites and extract relevant content',
            backstory="""You are an expert web crawler specialized in documentation websites.
            You know how to extract meaningful content while filtering out navigation elements
            and maintaining proper context hierarchy.""",
            tools=[self.crawl_documentation],
            llm=self.llm,
            verbose=True
        )
        
        # Content Processor Agent
        processor_agent = Agent(
            role='Content Processor',
            goal='Process and index documentation content for efficient querying',
            backstory="""You are an expert in processing and organizing documentation content.
            You know how to maintain context and structure while preparing content for
            efficient retrieval.""",
            tools=[self.process_content],
            llm=self.llm,
            verbose=True
        )
        
        # QA Agent
        qa_agent = Agent(
            role='Question Answerer',
            goal='Answer questions based on processed documentation',
            backstory="""You are an expert in answering questions about documentation.
            You provide clear, accurate answers and indicate when information is not available.""",
            tools=[self.answer_question],
            llm=self.llm,
            verbose=True
        )
        
        return crawler_agent, processor_agent, qa_agent

    def create_crew(self, url: str):
        """Create and return the crew for processing documentation."""
        crawler_agent, processor_agent, qa_agent = self.create_agents()
        
        # Create tasks
        crawl_task = Task(
            description=f"Crawl the documentation website at {url}",
            agent=crawler_agent
        )
        
        process_task = Task(
            description="Process the crawled content and create vector store",
            agent=processor_agent
        )
        
        # Create crew
        crew = Crew(
            agents=[crawler_agent, processor_agent, qa_agent],
            tasks=[crawl_task, process_task],
            verbose=True
        )
        
        return crew 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn
from qa_agent import QAAgent
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG FAQ Agent API",
    description="API for processing help websites and answering questions using free and open-source models",
    version="1.0.0"
)

# Initialize QA agent
agent = QAAgent()

class WebsiteRequest(BaseModel):
    url: HttpUrl

class QuestionRequest(BaseModel):
    question: str

class Response(BaseModel):
    answer: str

@app.post("/process-website", response_model=dict)
async def process_website(request: WebsiteRequest):
    """Process a help website and store its content."""
    try:
        agent.process_website(str(request.url))
        return {"message": "Website processed successfully"}
    except Exception as e:
        logger.error(f"Error processing website: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=Response)
async def ask_question(request: QuestionRequest):
    """Ask a question about the processed website."""
    try:
        answer = agent.answer_question(request.question)
        return Response(answer=answer)
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 
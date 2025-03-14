from contextlib import asynccontextmanager
from fastapi import HTTPException, APIRouter, status
from starlette import status
from app.models import QuestionRequest

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import re  # Add this import
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
KB_FILE_PATH = "data/resume.pdf"

vector_index = None

def initialize_knowledge_base():
    global vector_index
    try:
        logger.info(f"Loading PDF from {KB_FILE_PATH}")
        pdf_loader = PyPDFLoader(KB_FILE_PATH)
        pages = pdf_loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        context = "\n\n".join(str(page.page_content) for page in pages)
        texts = text_splitter.split_text(context)
        
        logger.info("Creating embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
        logger.info("Knowledge base loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        return False

router = APIRouter(
    prefix='/v2/chatbot',
    tags=['Chatbot']
)


initialize_knowledge_base()

def format_markdown_response(text: str) -> str:
    """
    Converts markdown text into a well-formatted readable response.
    Handles bullet points, headings, and general text formatting.
    """

    text = re.sub(r'<[^>]+>', '', text)
    

    text = re.sub(r'^#+ (.+)$', r'\1:', text, flags=re.MULTILINE)
    

    text = re.sub(r'^\s*[-*]\s+(.+)$', r'• \1', text, flags=re.MULTILINE)
    

    text = re.sub(r'^\s*\d+\.\s+(.+)$', r'• \1', text, flags=re.MULTILINE)
    

    text = re.sub(r'• ', '\n• ', text)
    

    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  
    text = re.sub(r'_(.+?)_', r'\1', text)        
    
    
    text = re.sub(r'\n{3,}', '\n\n', text)        
    text = re.sub(r':\s*', ': ', text)           
    text = re.sub(r'([.!?])\n• ', r'\1\n\n• ', text)  
    

    text = text.replace('\n', '<br/>')
    

    text = text.strip()
    
    return text

@router.post("/ask-question/", status_code=status.HTTP_200_OK)
async def ask_question(question_request: QuestionRequest):
    logger.info(f"Received request: {question_request}")
    
    if vector_index is None:
        success = initialize_knowledge_base()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initialize knowledge base")
    
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
        
        template = """You are an AI consultant who can guide user's to their dream careers. Your role is to help people understand
          people's skill level, expertise and provide them with career guidance in a professional and 
          informative manner.

          FORMATTING GUIDELINES:
          1. Use markdown formatting for structure
          2. Use bullet points (-) for lists
          3. Use headings (#) for sections
          4. Keep responses concise and well-organized
          5. If information isn't available, say so politely
          """
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        result = qa_chain({"query": question_request.question})
        formatted_answer = format_markdown_response(result["result"])
        
        return {"answer": formatted_answer}
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 

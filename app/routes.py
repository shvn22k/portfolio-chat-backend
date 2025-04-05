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
from typing import Optional, Dict
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
KB_FILE_PATH = "../data/resume.pdf"

vector_index = None

# Fallback responses for common questions
FALLBACK_RESPONSES = {"Getting an error, please contact shiven to inform."}

def get_fallback_response(query: str) -> Optional[str]:
    """Try to match a query to predefined responses for when the AI service fails"""
    # Normalize the query
    query = query.lower().strip()
    
    # Check for exact matches first
    if query in FALLBACK_RESPONSES:
        return FALLBACK_RESPONSES[query]
    
    # Check for more specific partial matches (more strict matching)
    for key, response in FALLBACK_RESPONSES.items():
        # Match only if all words in the key appear in the query
        key_words = key.lower().split()
        if len(key_words) > 1 and all(word in query for word in key_words):
            return response
            
    return None

def initialize_knowledge_base():
    global vector_index
    try:
        logger.info(f"Loading PDF from {KB_FILE_PATH}")
        # Check if file exists
        if not os.path.exists(KB_FILE_PATH):
            logger.error(f"PDF file not found at path: {KB_FILE_PATH}")
            logger.error(f"Current working directory: {os.getcwd()}")
            return False
            
        pdf_loader = PyPDFLoader(KB_FILE_PATH)
        pages = pdf_loader.load_and_split()
        
        # Add log to check content
        logger.info(f"Loaded {len(pages)} pages from PDF")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        context = "\n\n".join(str(page.page_content) for page in pages)
        texts = text_splitter.split_text(context)
        
        logger.info(f"Split text into {len(texts)} chunks")
        logger.info("Creating embeddings...")
        
        # Check if API key is present
        if not GOOGLE_API_KEY:
            logger.error("Google API key is not set or empty")
            return False
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )
        
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
        logger.info("Knowledge base loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        # Check for specific errors
        if "API Key not found" in str(e) or "API_KEY_INVALID" in str(e):
            logger.error("Google API key is invalid or missing")
        elif "File path" in str(e) and "is not a valid file" in str(e):
            logger.error(f"PDF file not found at {KB_FILE_PATH}. Check that the file exists.")
        # Log traceback for debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    # Only use fallback response if there's an error, not by default
    fallback = None
    
    if vector_index is None:
        logger.info("Vector index is None, initializing knowledge base")
        success = initialize_knowledge_base()
        if not success:
            error_msg = "Failed to initialize knowledge base. Check server logs for details."
            logger.error(error_msg)
            # Now try fallback response
            fallback = get_fallback_response(question_request.question)
            if fallback:
                return {"answer": fallback}
            return {"answer": "I'm having trouble accessing my knowledge base at the moment. Please try again in a few moments while I get back online."}
    
    try:
        logger.info("Creating language model")
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
        
        template = """You are an AI assistant trained on Shiven's resume. Your role is to help people understand
          Shiven's experience, skills, and qualifications by answering their questions in a professional and 
          informative manner. DO NOT EXAGGERATE OR MAKE UP INFORMATION ABOUT SHIVEN'S SKILLS OR EXPERIENCE JUST ANSWER THE QUESTION AS IT IS IN THE RESUME.

          FORMATTING GUIDELINES:
          1. Use markdown formatting for structure
          2. Use bullet points (-) for lists
          3. Use headings (#) for sections
          4. Keep responses concise and well-organized
          5. If information isn't available, say so politely

          Example Response:
          # Key Projects
          - Project 1: Brief description
          - Project 2: Clear explanation
          - Project 3: Main achievements

          {context}
          
          Question: {question}
          Helpful Answer:"""
        
        logger.info("Setting up QA chain")
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        logger.info(f"Processing query: {question_request.question}")
        result = qa_chain({"query": question_request.question})
        logger.info("Got result from QA chain")
        
        formatted_answer = format_markdown_response(result["result"])
        
        return {"answer": formatted_answer}
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing question: {error_message}")
        
        # Log full traceback
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Only use fallback response when the main system fails
        fallback = get_fallback_response(question_request.question)
        if fallback:
            return {"answer": fallback}
            
        # Otherwise return generic message
        fallback_response = """I apologize, but I'm having trouble connecting to my knowledge database right now. 
        
This could be due to:
• My API key needs updating
• The AI model is currently unavailable
• There's a temporary service interruption

Please try again in a few moments. If the problem persists, it might need technical attention."""

        return {"answer": fallback_response} 
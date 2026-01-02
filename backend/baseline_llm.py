"""
Baseline LLM Module - Direct LLM interaction without RAG
"""
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import get_settings


BASELINE_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on your knowledge.
Be concise and accurate. If you don't know the answer, say so honestly.
Answer in Korean."""


class BaselineLLM:
    """Direct LLM interaction without any retrieval"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize BaselineLLM with optional custom LLM configuration
        
        Args:
            api_key: OpenAI API key (uses env if not provided)
            model: Model name (uses env if not provided)
            base_url: Custom API base URL (uses env if not provided)
        """
        settings = get_settings()
        
        # Use provided values or fall back to settings
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        
        # Base URL priority: user-provided > settings > OpenAI default
        if base_url and base_url.strip():
            # User explicitly provided a custom base URL (e.g., Together AI, Groq)
            self.base_url = base_url
        elif api_key:
            # User provided API key but no base URL -> use OpenAI official API
            self.base_url = "https://api.openai.com/v1"
        else:
            # Use settings (sandbox proxy)
            self.base_url = settings.openai_base_url
        
        # Initialize LLM
        llm_kwargs = {
            "model": self.model,
            "temperature": 0.7,
            "api_key": self.api_key,
        }
        
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", BASELINE_SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm
    
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Generate answer using only LLM knowledge (no retrieval)
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            response = self.chain.invoke({"question": question})
            
            return {
                "success": True,
                "answer": response.content,
                "model": self.model,
                "method": "baseline",
                "context": None,
                "sources": []
            }
        except Exception as e:
            return {
                "success": False,
                "answer": "",
                "error": str(e),
                "method": "baseline"
            }

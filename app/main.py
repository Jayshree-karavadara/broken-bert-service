"""
Main FastAPI application for sentiment analysis API.

This module sets up the FastAPI application, handles model loading during startup,
and configures all the necessary middleware and routes.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add the parent directory to the Python path to import ml module
sys.path.append(str(Path(__file__).parent.parent))

from ml.model import ReviewClassifier
from app.endpoints import router, set_classifier
from app.schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global classifier instance
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Handles startup and shutdown events.
    """
    global classifier
    
    # Startup
    logger.info("Starting sentiment analysis API...")
    
    try:
        # Load the model during startup
        model_path = "assets/model.pth"
        tokenizer_path = "assets/tokenizer/"
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            logger.error("Please run 'python -m ml.train' first to train the model.")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not os.path.exists(tokenizer_path):
            logger.error(f"Tokenizer directory not found: {tokenizer_path}")
            logger.error("Please run 'python -m ml.train' first to train the model.")
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        
        # Initialize the classifier
        classifier = ReviewClassifier(
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
        
        # Set the classifier in the endpoints module
        set_classifier(classifier)
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model info: {classifier.get_model_info()}")
        
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down sentiment analysis API...")
    # Cleanup if needed
    classifier = None


# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using fine-tuned DistilBERT model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API routes
app.include_router(router)


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "NotFound",
            "message": "The requested endpoint was not found",
            "details": f"Path '{request.url.path}' does not exist"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal server error occurred",
            "details": None
        }
    )


def main():
    """
    Main function to run the FastAPI application.
    """
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting server at http://{host}:{port}")
    logger.info("API Documentation available at:")
    logger.info(f"  - Swagger UI: http://{host}:{port}/docs")
    logger.info(f"  - ReDoc: http://{host}:{port}/redoc")
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
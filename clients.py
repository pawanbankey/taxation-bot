# clients.py
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncAzureOpenAI
from config import Config
from logger import logger

# Global Clients
mongo_client: Optional[AsyncIOMotorClient] = None
db = None
collection = None
aoai_client: Optional[AsyncAzureOpenAI] = None

def init_clients():
    """Initialize global clients - call once at startup."""
    global mongo_client, db, collection, aoai_client
    
    try:
        mongo_client = AsyncIOMotorClient(
            Config.MONGO_URI,
            serverSelectionTimeoutMS=5000,
            maxPoolSize=50,
            minPoolSize=10
        )
        db = mongo_client[Config.DB_NAME]
        collection = db[Config.COLLECTION_NAME]
        logger.info("MongoDB client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB client: {e}")
        raise

    try:
        aoai_client = AsyncAzureOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            max_retries=3,
            timeout=30.0
        )
        logger.info("Azure OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}")
        raise

async def close_clients():
    """Gracefully close all clients."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB client closed")

async def init_db():
    """Initialize database indexes."""
    global collection
    try:
        await collection.create_index(
            [("user_id", 1)],
            background=True
        )
        await collection.create_index([("last_updated", -1)], background=True)
        logger.info("MongoDB indexes verified successfully")
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
        raise

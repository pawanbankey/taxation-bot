# config.py
class Config:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = ""
    AZURE_OPENAI_API_KEY = ""
    AZURE_OPENAI_API_VERSION = ""
    AZURE_OPENAI_DEPLOYMENT = ""   

    # Azure Search
    AZURE_SEARCH_ENDPOINT = ""
    AZURE_SEARCH_API_KEY = ""

    # MongoDB
    MONGO_URI = ""
    DB_NAME = ""
    COLLECTION_NAME = "" 

    # Available Search Indexes
    SEARCH_INDEXES = [
        "taxation-acts",
        "taxation-rules",
        "taxation-circular-notification"
    ]
import json
import logging
import datetime
import urllib3
import requests
import faiss
from typing import Any, Dict, List, Optional
from pymongo import MongoClient, DESCENDING

# Azure SDK imports
from openai import AzureOpenAI

# ---------------------------------------------------------
# 1. CONFIGURATION & SETUP
# ---------------------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

# --- CREDENTIALS (REPLACE WITH ENV VARIABLES IN PRODUCTION) ---
AZURE_OPENAI_ENDPOINT = "https://hrtransformation.openai.azure.com/"
AZURE_OPENAI_API_KEY = "AFd7UMq9pERgXT9w1z7IOvcD84px0dHEUHwHgMh1PDycGm6JJVC5JQQJ99BBAC77bzfXJ3w3AAABACOG28lv"
AZURE_OPENAI_API_VERSION = "2024-05-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"

AZURE_SEARCH_ENDPOINT = "https://hrtransformation.search.windows.net"
AZURE_SEARCH_API_KEY = "kFixV6YiahYG6AvOIInbQQfqCJrXbN0IUWTlw9ZYkOAzSeDFroKy"

# MongoDB Config
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "taxation_bot"
COLLECTION_NAME = "chat_history"

# Index Mapping
INDEX_MAP = {
    "Act": "taxation-acts",
    "Rule": "taxation-rules",
    "Circular": "taxation-circular-notification",
}

# ---------------------------------------------------------
# 2. MONGODB HISTORY MANAGER
# ---------------------------------------------------------
class MongoHistoryManager:
    """
    Manages chat history storage and retrieval for multi-user environments.
    """
    def __init__(self, user_id: str, conversation_id: str):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        
        # Unique filter for specific user in specific chat thread
        self.filter_query = {
            "user_id": user_id, 
            "conversation_id": conversation_id
        }

    def save_turn(self, original_query: str, llm_answer: str):
        """
        Saves the ORIGINAL user query and the final LLM response with a timestamp.
        """
        timestamp = datetime.datetime.utcnow()
        
        new_entry = {
            "role_user": "user",
            "content_user": original_query,
            "role_assistant": "assistant",
            "content_assistant": llm_answer,
            "timestamp": timestamp
        }

        # We push to a 'turns' array inside the user's document
        self.collection.update_one(
            self.filter_query,
            {
                "$push": {"turns": new_entry},
                "$set": {"last_updated": timestamp}
            },
            upsert=True
        )
        logging.info(f"Saved turn to MongoDB for User: {self.filter_query['user_id']}")

    def get_latest_context(self, num_turns: int = 2) -> List[Dict[str, str]]:
        """
        Fetches the last N turns to provide context for follow-up questions.
        Returns a list of messages formatted for Azure OpenAI.
        """
        doc = self.collection.find_one(
            self.filter_query, 
            {"turns": {"$slice": -num_turns}} # Fetch only last N turns
        )
        
        messages = []
        if doc and "turns" in doc:
            for turn in doc["turns"]:
                # Reconstruct chat format
                messages.append({"role": "user", "content": turn["content_user"]})
                messages.append({"role": "assistant", "content": turn["content_assistant"]})
        
        return messages

# ---------------------------------------------------------
# 3. AZURE OPENAI & SEARCH CLIENTS
# ---------------------------------------------------------
def gpt_client_json(prompt: str) -> Dict[str, Any]:
    """Helper for internal reasoning (Classification/Rewriting) returning JSON."""
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    try:
        completion = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a backend processor. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            timeout=30
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        logging.error(f"GPT JSON Error: {e}")
        return {}

def gpt_client_chat(messages: List[Dict[str, str]]) -> str:
    """Main Chat Client handling history."""
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    try:
        completion = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.7,
            timeout=60
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"GPT Chat Error: {e}")
        return "I encountered an error generating the response."

def search_azure_index(query: str, index_name: str, top_k: int = 3) -> List[str]:
    """Searches Azure AI Search and returns chunk text."""
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index_name}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    body = {"search": query, "top": top_k}
    
    try:
        resp = requests.post(url, headers=headers, json=body, verify=False, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [doc.get("chunk", "") for doc in data.get("value", []) if doc.get("chunk")]
    except Exception as e:
        logging.error(f"Search Error on {index_name}: {e}")
        return []

# ---------------------------------------------------------
# 4. RAG PIPELINE LOGIC (Internal)
# ---------------------------------------------------------
def run_rag_retrieval(user_query: str) -> str:
    """
    Executes: Classify -> Rewrite -> Search -> Aggregate Context
    Returns: A single string containing the retrieved context.
    """
    logging.info("Starting RAG Retrieval Pipeline...")
    
    # Step A: Classify
    class_prompt = f"Classify query: '{user_query}' into categories [Act, Rule, Circular]. Return JSON: {{'category': ['Act']}}"
    cats = gpt_client_json(class_prompt).get("category", ["Act"])
    if isinstance(cats, str): cats = [cats]
    
    # Step B: Rewrite
    rewrite_prompt = f"Rewrite for search relevance: '{user_query}'. Return JSON: {{'rewritten': '...'}}"
    rewritten_query = gpt_client_json(rewrite_prompt).get("rewritten", user_query)
    logging.info(f"Rewritten Query: {rewritten_query}")

    # Step C: Search
    all_chunks = []
    for cat in cats:
        if cat in INDEX_MAP:
            chunks = search_azure_index(rewritten_query, INDEX_MAP[cat])
            all_chunks.extend(chunks)
    
    # Step D: Format Context
    if not all_chunks:
        return "No specific documents found."
    
    # Deduplicate and format
    unique_chunks = list(set(all_chunks))[:5] # Top 5 chunks
    formatted_context = "\n\n".join([f"[Doc {i+1}]: {chunk}" for i, chunk in enumerate(unique_chunks)])
    return formatted_context

# ---------------------------------------------------------
# 5. MAIN ORCHESTRATOR
# ---------------------------------------------------------
def process_user_query(user_query: str, user_id: str, conversation_id: str, is_follow_up: bool = False) -> str:
    """
    Main entry point for the bot.
    """
    # Initialize History Manager for this specific user/chat
    history_db = MongoHistoryManager(user_id, conversation_id)
    
    # Base System Prompt
    messages = [
        {"role": "system", "content": "You are a Taxation Expert for Indian Income Tax. Answer based on context provided."}
    ]

    if is_follow_up:
        logging.info(f"--- Follow-Up Query Detected (User: {user_id}) ---")
        logging.info("Skipping RAG Retrieval. Fetching history from MongoDB...")
        
        # 1. Fetch History (Last query + Last Answer)
        prev_history = history_db.get_latest_context(num_turns=2) # Gets last 2 turns (User, AI, User, AI)
        
        if not prev_history:
            # Fallback if user claims follow-up but no history exists
            logging.warning("No history found in DB, reverting to RAG.")
            return process_user_query(user_query, user_id, conversation_id, is_follow_up=False)

        # 2. Append History & Current Query
        messages.extend(prev_history)
        messages.append({"role": "user", "content": user_query})
        
    else:
        logging.info(f"--- Initial Query Detected (User: {user_id}) ---")
        
        # 1. Run RAG Pipeline
        retrieved_context = run_rag_retrieval(user_query)
        
        # 2. Create Prompt with Context
        full_prompt = (
            f"Please answer the following question using the provided context.\n\n"
            f"--- CONTEXT START ---\n{retrieved_context}\n--- CONTEXT END ---\n\n"
            f"Question: {user_query}"
        )
        messages.append({"role": "user", "content": full_prompt})

    # 3. Get LLM Response
    logging.info("Calling Azure OpenAI...")
    final_answer = gpt_client_chat(messages)
    
    # 4. Save to MongoDB (Original Query + Answer)
    logging.info("Saving interaction to MongoDB...")
    history_db.save_turn(user_query, final_answer)
    
    return final_answer

# ---------------------------------------------------------
# 6. EXAMPLE USAGE (Simulating Teams)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Simulate User Details from Teams (Activity Object)
    TEAMS_USER_ID = "29:1-abc-123-teams-id"
    TEAMS_CHAT_ID = "a:1-xyz-789-thread-id"
    
    print("="*60)
    
    # --- TURN 1: Initial Query (Triggers RAG) ---
    q1 = "What are the tax rates for domestic companies for AY 2025-26?"
    print(f"User: {q1}")
    ans1 = process_user_query(q1, TEAMS_USER_ID, TEAMS_CHAT_ID, is_follow_up=False)
    print(f"Bot: {ans1}\n")
    
    print("-" * 60)
    
    # --- TURN 2: Follow-up (Triggers Mongo Fetch) ---
    q2 = "Does this include the health cess?" 
    # Logic to detect follow-up would determine 'is_follow_up=True' here
    print(f"User (Follow-up): {q2}")
    ans2 = process_user_query(q2, TEAMS_USER_ID, TEAMS_CHAT_ID, is_follow_up=True)
    print(f"Bot: {ans2}\n")

    print("="*60)
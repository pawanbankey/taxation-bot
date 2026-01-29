import json
import logging
import datetime
import asyncio
import os
from typing import Any, Dict, List, Optional
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncAzureOpenAI

# ---------------------------------------------------------
# 1. CONFIGURATION (Environment Variables recommended)
# ---------------------------------------------------------
# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

class Config:
    # Azure OpenAI
    AOAI_ENDPOINT = "https://hrtransformation.openai.azure.com/"
    AOAI_KEY = "AFd7UMq9pERgXT9w1z7IOvcD84px0dHEUHwHgMh1PDycGm6JJVC5JQQJ99BBAC77bzfXJ3w3AAABACOG28lv"
    AOAI_VERSION = "2024-05-01-preview"
    AOAI_DEPLOYMENT = "gpt-4.1"

    # Azure AI Search
    SEARCH_ENDPOINT = "https://hrtransformation.search.windows.net"
    SEARCH_KEY = "kFixV6YiahYG6AvOIInbQQfqCJrXbN0IUWTlw9ZYkOAzSeDFroKy"
    
    # MongoDB
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
# 2. GLOBAL CLIENTS (Connection Pooling)
# ---------------------------------------------------------
# Reusing clients prevents overhead of re-establishing connections
mongo_client = AsyncIOMotorClient(Config.MONGO_URI)
db = mongo_client[Config.DB_NAME]
collection = db[Config.COLLECTION_NAME]

aoai_client = AsyncAzureOpenAI(
    azure_endpoint=Config.AOAI_ENDPOINT,
    api_key=Config.AOAI_KEY,
    api_version=Config.AOAI_VERSION
)

# Ensure Indexes (Run once on startup)
async def init_db():
    # Compound index for fast retrieval by user and thread
    await collection.create_index([("user_id", 1), ("conversation_id", 1)])
    logging.info("MongoDB Indexes Verified.")

# ---------------------------------------------------------
# 3. ASYNC DATABASE MANAGER
# ---------------------------------------------------------
class AsyncHistoryManager:
    def __init__(self, user_id: str, conversation_id: str):
        self.filter = {"user_id": user_id, "conversation_id": conversation_id}

    async def save_turn(self, user_q: str, ai_a: str):
        timestamp = datetime.datetime.utcnow()
        new_entry = {
            "role_user": "user",
            "content_user": user_q,
            "role_assistant": "assistant",
            "content_assistant": ai_a,
            "timestamp": timestamp
        }
        await collection.update_one(
            self.filter,
            {
                "$push": {"turns": new_entry},
                "$set": {"last_updated": timestamp}
            },
            upsert=True
        )

    async def get_recent_history(self, limit: int = 1) -> Optional[Dict]:
        """Fetches the last N turns. Efficient projection to save bandwidth."""
        doc = await collection.find_one(
            self.filter,
            {"turns": {"$slice": -limit}}
        )
        if doc and "turns" in doc and doc["turns"]:
            return doc["turns"]
        return None

# ---------------------------------------------------------
# 4. ASYNC SEARCH & GPT TOOLS
# ---------------------------------------------------------
async def search_azure_index(session: aiohttp.ClientSession, query: str, index_name: str, top_k: int = 3) -> List[str]:
    """Async HTTP call to Azure Search REST API."""
    url = f"{Config.SEARCH_ENDPOINT}/indexes/{index_name}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": Config.SEARCH_KEY}
    body = {"search": query, "top": top_k}
    
    try:
        async with session.post(url, headers=headers, json=body, ssl=False) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [doc.get("chunk", "") for doc in data.get("value", []) if doc.get("chunk")]
            else:
                logging.error(f"Search Failed {resp.status}: {await resp.text()}")
                return []
    except Exception as e:
        logging.error(f"Async Search Error: {e}")
        return []

async def analyze_and_route(user_query: str, last_turn: Optional[Dict]) -> Dict[str, Any]:
    """
    COMBINED ROUTER: Detects intent AND rewrites query in ONE call.
    Returns JSON: { "is_follow_up": bool, "rewritten_query": str, "search_needed": bool }
    """
    
    # Context String
    prev_context = ""
    if last_turn:
        prev_context = f"""
        PREVIOUS Q: {last_turn[-1]['content_user']}
        PREVIOUS A: {last_turn[-1]['content_assistant']}
        """

    prompt = f"""
    You are a Router. Analyze the New Query based on the Previous Context (if any).
    
    Tasks:
    1. Boolean 'is_follow_up': True if user refers to previous context (e.g., "what about for seniors?", "explain more"). False if it's a new independent topic.
    2. String 'rewritten_query': Rewrite the query to be standalone and keyword-rich for search. If it is a follow-up, resolve pronouns (e.g., change "it" to "Tax Slabs").
    
    Context: {prev_context}
    New Query: {user_query}
    
    Return JSON ONLY: {{ "is_follow_up": bool, "rewritten_query": "string" }}
    """
    
    try:
        response = await aoai_client.chat.completions.create(
            model=Config.AOAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Router Error: {e}")
        # Fallback safe defaults
        return {"is_follow_up": False, "rewritten_query": user_query}

# ---------------------------------------------------------
# 5. MAIN ASYNC ORCHESTRATOR
# ---------------------------------------------------------
async def process_message(user_query: str, user_id: str, conversation_id: str) -> str:
    db_manager = AsyncHistoryManager(user_id, conversation_id)
    
    # 1. Fetch History (Concurrent with other ops if needed, but we need it for routing)
    history_turns = await db_manager.get_recent_history(limit=3)
    
    # 2. Route & Rewrite (The Brain)
    route_data = await analyze_and_route(user_query, history_turns)
    is_follow_up = route_data.get("is_follow_up", False)
    rewritten_query = route_data.get("rewritten_query", user_query)
    
    logging.info(f"Router Decision: Follow-Up={is_follow_up} | Rewritten='{rewritten_query}'")

    messages = [{"role": "system", "content": "You are a Taxation Expert."}]

    # 3. Branching Logic
    if is_follow_up and history_turns:
        # --- PATH A: CONTEXTUAL FOLLOW-UP ---
        # Load history into messages
        for turn in history_turns:
            messages.append({"role": "user", "content": turn["content_user"]})
            messages.append({"role": "assistant", "content": turn["content_assistant"]})
        
        messages.append({"role": "user", "content": user_query})
        
    else:
        # --- PATH B: NEW RAG RETRIEVAL ---
        # Concurrent Search across all indexes
        async with aiohttp.ClientSession() as session:
            tasks = [
                search_azure_index(session, rewritten_query, idx_name)
                for idx_name in Config.INDEX_MAP.values()
            ]
            results = await asyncio.gather(*tasks)
        
        # Flatten and Deduplicate results
        all_chunks = list(set([chunk for sublist in results for chunk in sublist]))[:5]
        
        if not all_chunks:
            context_text = "No specific documents found."
        else:
            context_text = "\n\n".join(all_chunks)

        # Build RAG Prompt
        rag_prompt = f"""
        Context:
        {context_text}
        
        Question: {user_query}
        """
        messages.append({"role": "user", "content": rag_prompt})

    # 4. Generate Answer
    response = await aoai_client.chat.completions.create(
        model=Config.AOAI_DEPLOYMENT,
        messages=messages,
        temperature=0.7
    )
    final_answer = response.choices[0].message.content

    # 5. Save Turn (Fire and forget - assume success to return faster, or await)
    await db_manager.save_turn(user_query, final_answer)
    
    return final_answer

# ---------------------------------------------------------
# 6. RUNNER (Simulating concurrent users)
# ---------------------------------------------------------
async def main():
    await init_db()
    
    # Simulating 2 users chatting at the same time
    user1_task = process_message("What are tax rates for companies?", "UserA", "Chat1")
    user2_task = process_message("Tell me about Section 80C", "UserB", "Chat2")
    
    results = await asyncio.gather(user1_task, user2_task)
    
    print(f"\nUser A: {results[0]}")
    print(f"User B: {results[1]}")

    # Simulating Follow-up for User A
    print("\n--- User A Follow-up ---")
    follow_up = await process_message("Does this include cess?", "UserA", "Chat1")
    print(f"User A (Turn 2): {follow_up}")

if __name__ == "__main__":
    # Windows-specific event loop fix if needed
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
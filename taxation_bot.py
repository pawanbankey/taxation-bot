import json
import logging
import datetime
import asyncio
from typing import Any, Dict, List, Optional
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncAzureOpenAI

# --------------------------------------------------------- 
# 1. CONFIGURATION (Environment Variables recommended) 
# --------------------------------------------------------- 
# Logging setup - console only (all logs saved to MongoDB)

class LogCaptureHandler(logging.Handler):
    """Custom handler to capture logs in memory during request processing."""
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record)
            }
            self.logs.append(log_entry)
        except Exception:
            self.handleError(record)
    
    def get_logs(self) -> List[Dict]:
        return self.logs

# Create console handler only
console_handler = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
console_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
)
logger = logging.getLogger("taxation_bot")
logger.info("Logging initialized. Logs saved to MongoDB only.")

class Config:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = ""
    AZURE_OPENAI_API_KEY = ""
    AZURE_OPENAI_API_VERSION = ""
    AZURE_OPENAI_DEPLOYMENT = ""   

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

# ---------------------------------------------------------
# 2. GLOBAL CLIENTS (Connection Pooling)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 3. ASYNC DATABASE MANAGER
# ---------------------------------------------------------
class AsyncHistoryManager:
    def __init__(self, user_id: str):
        if not user_id:
            raise ValueError("user_id must be non-empty string")
        self.filter = {"user_id": user_id}

    async def save_turn(self, user_q: str, ai_a: str, is_follow_up: bool, confidence_score: float, combined_query: Optional[str] = None, status: str = "Success", logs: Optional[List[Dict]] = None) -> bool:
        """Save conversation turn with logs. Returns True on success."""
        try:
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            new_entry = {
                "user_query": user_q,
                "bot_response": ai_a,
                "is_follow_up": is_follow_up,
                "confidence_score": confidence_score,
                "combined_query": combined_query, 
                "status": status, 
                "timestamp": timestamp,
                "logs": logs or []  # Save logs as array
            }
            result = await collection.update_one(
                self.filter,
                {
                    "$push": {"turns": new_entry},
                    "$set": {"last_updated": timestamp},
                    "$setOnInsert": {
                        "user_id": self.filter["user_id"],
                        "created_at": timestamp
                    }
                },
                upsert=True
            )
            logger.info(f"Saved turn for {self.filter['user_id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to save turn: {e}", exc_info=True)
            return False

    async def get_recent_history(self, limit: int = 3) -> List[Dict]:
        """Fetches the last N turns."""
        try:
            doc = await collection.find_one(
                self.filter,
                {"turns": {"$slice": -limit}, "_id": 0}
            )
            if doc and "turns" in doc and doc["turns"]:
                logger.info(f"Retrieved {len(doc['turns'])} turns for {self.filter['user_id']}")
                return doc["turns"]
            return []
        except Exception as e:
            logger.error(f"Failed to retrieve history: {e}", exc_info=True)
            return []
    
    async def get_last_combined_query(self) -> Optional[str]:
        """Get the combined query from the last turn for follow-up chaining."""
        try:
            doc = await collection.find_one(
                self.filter,
                {"turns": {"$slice": -1}, "_id": 0}  # Get only the last turn
            )
            if doc and "turns" in doc and doc["turns"]:
                last_turn = doc["turns"][-1]
                combined_query = last_turn.get("combined_query")
                if combined_query:
                    logger.info(f"Retrieved combined query from last turn: {combined_query}")
                    return combined_query
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve last combined query: {e}", exc_info=True)
            return None


# ---------------------------------------------------------
# 4. ASYNC SEARCH & GPT TOOLS
# ---------------------------------------------------------
async def search_azure_index(
    session: aiohttp.ClientSession,
    query: str,
    index_name: str,
    top_k: int = 5
) -> List[str]:
    """Async HTTP call to Azure Search REST API."""
    url = f"{Config.AZURE_SEARCH_ENDPOINT}/indexes/{index_name}/docs/search?api-version=2024-07-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": Config.AZURE_SEARCH_API_KEY
    }
    body = {"search": query, "top": top_k}
    
    try:
        start_time = datetime.datetime.now(datetime.timezone.utc)
        async with session.post(url, headers=headers, json=body, ssl=False, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            latency = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
            
            if resp.status == 200:
                data = await resp.json()
                results = [doc.get("chunk", "") for doc in data.get("value", []) if doc.get("chunk")]
                logger.info(f"Search on {index_name}: {len(results)} results in {latency:.2f}s")
                return results
            else:
                error_text = await resp.text()
                logger.error(f"Search failed for {index_name} ({resp.status}): {error_text}")
                return []
    except asyncio.TimeoutError:
        logger.error(f"Search timeout for {index_name}")
        return []
    except Exception as e:
        logger.error(f"Search error for {index_name}: {e}", exc_info=True)
        return []

async def analyze_and_route(user_query: str, last_turns: List[Dict]) -> Dict[str, Any]:
    """
    COMBINED ROUTER: Detects intent AND rewrites query in ONE call.
    Returns JSON: { "is_follow_up": bool, "rewritten_query": str }
    """
    prev_context = ""
    if last_turns:
        # Handle both old format (user_query/bot_response) and new format (content_user/content_assistant)
        last_turn = last_turns[-1]
        prev_q = last_turn.get('content_user') or last_turn.get('user_query', '')
        prev_a = last_turn.get('content_assistant') or last_turn.get('bot_response', '')
        
        prev_context = f"""
        PREVIOUS Q: {prev_q}
        PREVIOUS A: {prev_a}
        """

    prompt = f"""
    You are a Router. Analyze the New Query based on the Previous Context (if any).
    
    Tasks:
    1. Boolean 'is_follow_up': True if user refers to previous context. False if it's a new independent topic.
    2. String 'rewritten_query': Rewrite the query to be standalone and keyword-rich for search. If it is a follow-up, resolve pronouns (e.g., change "it" to "Tax Slabs").
    
    Context: {prev_context}
    New Query: {user_query}
    
    Return JSON ONLY: {{ "is_follow_up": bool, "rewritten_query": "string" }}
    """
    
    try:
        start_time = datetime.datetime.now(datetime.timezone.utc)
        response = await aoai_client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200
        )
        latency = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
        
        result = json.loads(response.choices[0].message.content)
        logger.info(f"Router completed in {latency:.2f}s: is_follow_up={result.get('is_follow_up')}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Router JSON decode error: {e}")
        return {"is_follow_up": False, "rewritten_query": user_query}
    except Exception as e:
        logger.error(f"Router error: {e}", exc_info=True)
        return {"is_follow_up": False, "rewritten_query": user_query}

async def combine_and_rewrite_query(current_query: str, previous_combined_query: Optional[str]) -> str:
    """
    Combine current follow-up query with previous combined query and rewrite.
    
    Example:
    - Previous combined: "What are company tax rates in India?"
    - Current query: "Does this include cess?"
    - Output: "Do company tax rates in India include cess component?"
    """
    try:
        if previous_combined_query:
            combination_context = f"Previous query context: {previous_combined_query}"
        else:
            combination_context = "No previous query context."
        
        prompt = f"""
        You are a Query Rewriter. Your task is to create a comprehensive standalone query.
        
        {combination_context}
        New follow-up question: {current_query}
        
        Task: Combine the previous query context with the new follow-up question into ONE comprehensive, 
        standalone, search-optimized query that captures the complete user intent.
        
        Rules:
        1. Resolve all pronouns ("it", "this", "that") to their actual referents
        2. Make the query self-contained and keyword-rich
        3. Preserve the original intent while adding context
        4. Return ONLY the rewritten query as plain text
        
        """
        
        start_time = datetime.datetime.now(datetime.timezone.utc)
        response = await aoai_client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a query rewriting expert. Output only the rewritten query."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )
        latency = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
        
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"Query combination completed in {latency:.2f}s: {rewritten}")
        return rewritten
        
    except Exception as e:
        logger.error(f"Query combination error: {e}", exc_info=True)
        # Fallback: combine manually
        if previous_combined_query:
            return f"{previous_combined_query} {current_query}"
        return current_query

async def select_relevant_indexes(query: str) -> List[str]:
    """
    Use LLM to intelligently select which indexes to search based on query content.
    
    Available indexes:
    - taxation-acts: Income Tax Act sections, provisions
    - taxation-rules: Income Tax Rules, procedural guidelines
    - taxation-circular-notification: Circulars and notifications
    
    Returns list of index names to search.
    """
    try:
        prompt = f"""
        You are an Index Selection Expert for a taxation knowledge base.
        
        Available indexes:
        1. "taxation-acts" - Contains Income Tax Act sections, provisions, deductions, exemptions
        2. "taxation-rules" - Contains Income Tax Rules, procedural guidelines, compliance requirements
        3. "taxation-circular-notification" - Contains circulars and notifications
        
        Task: Analyze the query and determine which index(es) would be most relevant to search.
        
        Query: {query}
        
        Rules:
        - If query mentions specific sections (e.g., Section 80C, Section 115BA), include "taxation-acts"
        - If query is about procedures, filing, compliance, include "taxation-rules"
        - If query is about recent changes, notifications, clarifications, include "taxation-circular-notification"
        - You can select multiple indexes if query spans multiple areas
        - Always select at least one index
        - Return ONLY a JSON array of index names
        
        Return JSON format: {{"indexes": ["index1", "index2"]}}
        """
        
        start_time = datetime.datetime.now(datetime.timezone.utc)
        response = await aoai_client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are an index selection expert. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=100
        )
        latency = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
        
        result = json.loads(response.choices[0].message.content)
        selected_indexes = result.get("indexes", [])
        
        # Validate indexes
        valid_indexes = []
        for idx in selected_indexes:
            if idx in Config.SEARCH_INDEXES:
                valid_indexes.append(idx)
        
        # If no valid indexes, search all as fallback
        if not valid_indexes:
            logger.warning(f"No valid indexes selected, using all indexes as fallback")
            valid_indexes = Config.SEARCH_INDEXES
        
        logger.info(f"Index selection completed in {latency:.2f}s: {valid_indexes}")
        return valid_indexes
        
    except Exception as e:
        logger.error(f"Index selection error: {e}", exc_info=True)
        # Fallback: search all indexes
        return Config.SEARCH_INDEXES

# ---------------------------------------------------------
# 5. MAIN ASYNC ORCHESTRATOR
# ---------------------------------------------------------
async def process_message(user_query: str, user_id: str) -> Dict[str, Any]:
    """Main entry point for processing a user message. Returns dict with 'answer' and 'confidence_score'."""
    start_time = datetime.datetime.now(datetime.timezone.utc)
    
    # Setup log capture for this request
    log_capture = LogCaptureHandler()
    log_capture.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(log_capture)
    
    try:
        # Validate inputs
        if not user_query or not user_query.strip():
            db_manager = AsyncHistoryManager(user_id)
            error_msg = "Please provide a valid question."
            captured_logs = log_capture.get_logs()
            await db_manager.save_turn(user_query or "", error_msg, False, 0.0, None, "Failure", captured_logs)
            return {"answer": error_msg, "status": "Failure", "confidence_score": 0.0, "context": ""}
        
        db_manager = AsyncHistoryManager(user_id)
        
        # 1. Fetch History
        history_turns = await db_manager.get_recent_history(limit=2)
        
        # 2. Route & Rewrite
        route_data = await analyze_and_route(user_query, history_turns)
        is_follow_up = route_data.get("is_follow_up", False)
        
        logger.info(f"Processing: user={user_id}, follow_up={is_follow_up}")

        # 3. Handle Follow-up vs New Query
        if is_follow_up:
            # For follow-ups: Use conversation history as context instead of searching
            logger.info(f"Follow-up detected. Using conversation history as context.")
            
            # Format conversation history as context
            if history_turns:
                context_parts = []
                for i, turn in enumerate(history_turns, 1):
                    # Handle both old format (user_query/bot_response) and new format (content_user/content_assistant)
                    user_q = turn.get('content_user') or turn.get('user_query', '')
                    bot_a = turn.get('content_assistant') or turn.get('bot_response', '')
                    context_parts.append(f"Previous Q{i}: {user_q}")
                    context_parts.append(f"Previous A{i}: {bot_a}")
                context_text = "\n\n".join(context_parts)
            else:
                context_text = "No previous conversation found."
            
            search_query = user_query  # Use original query for saving
        else:
            # New independent query - perform Azure AI Search
            search_query = route_data.get("rewritten_query", user_query)
            logger.info(f"New query. Search query: {search_query}")
            
            # 4. INTELLIGENT INDEX SELECTION
            selected_indexes = await select_relevant_indexes(search_query)
            logger.info(f"Selected indexes for search: {selected_indexes}")
            
            # 5. RAG RETRIEVAL (only on selected indexes)
            async with aiohttp.ClientSession() as session:
                tasks = [
                    search_azure_index(session, search_query, idx_name)
                    for idx_name in selected_indexes  # Only search selected indexes
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions from search tasks
            all_chunks = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Search task failed: {result}")
                elif isinstance(result, list):
                    all_chunks.extend(result)
            
            # Deduplicate and limit
            all_chunks = list(dict.fromkeys(all_chunks))[:5]
            
            context_text = "\n\n".join(all_chunks) if all_chunks else "No specific documents found."

        # 6. Build RAG Prompt (same for both follow-up and new queries)
        rag_prompt = f"""You are an expert assistant. Answer the user's question using:
1) Your general/domain knowledge as the primary source of reasoning, and
2) The provided context as supporting evidence when it is relevant, consistent, and useful.

Guidelines:
- Start from your domain knowledge to form a complete, helpful answer.
- Use the context to (a) confirm details, (b) add specifics, or (c) quote/ground claims ONLY when it genuinely supports them.
- If the context is missing, vague, low-quality, or irrelevant, do NOT force-fit it. Treat it as optional.

Output requirements:
- Write a clear, structured response with headings and bullet points where helpful.
- Return ONLY valid JSON in the following schema:
{{
  "answer": "detailed answer text",
  "confidence_score": <number between 0.0 and 1.0>,
  "status": "Success" or "Failure"
}}

Status Guidelines:
- "Success": You can provide a complete, accurate answer (with or without context)
- "Partial": You can provide some information but answer is incomplete or uncertain
- "No_Information": Cannot provide reliable information on this specific query

Context:
{context_text}

Question:
{user_query}
"""
        
        messages = [
            {"role": "system", "content": "You are a Taxation Expert. Always respond in JSON format with 'answer', 'confidence_score', and 'status' fields."},
            {"role": "user", "content": rag_prompt}
        ]

        # 7. Generate Answer
        try:
            response = await aoai_client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                parsed_response = json.loads(response_content)
                final_answer = parsed_response.get("answer", response_content)
                confidence_score = parsed_response.get("confidence_score", 0.5)
                llm_status = parsed_response.get("status", "Success")
                
                # Validate confidence score
                if not isinstance(confidence_score, (int, float)) or not (0.0 <= confidence_score <= 1.0):
                    logger.warning(f"Invalid confidence score: {confidence_score}, defaulting to 0.5")
                    confidence_score = 0.5
                
                # Validate status
                if llm_status not in ["Success", "Partial", "No_Information"]:
                    logger.warning(f"Invalid status from LLM: {llm_status}, defaulting to 'Success'")
                    llm_status = "Success"
                    
            except json.JSONDecodeError as je:
                logger.warning(f"Failed to parse JSON response: {je}. Using raw response.")
                final_answer = response_content
                confidence_score = 0.5
                llm_status = "Success"
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            error_msg = "I apologize, but I'm having trouble generating a response. Please try again."
            captured_logs = log_capture.get_logs()
            await db_manager.save_turn(user_query, error_msg, is_follow_up, 0.0, search_query, "Failure", captured_logs)
            return {
                "answer": error_msg,
                "status": "Failure",
                "confidence_score": 0.0
            }

        # 8. Save Turn (save with the combined query used for this turn)
        captured_logs = log_capture.get_logs()
        await db_manager.save_turn(user_query, final_answer, is_follow_up, confidence_score, search_query, llm_status, captured_logs)
        
        total_latency = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()
        logger.info(f"Request completed in {total_latency:.2f}s for user={user_id}, status={llm_status}, confidence={confidence_score}, follow_up={is_follow_up}")
        
        return {"answer": final_answer, "status": llm_status, "confidence_score": confidence_score, "context": context_text}
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        try:
            db_manager = AsyncHistoryManager(user_id)
            error_msg = "An error occurred while processing your request. Please try again."
            captured_logs = log_capture.get_logs()
            await db_manager.save_turn(user_query, error_msg, False, 0.0, None, "Failure", captured_logs)
        except:
            pass  # Don't fail if DB save fails
        return {
            "answer": "An error occurred while processing your request. Please try again.",
            "status": "Failure",
            "confidence_score": 0.0,
            "context": ""
        }
    finally:
        # Remove log capture handler after request completes
        logger.removeHandler(log_capture)

# ---------------------------------------------------------
# 6. ENTRY POINT
# ---------------------------------------------------------
async def main(test_email, test_question):
    """Main function for testing the bot."""
    try:
        # Initialize clients
        init_clients()
        await init_db()
        logger.info("Taxation Bot initialized successfully")
        
        print("\n" + "="*80)
        print("TAXATION BOT - TEST RUN")
        print("="*80)
        print(f"\nEmail: {test_email}")
        print(f"Question: {test_question}")
        
        # Process message
        result = await process_message(test_question, test_email)
        
        # Display results
        print("\n" + "-"*80)
        print("RETRIEVED CONTEXT:")
        print("-"*80)
        if result.get('context'):
            print(result['context'])
        else:
            print("No context retrieved")
        
        print("\n" + "-"*80)
        print("RESPONSE:")
        print("-"*80)
        print(f"Status: {result['status']}")
        print(f"Confidence Score: {result['confidence_score']}")
        print(f"\nAnswer:\n{result['answer']}")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        await close_clients()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        test_email = "test4@example.com"
        test_question = "please explain me the rule related to tax deduction at source for salaried employees"
        asyncio.run(main(test_email, test_question))
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

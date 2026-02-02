# services/ai_engine.py
import json
import datetime
from typing import Any, Dict, List, Optional
import clients
from config import Config
from logger import logger

async def analyze_and_route(user_query: str, last_turns: List[Dict]) -> Dict[str, Any]:
    """
    COMBINED ROUTER: Detects intent AND rewrites query in ONE call.
    Returns JSON: { "is_follow_up": bool, "rewritten_query": str }
    """
    prev_context = ""
    if last_turns:
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
        response = await clients.aoai_client.chat.completions.create(
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
    """Combine current follow-up query with previous combined query and rewrite."""
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
        response = await clients.aoai_client.chat.completions.create(
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
        if previous_combined_query:
            return f"{previous_combined_query} {current_query}"
        return current_query

async def select_relevant_indexes(query: str) -> List[str]:
    """Use LLM to intelligently select which indexes to search."""
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
        response = await clients.aoai_client.chat.completions.create(
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
        
        valid_indexes = []
        for idx in selected_indexes:
            if idx in Config.SEARCH_INDEXES:
                valid_indexes.append(idx)
        
        if not valid_indexes:
            logger.warning(f"No valid indexes selected, using all indexes as fallback")
            valid_indexes = Config.SEARCH_INDEXES
        
        logger.info(f"Index selection completed in {latency:.2f}s: {valid_indexes}")
        return valid_indexes
        
    except Exception as e:
        logger.error(f"Index selection error: {e}", exc_info=True)
        return Config.SEARCH_INDEXES
# bot.py
import json
import logging
import datetime
import asyncio
import aiohttp
from typing import Dict, Any

from config import Config
from logger import logger, LogCaptureHandler
import clients
from database import AsyncHistoryManager
from services.ai_engine import analyze_and_route, select_relevant_indexes
from services.search_engine import search_azure_index

async def process_message(user_query: str, user_id: str) -> Dict[str, Any]:
    """Main entry point for processing a user message."""
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
            response = await clients.aoai_client.chat.completions.create(
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
# database.py
import datetime
from typing import List, Dict, Optional
import clients
from logger import logger

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
                "logs": logs or []
            }
            # Access collection from clients module
            result = await clients.collection.update_one(
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
            doc = await clients.collection.find_one(
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
            doc = await clients.collection.find_one(
                self.filter,
                {"turns": {"$slice": -1}, "_id": 0}
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
# services/search_engine.py
import aiohttp
import datetime
import asyncio
from typing import List
from config import Config
from logger import logger

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
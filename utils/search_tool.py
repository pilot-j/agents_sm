
from helpers.tool_pattern import tool
from fastapi import HTTPException
from pydantic import BaseModel
import httpx
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






class SearchResponse(BaseModel):
    query: str
    response: str

SONAR_API_KEY = "{your key}"

async def sonar_search(
        system_prompt: str, 
        search_query: str
    ):
    url = f"https://api.perplexity.ai/chat/completions"
    json_in = "Please output a JSON object containing the following fields: query:, response: "

    model = "sonar"
    context_size = "low"
    max_tokens= 500
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (search_query + json_in)}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "web_search_options": {"search_context_size": context_size},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": SearchResponse.model_json_schema()
            }
        }
    }

    payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {SONAR_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=3000.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"SONAR API error: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Request failed: {str(e)}"
            )


search_tool = tool(sonar_search)
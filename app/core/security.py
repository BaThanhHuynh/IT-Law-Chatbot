from fastapi import Header, HTTPException
from app.core.config import Config


async def verify_api_key(x_api_key: str = Header(None)):
    """
    Optional API key authentication via X-Api-Key header.
    Disabled entirely when API_KEY is not set in environment.
    """
    if not Config.API_KEY:
        return
    if x_api_key != Config.API_KEY:
        raise HTTPException(status_code=401, detail="API key không hợp lệ hoặc bị thiếu.")

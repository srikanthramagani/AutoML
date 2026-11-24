"""Optional API-key authentication and owner identity."""
from typing import Optional

from fastapi import Header, HTTPException

from ..config.settings import backend_settings
from .state_owner import set_owner_id


def get_owner_id(x_api_key: Optional[str] = Header(default=None)) -> str:
    configured_keys = {
        key.strip()
        for key in backend_settings.api_keys.split(",")
        if key.strip()
    }
    if not configured_keys:
        owner_id = backend_settings.default_owner_id
        set_owner_id(owner_id)
        return owner_id
    if x_api_key not in configured_keys:
        raise HTTPException(status_code=401, detail="A valid X-API-Key is required.")
    set_owner_id(x_api_key)
    return x_api_key

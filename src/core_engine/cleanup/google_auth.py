"""Google Photos OAuth (desktop/installed-app flow) + secure token storage.

Only the upload scope is requested — the local-first design never reads the
library via the API (which is no longer permitted anyway). Tokens are stored in
the macOS Keychain via `keyring`, falling back to an encrypted-permission local
JSON file. Heavy google-auth imports are lazy so the package imports without the
`[gphoto]` extra installed.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ..config import APP_DIR

# Upload / append-only is all a local-first cleanup needs; library-read scopes were
# removed by Google in 2025 and are intentionally not requested.
SCOPES = ["https://www.googleapis.com/auth/photoslibrary.appendonly"]

_KEYRING_SERVICE = "pixel-purge"
_KEYRING_USER = "google-photos-token"
_TOKEN_FALLBACK = APP_DIR / "token.json"


# ---- token persistence (isolated so it is unit-testable) --------------------
def save_token(token_json: str) -> None:
    try:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, token_json)
        return
    except Exception:  # noqa: BLE001 - no keyring backend; fall back to file
        pass
    _TOKEN_FALLBACK.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_FALLBACK.write_text(token_json)
    os.chmod(_TOKEN_FALLBACK, stat.S_IRUSR | stat.S_IWUSR)  # 0600


def load_token() -> str | None:
    try:
        import keyring

        tok = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
        if tok:
            return tok
    except Exception:  # noqa: BLE001
        pass
    if _TOKEN_FALLBACK.exists():
        return _TOKEN_FALLBACK.read_text()
    return None


# ---- flow -------------------------------------------------------------------
def get_credentials(client_secret_path: Path | None = None):
    """Return valid google credentials, refreshing or running the flow as needed."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    token_json = load_token()
    if token_json:
        creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        save_token(creds.to_json())
        return creds

    client_secret_path = client_secret_path or (APP_DIR / "client_secret.json")
    if not Path(client_secret_path).exists():
        raise FileNotFoundError(
            f"OAuth client secret not found at {client_secret_path}. "
            "Create a Desktop OAuth client in Google Cloud Console and save it there."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), SCOPES)
    creds = flow.run_local_server(port=0)
    save_token(creds.to_json())
    return creds


def build_photos_service(client_secret_path: Path | None = None):
    """Build an authenticated Google Photos Library API client."""
    from googleapiclient.discovery import build

    creds = get_credentials(client_secret_path)
    return build("photoslibrary", "v1", credentials=creds, static_discovery=False)

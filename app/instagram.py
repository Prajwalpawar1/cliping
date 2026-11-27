"""Instagram Reels automation and publishing module.

Supports direct login via Instagram ID / credentials (with 2FA support)
or session ID, and automatic vertical Reel publishing.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

from . import config

SESSION_FILE = config.DATA_DIR / "instagram_settings.json"
TOKEN_FILE = config.DATA_DIR / "instagram_session.json"


class NotConfigured(RuntimeError):
    pass


class AuthError(RuntimeError):
    pass


class TwoFactorRequiredError(AuthError):
    def __init__(self, message: str = "Two-factor authentication code required.", two_factor_info: dict | None = None):
        super().__init__(message)
        self.two_factor_info = two_factor_info or {}


class UploadError(RuntimeError):
    pass


def sanitize_sessionid(raw: str) -> str:
    """Cleans and extracts the raw sessionid cookie value from user input.

    Handles full cookie headers (e.g. 'sessionid=XYZ; ds_user_id=123'),
    leading 'sessionid=' or 'sessionid:', surrounding quotes, and whitespace.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    # If pasted as cookie string with semicolons
    if ";" in s:
        for part in s.split(";"):
            part = part.strip()
            if re.match(r"^sessionid\s*=", part, flags=re.IGNORECASE):
                s = re.sub(r"^sessionid\s*=", "", part, flags=re.IGNORECASE)
                break
    # Remove leading "sessionid=" or "sessionid: "
    s = re.sub(r"^sessionid\s*[=:]\s*", "", s, flags=re.IGNORECASE)
    # Strip quotes and whitespace
    s = s.strip("\"' \t\r\n")
    return s


def _clean_caption(text: str) -> str:
    return (text or "").strip()


def build_caption(
    caption: str,
    hashtags: list[str] | None = None,
    max_len: int = 2200,
) -> str:
    """Combines caption and hashtags for Instagram Reels within Instagram's 2200-character limit."""
    caption = _clean_caption(caption)
    tags: list[str] = []
    for tag in hashtags or []:
        tag = tag.strip().lstrip("#")
        if tag and tag not in tags:
            tags.append(tag)
    # Instagram allows up to 30 hashtags per post
    tags = tags[:30]
    tags_text = " ".join(f"#{t}" for t in tags)
    if tags_text:
        combined = f"{caption}\n\n{tags_text}" if caption else tags_text
    else:
        combined = caption
    if len(combined) > max_len:
        combined = combined[:max_len].rstrip()
    return combined


def _read_session() -> dict | None:
    try:
        if TOKEN_FILE.exists():
            return json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _save_session(data: dict) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(TOKEN_FILE)


def status() -> dict:
    sess = _read_session()
    if not sess or not sess.get("username"):
        return {
            "configured": True,
            "connected": False,
            "account": None,
        }
    return {
        "configured": True,
        "connected": True,
        "account": {
            "username": sess.get("username"),
            "full_name": sess.get("full_name") or sess.get("username"),
            "user_id": sess.get("user_id"),
            "profile_pic_url": sess.get("profile_pic_url"),
            "connected_at": sess.get("connected_at"),
            "method": sess.get("method", "credentials"),
        },
    }


def disconnect() -> None:
    try:
        TOKEN_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        SESSION_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def get_client(refresh_if_needed: bool = True):
    from instagrapi import Client

    cl = Client()
    if SESSION_FILE.exists():
        try:
            cl.load_settings(SESSION_FILE)
        except Exception:
            pass

    sess = _read_session()
    if not sess:
        raise AuthError("Instagram is not connected. Please log in first.")

    sessionid = sess.get("sessionid")
    if sessionid:
        try:
            cl.login_by_sessionid(sessionid)
            return cl
        except Exception as exc:
            raise AuthError(f"Instagram session expired: {exc}") from exc

    if cl.user_id:
        return cl

    raise AuthError("Instagram session expired or not authenticated. Please log in again in Connections.")


def login_with_credentials(
    username: str,
    password: str,
    verification_code: str | None = None,
) -> dict:
    """Log in to Instagram using Username & Password, with optional 2FA verification code."""
    from instagrapi import Client
    from instagrapi.exceptions import (
        BadPassword,
        ChallengeRequired,
        ClientNeedsUpgradeError,
        FeedbackRequired,
        PleaseWaitFewMinutes,
        TwoFactorRequired,
    )

    username = username.strip().lstrip("@")
    if not username:
        raise ValueError("Instagram username cannot be empty.")
    if not password:
        raise ValueError("Instagram password cannot be empty.")

    cl = Client()
    if SESSION_FILE.exists():
        try:
            cl.load_settings(SESSION_FILE)
        except Exception:
            pass

    try:
        cl.login(username, password, verification_code=verification_code or "")
    except TwoFactorRequired as exc:
        raise TwoFactorRequiredError(
            "Two-Factor Authentication is enabled on this account. Enter the 6-digit verification code.",
            two_factor_info={"username": username},
        ) from exc
    except BadPassword as exc:
        raise AuthError("Incorrect Instagram username or password.") from exc
    except ClientNeedsUpgradeError as exc:
        raise AuthError(
            "Instagram rejected mobile app login with 'Version out of date'. "
            "Instagram blocks automated password logins on modern accounts. "
            "Please switch to the 'Session ID Cookie' tab above to connect instantly in 1-click using your browser session cookie."
        ) from exc
    except ChallengeRequired as exc:
        raise AuthError(
            "Instagram requested a checkpoint verification. Please approve 'This Was Me' on your Instagram mobile app, or use Session ID login."
        ) from exc
    except PleaseWaitFewMinutes as exc:
        raise AuthError("Instagram is temporarily rate-limiting logins. Please wait a few minutes or switch to Session ID login.") from exc
    except FeedbackRequired as exc:
        raise AuthError(f"Instagram security feedback required: {exc}. Please use Session ID login to connect directly.") from exc
    except Exception as exc:
        err_msg = str(exc)
        if any(w in err_msg.lower() for w in ["out of date", "upgrade your app", "needs_upgrade", "caa login"]):
            raise AuthError(
                "Instagram rejected mobile app login with 'Version out of date'. "
                "Instagram blocks automated password logins on modern accounts. "
                "Please switch to the 'Session ID Cookie' tab above to connect instantly in 1-click using your browser session cookie."
            ) from exc
        raise AuthError(f"Instagram login failed: {err_msg}") from exc

    # Persist client settings
    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        cl.dump_settings(SESSION_FILE)
    except Exception:
        pass

    full_name = username
    profile_pic_url = None
    user_id = str(cl.user_id) if cl.user_id else None
    try:
        user_info = cl.user_info_by_username(username)
        full_name = user_info.full_name or username
        profile_pic_url = str(user_info.profile_pic_url) if user_info.profile_pic_url else None
        user_id = str(user_info.pk)
    except Exception:
        pass

    session_data = {
        "username": username,
        "user_id": user_id,
        "full_name": full_name,
        "profile_pic_url": profile_pic_url,
        "connected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "credentials",
    }
    _save_session(session_data)
    return status()


def login_with_sessionid(sessionid: str) -> dict:
    """Log in to Instagram using a session ID cookie."""
    from instagrapi import Client

    clean_sid = sanitize_sessionid(sessionid)
    if not clean_sid:
        raise ValueError("Session ID cannot be empty.")

    cl = Client()
    try:
        cl.login_by_sessionid(clean_sid)
    except Exception as exc:
        raise AuthError(
            f"Invalid or expired Instagram session ID: {exc}. "
            "Please ensure you are logged into instagram.com and copied the valid 'sessionid' cookie value."
        ) from exc

    try:
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        cl.dump_settings(SESSION_FILE)
    except Exception:
        pass

    username = "instagram_creator"
    full_name = "Instagram Creator"
    profile_pic_url = None
    user_id = str(cl.user_id) if cl.user_id else None
    try:
        account_data = cl.account_info()
        username = account_data.username or username
        full_name = account_data.full_name or username
        profile_pic_url = str(account_data.profile_pic_url) if account_data.profile_pic_url else None
        user_id = str(account_data.pk)
    except Exception:
        pass

    session_data = {
        "username": username,
        "user_id": user_id,
        "full_name": full_name,
        "profile_pic_url": profile_pic_url,
        "sessionid": clean_sid,
        "connected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "sessionid",
    }
    _save_session(session_data)
    return status()


def upload_reel(
    path: Path,
    caption: str,
    hashtags: list[str] | None = None,
    thumbnail_path: Path | None = None,
    progress: Callable[[int], None] | None = None,
    client=None,
) -> dict:
    """Uploads a video to Instagram as a vertical 9:16 Reel."""
    if not path.exists():
        raise UploadError(f"Video file not found at {path}")

    cl = client or get_client()
    full_caption = build_caption(caption, hashtags)

    if progress:
        progress(15)

    try:
        if progress:
            progress(40)

        thumb = thumbnail_path if (thumbnail_path and thumbnail_path.exists()) else None
        media = cl.clip_upload(
            path=path,
            caption=full_caption,
            thumbnail=thumb,
        )

        if progress:
            progress(95)
    except Exception as exc:
        err_msg = str(exc)
        if "login_required" in err_msg.lower():
            raise AuthError("Instagram session expired. Please log in again in Connections.") from exc
        raise UploadError(f"Instagram upload failed: {err_msg}") from exc

    if progress:
        progress(100)

    code = getattr(media, "code", "") or getattr(media, "pk", "")
    media_id = str(getattr(media, "pk", "") or getattr(media, "id", ""))
    url = f"https://www.instagram.com/reel/{code}/" if code else f"https://www.instagram.com/p/{media_id}/"

    return {
        "status": "done",
        "media_id": media_id,
        "code": str(code),
        "url": url,
    }

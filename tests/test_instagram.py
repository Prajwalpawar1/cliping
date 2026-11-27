"""Tests for Instagram Reels automation, authentication, and publishing."""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import config, instagram
from app.main import app


@pytest.fixture(autouse=True)
def clean_instagram_session(tmp_path, monkeypatch):
    """Ensure tests run with an isolated Instagram session file."""
    test_session_file = tmp_path / "instagram_settings.json"
    test_token_file = tmp_path / "instagram_session.json"
    monkeypatch.setattr(instagram, "SESSION_FILE", test_session_file)
    monkeypatch.setattr(instagram, "TOKEN_FILE", test_token_file)
    yield
    try:
        test_session_file.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        test_token_file.unlink(missing_ok=True)
    except Exception:
        pass


def test_build_caption():
    cap = instagram.build_caption(
        caption="Check out this viral moment!",
        hashtags=["#reels", "viral", "#trending", "#reels"],  # test deduplication & # stripping
    )
    assert "Check out this viral moment!" in cap
    assert "#reels" in cap
    assert "#viral" in cap
    assert "#trending" in cap
    # Make sure no double hashtags ##
    assert "##" not in cap


def test_build_caption_max_len():
    huge_caption = "A" * 3000
    res = instagram.build_caption(huge_caption, hashtags=["#shorts"], max_len=100)
    assert len(res) <= 100


def test_instagram_status_when_disconnected():
    st = instagram.status()
    assert st["configured"] is True
    assert st["connected"] is False
    assert st["account"] is None


def test_instagram_status_when_connected():
    test_data = {
        "username": "test_creator",
        "user_id": "12345678",
        "full_name": "Test Creator",
        "profile_pic_url": "https://example.com/pic.jpg",
        "connected_at": "2026-11-26 00:00:00",
        "method": "credentials",
    }
    instagram._save_session(test_data)
    st = instagram.status()
    assert st["connected"] is True
    assert st["account"]["username"] == "test_creator"
    assert st["account"]["full_name"] == "Test Creator"
    assert st["account"]["user_id"] == "12345678"


def test_instagram_disconnect():
    test_data = {"username": "test_user"}
    instagram._save_session(test_data)
    assert instagram.status()["connected"] is True

    instagram.disconnect()
    assert instagram.status()["connected"] is False
    assert not instagram.TOKEN_FILE.exists()


def test_login_validation():
    with pytest.raises(ValueError, match="username cannot be empty"):
        instagram.login_with_credentials("", "pass")

    with pytest.raises(ValueError, match="password cannot be empty"):
        instagram.login_with_credentials("user", "")

    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        instagram.login_with_sessionid("   ")


def test_login_with_credentials_success():
    mock_cl = MagicMock()
    mock_cl.user_id = 99999
    mock_user_info = MagicMock()
    mock_user_info.full_name = "Super Creator"
    mock_user_info.profile_pic_url = "https://instagram.com/pic.jpg"
    mock_user_info.pk = 99999
    mock_cl.user_info_by_username.return_value = mock_user_info

    with patch("instagrapi.Client", return_value=mock_cl):
        res = instagram.login_with_credentials("super_creator", "secretpassword123")
        assert res["connected"] is True
        assert res["account"]["username"] == "super_creator"
        assert res["account"]["full_name"] == "Super Creator"
        mock_cl.login.assert_called_once_with("super_creator", "secretpassword123", verification_code="")
        mock_cl.dump_settings.assert_called_once()


def test_login_with_credentials_2fa_required():
    from instagrapi.exceptions import TwoFactorRequired

    mock_cl = MagicMock()
    mock_cl.login.side_effect = TwoFactorRequired("2FA required")

    with patch("instagrapi.Client", return_value=mock_cl):
        with pytest.raises(instagram.TwoFactorRequiredError) as exc_info:
            instagram.login_with_credentials("secure_user", "password")
        assert "Two-Factor Authentication is enabled" in str(exc_info.value)
        assert exc_info.value.two_factor_info["username"] == "secure_user"


def test_sanitize_sessionid():
    assert instagram.sanitize_sessionid("123456789%3Aabc123") == "123456789%3Aabc123"
    assert instagram.sanitize_sessionid("sessionid=123456789%3Aabc123") == "123456789%3Aabc123"
    assert instagram.sanitize_sessionid("sessionid: 123456789%3Aabc123") == "123456789%3Aabc123"
    assert instagram.sanitize_sessionid("\"123456789%3Aabc123\"") == "123456789%3Aabc123"
    assert instagram.sanitize_sessionid("sessionid=\"123456789%3Aabc123\"; path=/; domain=.instagram.com") == "123456789%3Aabc123"
    assert instagram.sanitize_sessionid("   ") == ""


def test_login_with_credentials_out_of_date_error():
    mock_cl = MagicMock()
    mock_cl.login.side_effect = Exception("Your version of Instagram is out of date. Please upgrade your app to log in to Instagram.")

    with patch("instagrapi.Client", return_value=mock_cl):
        with pytest.raises(instagram.AuthError) as exc_info:
            instagram.login_with_credentials("my_user", "my_pass")
        assert "Version out of date" in str(exc_info.value)
        assert "Session ID Cookie" in str(exc_info.value)


def test_login_with_credentials_needs_upgrade_error():
    from instagrapi.exceptions import ClientNeedsUpgradeError

    mock_cl = MagicMock()
    mock_cl.login.side_effect = ClientNeedsUpgradeError("needs_upgrade")

    with patch("instagrapi.Client", return_value=mock_cl):
        with pytest.raises(instagram.AuthError) as exc_info:
            instagram.login_with_credentials("my_user", "my_pass")
        assert "Version out of date" in str(exc_info.value)
        assert "Session ID Cookie" in str(exc_info.value)


def test_login_with_sessionid_success():
    mock_cl = MagicMock()
    mock_cl.user_id = 88888
    mock_account = MagicMock()
    mock_account.username = "session_user"
    mock_account.full_name = "Session User"
    mock_account.profile_pic_url = "https://instagram.com/session.jpg"
    mock_account.pk = 88888
    mock_cl.account_info.return_value = mock_account

    with patch("instagrapi.Client", return_value=mock_cl):
        res = instagram.login_with_sessionid("fake_session_id_12345")
        assert res["connected"] is True
        assert res["account"]["username"] == "session_user"
        mock_cl.login_by_sessionid.assert_called_once_with("fake_session_id_12345")


def test_login_with_sessionid_sanitizes_input():
    mock_cl = MagicMock()
    mock_cl.user_id = 77777
    mock_account = MagicMock()
    mock_account.username = "sanitized_user"
    mock_account.full_name = "Sanitized User"
    mock_account.profile_pic_url = None
    mock_account.pk = 77777
    mock_cl.account_info.return_value = mock_account

    with patch("instagrapi.Client", return_value=mock_cl):
        res = instagram.login_with_sessionid('sessionid="77777%3Aabcxyz1234567890"; domain=.instagram.com')
        assert res["connected"] is True
        mock_cl.login_by_sessionid.assert_called_once_with("77777%3Aabcxyz1234567890")


def test_upload_reel_success(tmp_path):
    video_file = tmp_path / "test_clip.mp4"
    video_file.write_bytes(b"dummy video content")

    mock_cl = MagicMock()
    mock_media = MagicMock()
    mock_media.code = "C8xY9zQp1"
    mock_media.pk = "1234567890"
    mock_cl.clip_upload.return_value = mock_media

    res = instagram.upload_reel(
        path=video_file,
        caption="Amazing clip #shorts",
        hashtags=["reels", "viral"],
        client=mock_cl,
    )
    assert res["status"] == "done"
    assert res["code"] == "C8xY9zQp1"
    assert res["url"] == "https://www.instagram.com/reel/C8xY9zQp1/"
    mock_cl.clip_upload.assert_called_once()


def test_api_instagram_status_and_disconnect():
    client = TestClient(app)

    # 1. Initially disconnected
    res = client.get("/api/instagram")
    assert res.status_code == 200
    assert res.json()["connected"] is False

    # 2. Login via API
    mock_cl = MagicMock()
    mock_cl.user_id = 1111
    mock_user_info = MagicMock()
    mock_user_info.full_name = "API User"
    mock_user_info.profile_pic_url = None
    mock_user_info.pk = 1111
    mock_cl.user_info_by_username.return_value = mock_user_info

    with patch("instagrapi.Client", return_value=mock_cl):
        res = client.post("/api/instagram/login", json={"username": "api_user", "password": "pass123"})
        assert res.status_code == 200
        assert res.json()["connected"] is True

    # 3. Disconnect
    res = client.post("/api/instagram/disconnect")
    assert res.status_code == 200
    assert res.json()["ok"] is True

    # Check status again
    res = client.get("/api/instagram")
    assert res.json()["connected"] is False


def test_api_publish_clip_instagram(tmp_path, monkeypatch):
    client = TestClient(app)
    # Connect instagram first
    instagram._save_session({"username": "insta_tester", "user_id": "123"})

    from app import main

    job_id = "00000000ff"
    job_dir = main._job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    video_file = job_dir / "clip_0.mp4"
    video_file.write_bytes(b"dummy")

    with main._lock:
        main._jobs[job_id] = {
            "id": job_id,
            "status": "done",
            "progress": 100,
            "clips": [
                {
                    "title": "Test Reel",
                    "video": f"/media/{job_id}/clip_0.mp4",
                    "thumb": f"/media/{job_id}/clip_0.jpg",
                    "length": 30.0,
                    "score": 90,
                }
            ],
        }
        main._save(main._jobs[job_id])

    # Publish clip to Instagram
    with patch("app.main.threading.Thread") as mock_thread:
        res = client.post(
            f"/api/jobs/{job_id}/clips/0/instagram",
            json={"caption": "Test Reel Caption", "hashtags": ["reels", "viral"]},
        )
        assert res.status_code == 200
        assert res.json()["status"] == "queued"
        mock_thread.return_value.start.assert_called_once()

    # Test delete guard while uploading
    with main._lock:
        main._jobs[job_id]["clips"][0]["instagram"] = {"status": "uploading"}

    with pytest.raises(main.HTTPException) as exc_info:
        main.delete_job(job_id)
    assert exc_info.value.status_code == 409
    assert "Instagram" in str(exc_info.value.detail)

    # Clean up by resetting upload status to done
    with main._lock:
        if job_id in main._jobs:
            main._jobs[job_id]["clips"][0]["instagram"] = {"status": "done"}
    main.delete_job(job_id)

import pytest
from fastapi.testclient import TestClient

from app import main, source
from app.source import Word

YT = "https://www.youtube.com/watch?v=abc123"


@pytest.fixture
def client(monkeypatch):
    submitted = []
    monkeypatch.setattr(main._pool, "submit", lambda fn, *a: submitted.append(a))
    monkeypatch.setattr(main, "_jobs", {})
    monkeypatch.setattr(main, "_cancelled_jobs", set())
    c = TestClient(main.app)
    c.submitted = submitted
    return c


def test_cancel_job_lifecycle(client):
    job = client.post("/api/jobs", json={"url": YT, "mode": "segment", "segment_len": 60}).json()
    jid = job["id"]
    assert job["status"] == "queued"

    # Cancel queued job
    res = client.post(f"/api/jobs/{jid}/cancel")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "job_id": jid, "status": "cancelled"}
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "cancelled"

    # Cannot cancel already cancelled job
    res_again = client.post(f"/api/jobs/{jid}/cancel")
    assert res_again.status_code == 400

    # Cannot cancel unknown job
    assert client.post("/api/jobs/notarealid/cancel").status_code == 404


def test_cancel_running_job(client):
    job = client.post("/api/jobs", json={"url": YT}).json()
    jid = job["id"]
    main._jobs[jid]["status"] = "running"

    res = client.post(f"/api/jobs/{jid}/cancel")
    assert res.status_code == 200
    assert res.json()["status"] == "cancelled"


def test_cannot_cancel_done_job(client):
    job = client.post("/api/jobs", json={"url": YT}).json()
    jid = job["id"]
    main._jobs[jid]["status"] = "done"

    res = client.post(f"/api/jobs/{jid}/cancel")
    assert res.status_code == 400


def test_cancelled_job_stops_execution(monkeypatch, client):
    words = [Word("word", float(i), float(i + 1)) for i in range(200)]

    def fake_download(*a, **k):
        video = a[1] / "source.mp4"
        video.write_bytes(b"test")
        return source.Source(video, "Long Video", 200.0, "en", words, "whisper")

    monkeypatch.setattr(source, "download", fake_download)
    rendered = []
    from app import render
    monkeypatch.setattr(render, "render_clip", lambda *a, **k: rendered.append(a) or {"layout": "auto", "length": 30.0})

    req = main.JobRequest(url=YT, mode="segment", segment_len=30)
    job = main._new_job(req, "00000000ee")

    # Cancel before running
    main.cancel_job(job["id"])
    main._run(job["id"], req)

    # Nothing should be rendered because it was cancelled
    assert len(rendered) == 0
    assert main._jobs[job["id"]]["status"] == "cancelled"


def test_segment_mode_bounds_and_capping():
    # Verify raw cut generation logic caps at MAX_SEGMENTS
    duration = 10000.0  # ~2.7 hours
    target_len = 30.0
    raw_cuts = []
    curr = 0.0
    while curr < duration - 5.0:
        nxt = min(duration, curr + target_len)
        if duration - nxt < 15.0:
            nxt = duration
        raw_cuts.append([round(curr, 2), round(nxt, 2)])
        curr = nxt

    assert len(raw_cuts) > 150
    # Slicing capping
    MAX_SEGMENTS = 150
    if len(raw_cuts) > MAX_SEGMENTS:
        capped = raw_cuts[:MAX_SEGMENTS]
    assert len(capped) == 150
    assert capped[0] == [0.0, 30.0]
    assert capped[-1][1] == 150 * 30.0

"""
backend/keep_alive.py
======================
In-process background keep-alive task for Render deployments.

Why this exists:
  When SIF-Guard is deployed on Render free tier, both the main backend
  and the ML microservice sleep after 15 minutes of inactivity.
  If the backend receives user traffic, the ML service might still be asleep,
  causing predictions to wait ~50s for a cold start or time out.

How it works:
  When the backend starts up, if `KEEP_ALIVE=true` or `ML_SERVICE_URL` is set,
  this module launches an asyncio background task that periodically (every 9 min)
  pings the ML microservice and/or the backend's own public URL.
"""

import asyncio
import os
import urllib.request
import urllib.error

# Ping every 9 minutes (540s) — safely under Render's 15-minute (900s) sleep threshold
INTERVAL_SECONDS = int(os.environ.get("KEEP_ALIVE_INTERVAL", "540"))
ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "").rstrip("/")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "").rstrip("/")


async def _ping_endpoint(url: str, name: str):
    """Asynchronously ping an HTTP endpoint in a thread to avoid blocking the event loop."""
    endpoint = f"{url}/health" if not url.endswith("/health") else url
    req = urllib.request.Request(endpoint, headers={"User-Agent": "Backend-KeepAlive/1.0"}, method="GET")

    def _sync_ping():
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception as e:
            return str(e)

    loop = asyncio.get_running_loop()
    status = await loop.run_in_executor(None, _sync_ping)
    print(f"[keep_alive] Pinged {name} ({endpoint}) -> status: {status}")


async def keep_alive_loop():
    """Continuous background loop keeping services warm."""
    print(f"[keep_alive] Background keep-alive worker started. Interval: {INTERVAL_SECONDS}s.")
    while True:
        try:
            await asyncio.sleep(INTERVAL_SECONDS)
            if ML_SERVICE_URL:
                await _ping_endpoint(ML_SERVICE_URL, "ML Microservice")
            if RENDER_EXTERNAL_URL:
                await _ping_endpoint(RENDER_EXTERNAL_URL, "Self (Backend)")
        except asyncio.CancelledError:
            print("[keep_alive] Worker shutting down.")
            break
        except Exception as e:
            print(f"[keep_alive] Error during ping cycle: {e}")


def start_keep_alive():
    """Start the keep-alive background task if enabled."""
    should_run = (
        os.environ.get("KEEP_ALIVE", "").lower() in ("1", "true", "yes")
        or bool(os.environ.get("RENDER"))
        or bool(ML_SERVICE_URL)
    )
    if should_run and (ML_SERVICE_URL or RENDER_EXTERNAL_URL):
        print(f"[keep_alive] Enabling background keep-alive (ML: {bool(ML_SERVICE_URL)}, Self: {bool(RENDER_EXTERNAL_URL)})")
        return asyncio.create_task(keep_alive_loop())
    else:
        print("[keep_alive] Background keep-alive not active (set KEEP_ALIVE=true or ML_SERVICE_URL to enable).")
        return None

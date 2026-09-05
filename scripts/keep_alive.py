#!/usr/bin/env python3
"""
scripts/keep_alive.py
======================
Keep-Alive script for Render Free Tier deployments.

Problem:
  Render's free tier spins down web services after 15 minutes of inactivity.
  When an inactive service is called, it triggers a 50-second "cold start"
  delay or request timeout.

Solution:
  This script periodically pings the /health endpoints of both the Backend API
  and the ML microservice every 10 minutes (configurable), keeping them awake
  and responsive 24/7 so users never experience the 50-second lag.

Usage:
  # Continuous daemon (pings every 10 minutes):
  python scripts/keep_alive.py

  # Ping once and report status + response times:
  python scripts/keep_alive.py --once

  # Custom URLs and custom interval (e.g., every 8 minutes):
  python scripts/keep_alive.py --backend-url https://my-backend.onrender.com --ml-url https://my-ml.onrender.com --interval 480
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_BACKEND_URL = os.environ.get("RENDER_BACKEND_URL", "https://avertx.onrender.com").rstrip("/")
DEFAULT_ML_URL = os.environ.get("RENDER_ML_URL", "https://avertx-ml.onrender.com").rstrip("/")
DEFAULT_INTERVAL_SECONDS = int(os.environ.get("KEEP_ALIVE_INTERVAL", "600"))  # 10 minutes (Render sleeps at 15 min)
TIMEOUT_SECONDS = 75  # Allow up to 75s in case a cold-start is in progress


def ping_url(name: str, url: str) -> dict:
    """Send a GET request to the given endpoint and return status info."""
    endpoint = f"{url}/health" if not url.endswith("/health") else url
    t0 = time.time()
    result = {
        "name": name,
        "url": endpoint,
        "status_code": None,
        "elapsed_seconds": 0.0,
        "is_cold_start": False,
        "ok": False,
        "error": None,
    }

    req = urllib.request.Request(
        endpoint,
        headers={"User-Agent": "SIF-Guard-KeepAlive/1.0"},
        method="GET"
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            elapsed = time.time() - t0
            result["status_code"] = resp.status
            result["elapsed_seconds"] = round(elapsed, 2)
            result["ok"] = 200 <= resp.status < 400
            result["is_cold_start"] = elapsed > 25.0
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        result["status_code"] = e.code
        result["elapsed_seconds"] = round(elapsed, 2)
        result["ok"] = False
        result["error"] = f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 2)
        result["ok"] = False
        result["error"] = str(e.reason)
    except Exception as e:
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 2)
        result["ok"] = False
        result["error"] = str(e)

    return result


def format_log(res: dict) -> str:
    """Format the ping result into a clear colored log line."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    name = res["name"].ljust(12)
    elapsed = f"{res['elapsed_seconds']}s".rjust(7)

    if res["ok"]:
        flag = "[COLD START WAKENED]" if res["is_cold_start"] else "[WARM / READY]"
        return f"[{ts}] {name} -> Status: {res['status_code']} in {elapsed} {flag}"
    else:
        return f"[{ts}] {name} -> FAILED after {elapsed}: {res['error']}"


def run_cycle(backend_url: str, ml_url: str):
    """Run one ping cycle against both services."""
    print("\n" + "-" * 70)
    print(f"Keep-alive check at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("-" * 70)

    if backend_url:
        res_backend = ping_url("Backend", backend_url)
        print(" " + format_log(res_backend))

    if ml_url:
        res_ml = ping_url("ML Service", ml_url)
        print(" " + format_log(res_ml))


def main():
    parser = argparse.ArgumentParser(description="Render Free Tier Keep-Alive Daemon")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Base URL of backend service")
    parser.add_argument("--ml-url", default=DEFAULT_ML_URL, help="Base URL of ML microservice")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                        help="Interval in seconds between pings (default: 600s / 10m)")
    parser.add_argument("--once", action="store_true", help="Ping once and exit immediately")

    args = parser.parse_args()

    print("======================================================================")
    print("   SIF-Guard Render Free Tier Keep-Alive Monitor")
    print("======================================================================")
    print(f" Backend URL : {args.backend_url}")
    print(f" ML URL      : {args.ml_url}")
    print(f" Ping Rate   : Every {args.interval} seconds ({args.interval / 60:.1f} minutes)")
    print(" Note: Render free tier sleeps after 15 min of inactivity.")
    print("       Pinging every 10 min guarantees service stays permanently awake.")
    print("======================================================================\n")

    if args.once:
        run_cycle(args.backend_url, args.ml_url)
        print("\nSingle check complete.")
        return

    try:
        while True:
            run_cycle(args.backend_url, args.ml_url)
            print(f"\nNext keep-alive ping in {args.interval}s... (Press Ctrl+C to stop)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nKeep-alive monitor stopped by user.")


if __name__ == "__main__":
    main()

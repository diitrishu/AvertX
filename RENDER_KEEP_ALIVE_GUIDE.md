# Render Free Tier 50-Second Cold-Start & Keep-Alive Solution Guide

---

## 1. Why Does the "50-Second Limit" Happen on Render Free Tier?

Render's Free Tier includes **750 free instance hours per month**. To conserve resources, Render automatically applies the following rule to all free web services:

1. **15-Minute Inactivity Sleep:** If your service does not receive any HTTP traffic for **15 minutes (900 seconds)**, Render spins it down to 0 instances (puts it to sleep).
2. **50-Second Cold Start:** When a new visitor opens your site or sends an API request, Render must spin up the container, load Python, initialize the server, and load the ML models. This takes **40–50 seconds**.
3. **Timeout / 502 Gateway Limit:** Because the previous backend HTTP client had a 30-second timeout, cold starts would hit the timeout limit and fail with `502 Bad Gateway` or `Network Error`.

---

## 2. Solutions Provided in This Codebase

We have added a 4-layer solution to completely eliminate this issue:

| Layer | File | How It Works |
|---|---|---|
| **1. Cloud Keep-Alive (24/7)** | `.github/workflows/render-keep-alive.yml` | **GitHub Actions** runs every 10 minutes for free in the cloud. It pings your `/health` endpoints so Render **never goes to sleep**. |
| **2. Standalone Python Pinger** | `scripts/keep_alive.py` | Run on your local computer or VPS to keep Render warm or verify whether services are sleeping or warm. |
| **3. In-App ML Pinger** | `backend/keep_alive.py` | Whenever the backend is awake, it automatically pings the ML microservice every 9 minutes so both stay awake together. |
| **4. Cold-Start Tolerance** | `backend/main.py` + `frontend/src/App.tsx` | Raised timeout to 75s with retry logic; frontend pings `/health` immediately on initial page load. |

---

## 3. How to Use Each Solution

### Option A: GitHub Actions (Recommended — 100% Automated, Zero Maintenance)

A GitHub Actions workflow is now placed in:
```
.github/workflows/render-keep-alive.yml
```

**How to activate:**
1. Commit and push your code to your GitHub repository:
   ```bash
   git add .github/workflows/render-keep-alive.yml
   git commit -m "Add Render free tier keep-alive workflow"
   git push origin main
   ```
2. Go to your repository on **GitHub** → **Actions** tab.
3. Ensure GitHub Actions are enabled.
4. (Optional) If your Render URLs differ from the defaults (`https://avertx.onrender.com` and `https://avertx-ml.onrender.com`), go to:
   - **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:
     - `RENDER_BACKEND_URL`: your backend URL
     - `RENDER_ML_URL`: your ML service URL
5. The workflow will now automatically run **every 10 minutes** via GitHub cron, keeping your services awake 24/7 with zero lag.

---

### Option B: Free External Monitoring Service (Cron-job.org / UptimeRobot)

If you prefer an external web monitor (free forever, takes 1 minute to configure):

1. Go to [https://cron-job.org](https://cron-job.org) or [https://uptimerobot.com](https://uptimerobot.com) (both 100% free).
2. Create a free account.
3. Add two monitors:
   - **Monitor 1 (Backend):**
     - URL: `https://<your-backend-app>.onrender.com/health`
     - Interval: **Every 10 minutes**
   - **Monitor 2 (ML Service):**
     - URL: `https://<your-ml-app>.onrender.com/health`
     - Interval: **Every 10 minutes**
4. That's it! They will send a `GET` request every 10 minutes. Because Render's sleep timer is 15 minutes, your services will never spin down.

---

### Option C: Run Local Python Script (`scripts/keep_alive.py`)

You can also run the keep-alive monitor directly from your command line:

```bash
# Continuous keep-alive daemon (pings every 10 minutes):
python scripts/keep_alive.py

# Or check status and response times once:
python scripts/keep_alive.py --once

# With custom URLs:
python scripts/keep_alive.py --backend-url https://your-backend.onrender.com --ml-url https://your-ml.onrender.com
```

**Example output:**
```text
[2026-09-05 04:15:20 UTC] Backend      -> Status: 200 in    0.34s [WARM / READY]
[2026-09-05 04:15:21 UTC] ML Service   -> Status: 200 in    0.41s [WARM / READY]
```

---

## 4. Code Optimizations Added

### 1. Extended Timeout with Retries (`backend/main.py`)
In `backend/main.py`, the remote ML call previously had `timeout=30`, which caused cold starts to fail. It has now been updated to:
- **75-second timeout** with a retry attempt:
```python
# Render free-tier containers take 45-50s to cold-start when waking from sleep.
# Allow up to 75s with a retry to prevent 502/timeout failures.
last_err = None
for attempt in range(2):
    try:
        with urllib.request.urlopen(req, timeout=75) as resp:
            result = json.loads(resp.read())
            break
    except Exception as e:
        last_err = e
```

### 2. Startup Keep-Alive Worker (`backend/keep_alive.py`)
FastAPI now launches a background asyncio worker on startup that automatically pings `ML_SERVICE_URL/health` every 9 minutes. Whenever the backend is handling users, the ML service stays warm.

### 3. Frontend Pre-Warm Ping (`frontend/src/App.tsx`)
On initial mount of the React app, a background non-blocking fetch to `/health` is fired immediately:
```typescript
useEffect(() => {
  fetch(`${API}/health`).catch(() => {});
}, []);
```
By the time the user finishes typing their incident narrative, the backend and model are already warm.

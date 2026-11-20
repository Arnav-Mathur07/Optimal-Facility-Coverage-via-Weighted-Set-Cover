# Production Deployment Guide
## Optimal Facility Coverage & Weighted Set Cover Dashboard

This guide provides step-by-step instructions to deploy your **DAA Project-Based Learning Dashboard** to the cloud for free with public HTTPS URLs.

---

## 📋 Git Inclusion vs. Exclusion Matrix

Before committing, verify that your repository strictly adheres to this matrix:

### ✅ MUST COMMIT (Tracked in Git)
| File / Directory | Purpose |
| :--- | :--- |
| `app.py` | Main Streamlit interactive dashboard entry point |
| `index.html` & `web/index.html` | MotionSites cinematic standalone HTML5 Canvas application |
| `src/` | Core algorithm implementations (`models.py`, `generator.py`, `exact_solver.py`, `approx_solver.py`, `benchmark.py`) |
| `tests/` | Automated unit & invariant test suite (`test_solvers.py`) |
| `requirements.txt` | Clean, cloud-compatible package dependencies |
| `.streamlit/config.toml` | Production Streamlit server & theme configuration |
| `Dockerfile` & `docker-compose.yml` | Containerization configurations |
| `Procfile` & `render.yaml` | Cloud deployment manifests for Render / Railway |
| `README.md` | Academic problem specification and architectural documentation |
| `VIVA_QUESTIONS_AND_ANSWERS.md` | Viva defense cheat sheet and flashcard guide |
| `pytest.ini` | Test runner path configuration |
| `shots/*.png` | Visual QA audit screenshots (optional, for README previews) |

### ❌ NEVER COMMIT (Excluded via `.gitignore`)
| Pattern | Why It Is Excluded |
| :--- | :--- |
| `__pycache__/`, `*.pyc` | Compiled bytecode — automatically regenerated on host |
| `venv/`, `.venv/`, `env/` | Local virtual environments — heavy and machine-specific |
| `.pytest_cache/`, `.coverage` | Local test runner caches |
| `.streamlit/secrets.toml` | Never expose passwords, API keys, or private tokens |
| `shots/chrome_profile/` | Headless Chrome browser telemetry cache |
| `.DS_Store`, `Thumbs.db` | Operating system metadata junk |
| `.vscode/`, `.idea/` | Local IDE editor preferences |

---

## 🚀 Option 1: Deploy on Streamlit Community Cloud (Recommended & 100% Free)

Streamlit Community Cloud is the standard, free platform for hosting Streamlit applications directly from GitHub.

### Step 1: Initialize Git and Push to GitHub
Open your terminal in `daa-facility-approx-dashboard`:

```bash
# 1. Initialize git repository
git init

# 2. Stage all production files (respecting .gitignore)
git add .

# 3. Check status to ensure NO __pycache__ or venv is staged
git status

# 4. Create initial commit
git commit -m "feat: complete publication-grade DAA PBL facility coverage dashboard"

# 5. Set main branch
git branch -M main

# 6. Link to your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git

# 7. Push to GitHub
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click the **"New app"** button.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>`.
4. Set **Branch:** `main`.
5. Set **Main file path:** `app.py`.
6. Click **"Deploy!"**
7. Your dashboard will be live with a public HTTPS link (e.g., `https://facility-optimization.streamlit.app`) in ~90 seconds!

---

## 🌐 Option 2: Deploy on Render.com (Free Web Service)

Render allows zero-configuration deployment using the included `render.yaml` or `Dockerfile`:

1. Sign up for free at **[render.com](https://render.com)**.
2. Click **"New +"** $\rightarrow$ **"Web Service"**.
3. Connect your GitHub repository.
4. Settings:
   * **Environment:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **"Create Web Service"**.

---

## 🎨 Option 3: Deploy the MotionSites Web App (`index.html`) on GitHub Pages / Vercel

If you want to host the standalone cinematic HTML5 Canvas visualizer publicly:

### On GitHub Pages (Free):
1. In your GitHub repository, go to **Settings** $\rightarrow$ **Pages**.
2. Under **Build and deployment**:
   * **Source:** `Deploy from a branch`
   * **Branch:** `main` / `root`
3. Click **Save**. Your web visualizer will be live at:
   `https://<YOUR_GITHUB_USERNAME>.github.io/<YOUR_REPOSITORY_NAME>/`

### On Vercel (1-Click):
1. Go to **[vercel.com](https://vercel.com)**.
2. Import your GitHub repository.
3. Click **Deploy**. Vercel will automatically serve `index.html` on a lightning-fast global CDN.

---

## 🐳 Option 4: Local or Cloud Docker Container

To run the containerized application on any machine with Docker installed:

```bash
# Build and launch using docker-compose
docker-compose up --build -d

# Open in your browser:
# http://localhost:8501
```

To stop:
```bash
docker-compose down
```

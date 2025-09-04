
# Alfred Task Manager — Flask + PostgreSQL (Codespaces)

This is your Alfred Task Manager wired to **PostgreSQL** and prepped for **GitHub Codespaces**.

## Quickstart (Codespaces)

1. Push these files to your repo.
2. In GitHub, open the repo in **Codespaces** → it will detect `.devcontainer/` and build.
3. Once the container is ready, run:
   ```bash
   cp .env.example .env   # (optional; devcontainer already sets DATABASE_URL)
   python app.py
   ```
4. Open forwarded port **5000** → the UI appears.

## What’s inside

- `app.py` — Flask API + Alfred assistant (LLM optional). Includes:
  - Postgres-backed store (via `db.py`)
  - “yes → add groceries” fallback in LLM mode
  - Natural-language due dates with cleaned descriptions
- `db.py` — SQLAlchemy models/repo for Postgres
- `index.html` — UI you shared
- `.devcontainer/` — Codespaces config
  - `docker-compose.yml` spins up `db` (Postgres)
  - `devcontainer.json` installs deps & runs `init_db()`
- `requirements.txt`, `.env.example`, `README.md`

## Local (no Docker)

Install Postgres yourself and set:
```
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/tasks
```

Then:
```bash
pip install -r requirements.txt
python - <<'PY'
from db import init_db
init_db()
print("DB initialized.")
PY
python app.py
```

## Notes

- Data persists in the `pgdata` Docker volume across container restarts.
- To inspect the DB inside Codespaces:
  ```bash
  sudo apt-get update && sudo apt-get install -y postgresql-client
  psql -h db -U postgres -d tasks
  ```

Happy shipping!

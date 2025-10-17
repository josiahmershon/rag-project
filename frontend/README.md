Configuration
=============

The Chainlit app will read a local `.env` file if present. You can customize:

```
# .env
BACKEND_API_URL=http://localhost:8001
BACKEND_ENDPOINT=/query-lc
DEFAULT_USER_GROUPS=executives,engineering
```

If `.env` is absent, defaults are used:
- BACKEND_API_URL: http://localhost:8001
- BACKEND_ENDPOINT: /query-lc
- DEFAULT_USER_GROUPS: executives,engineering

Run
---

```
chainlit run app.py --host 0.0.0.0 --port 8002 -w
```
## Chainlit Frontend (Minimal)

### Setup
1. Create and activate a virtual environment.
2. Install deps:

   pip install -r frontend/requirements.txt

3. Configure environment variables (create a `.env` in `frontend/` or export in your shell):
   - `BACKEND_API_URL` (default: `http://localhost:8001`)
   - `BACKEND_ENDPOINT` (default: `/query`)
   - `DEFAULT_USER_GROUPS` (default: `engineering,executives`)

### Run
```bash
chainlit run frontend/app.py -w
```

Then open the provided local URL. Type a question; results will include a compact sources list.

### Notes
- Keep `DEFAULT_USER_GROUPS` as a placeholder until LDAP/AD is wired in.
- To use the precise endpoint, set `BACKEND_ENDPOINT=/query-precise`.


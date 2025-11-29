Configuration
=============

Copy `env.example` to `.env` to get a default configuration, then tweak as needed:

```
cp frontend/env.example frontend/.env
```

Key options:
- `BACKEND_API_URL`, `BACKEND_ENDPOINT`, `DEFAULT_USER_GROUPS`
- History controls: `MAX_HISTORY_ENTRIES`, `MAX_HISTORY_DISPLAY`
- Attachment allowlist: `ALLOWED_ATTACHMENT_EXTENSIONS`

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

Then open the provided local URL. Type a question; the UI will:
- retain the current chat history (up to 40 turns) and send it along with every request
- accept attachments (PDF, DOCX, TXT, MD) for the current turn only; uploads are echoed back in the chat
- show a compact sources list limited to citations that appear in the assistant’s response

### Notes
- Keep `DEFAULT_USER_GROUPS` as a placeholder until LDAP/AD is wired in.
- To use the precise endpoint, set `BACKEND_ENDPOINT=/query-precise`.


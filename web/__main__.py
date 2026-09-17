import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "web.server:app",
        host=os.environ.get("TRADINGAGENTS_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("TRADINGAGENTS_WEB_PORT", "8501")),
    )

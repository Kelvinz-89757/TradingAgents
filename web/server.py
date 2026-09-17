"""FastAPI app for the TradingAgents web UI.

Endpoints:
  GET  /                       single-page UI
  GET  /api/options            providers, model lists, defaults, key presence
  GET  /api/runs               run summaries (newest first)
  POST /api/runs               start a run; body = form selections
  GET  /api/runs/{id}          full snapshot
  GET  /api/runs/{id}/events   Server-Sent Events: snapshot, then live updates
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()  # picks up API keys from ./.env like the CLI does

from cli.utils import _llm_provider_table, detect_asset_type, provider_default_url  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV  # noqa: E402
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS  # noqa: E402
from web.runner import _EOF, AGENT_TEAMS, SECTION_TITLES, RunManager  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="TradingAgents Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
manager = RunManager()


class RunRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    analysis_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    analysts: list[str] = Field(default_factory=lambda: ["market", "social", "news", "fundamentals"])
    research_depth: int = Field(default=1, ge=1, le=5)
    llm_provider: str
    backend_url: str | None = None
    quick_think_llm: str
    deep_think_llm: str
    output_language: str = "English"
    google_thinking_level: str | None = None
    openai_reasoning_effort: str | None = None
    anthropic_effort: str | None = None
    checkpoint_enabled: bool = False


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon-32.png", media_type="image/png")


@app.get("/api/options")
def options() -> dict[str, Any]:
    providers = []
    for display, key, url in _llm_provider_table():
        if key == "glm":
            url = "https://api.z.ai/api/paas/v4/"  # picker shows the CN URL; the intl one is the base
        env_var = PROVIDER_API_KEY_ENV.get(key)
        models = MODEL_OPTIONS.get(key, {"quick": [], "deep": []})
        providers.append(
            {
                "key": key,
                "display": display,
                "backend_url": url,
                "api_key_env": env_var,
                "has_key": env_var is None or bool(os.environ.get(env_var)),
                "quick_models": [{"label": lbl, "value": v} for lbl, v in models["quick"]],
                "deep_models": [{"label": lbl, "value": v} for lbl, v in models["deep"]],
            }
        )
    # Regional variants share a base entry in the picker; expose them so the
    # dropdown can offer the CN endpoints too.
    for key, display, url in (
        ("qwen-cn", "Qwen (China / DashScope)", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("glm-cn", "GLM (China / BigModel)", "https://open.bigmodel.cn/api/paas/v4/"),
        ("minimax-cn", "MiniMax (China)", "https://api.minimaxi.com/v1"),
    ):
        env_var = PROVIDER_API_KEY_ENV.get(key)
        models = MODEL_OPTIONS.get(key, {"quick": [], "deep": []})
        providers.append(
            {
                "key": key,
                "display": display,
                "backend_url": url,
                "api_key_env": env_var,
                "has_key": bool(os.environ.get(env_var or "")),
                "quick_models": [{"label": lbl, "value": v} for lbl, v in models["quick"]],
                "deep_models": [{"label": lbl, "value": v} for lbl, v in models["deep"]],
            }
        )
    return {
        "providers": providers,
        "defaults": {
            "llm_provider": DEFAULT_CONFIG["llm_provider"],
            "quick_think_llm": DEFAULT_CONFIG["quick_think_llm"],
            "deep_think_llm": DEFAULT_CONFIG["deep_think_llm"],
            "backend_url": DEFAULT_CONFIG.get("backend_url")
            or provider_default_url(DEFAULT_CONFIG["llm_provider"]),
            "output_language": DEFAULT_CONFIG["output_language"],
            "research_depth": DEFAULT_CONFIG["max_debate_rounds"],
        },
        "analysts": [
            {"key": "market", "label": "Market Analyst"},
            {"key": "social", "label": "Sentiment Analyst"},
            {"key": "news", "label": "News Analyst"},
            {"key": "fundamentals", "label": "Fundamentals Analyst"},
        ],
        "depths": [
            {"label": "Shallow (1 round)", "value": 1},
            {"label": "Medium (3 rounds)", "value": 3},
            {"label": "Deep (5 rounds)", "value": 5},
        ],
        "languages": ["English", "中文", "日本語", "한국어", "Español", "Français", "Deutsch"],
        "teams": AGENT_TEAMS,
        "section_titles": SECTION_TITLES,
    }


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    return manager.list()


@app.post("/api/runs")
def create_run(req: RunRequest) -> dict[str, Any]:
    ticker = req.ticker.strip().upper()
    params = req.model_dump()
    params["ticker"] = ticker
    params["asset_type"] = detect_asset_type(ticker).value
    run = manager.start(params)
    return {"id": run.id, "asset_type": params["asset_type"]}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run.snapshot()


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, Any]:
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return {"id": run_id, "result": manager.cancel(run)}


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")

    async def gen():
        q = run.subscribe()
        try:
            # Snapshot first so a late-joining tab catches up; queued events
            # that arrive after this are strictly newer.
            yield _sse("snapshot", run.snapshot())
            loop = asyncio.get_running_loop()
            while True:
                try:
                    item = await loop.run_in_executor(None, q.get, True, 15)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if item is _EOF:
                    yield _sse("end", {"status": run.status})
                    break
                yield _sse(item["event"], item["data"])
        finally:
            run.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

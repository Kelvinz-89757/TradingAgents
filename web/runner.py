"""Background run manager for the web UI.

Mirrors the streaming loop in ``cli/main.py`` (analyst status transitions,
report-section updates, message log) but publishes the results as JSON events
on a per-run queue instead of painting a Rich layout. Runs are serialised on a
single worker lock because ``TradingAgentsGraph`` calls the process-global
``set_config`` on construction.
"""

from __future__ import annotations

import datetime
import queue
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from cli.main import (
    ANALYST_AGENT_NAMES,
    ANALYST_ORDER,
    ANALYST_REPORT_MAP,
    classify_message_type,
)
from cli.stats_handler import StatsCallbackHandler
from cli.utils import provider_default_url
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.reporting import write_report_tree

AGENT_TEAMS = {
    "Analyst Team": ["Market Analyst", "Sentiment Analyst", "News Analyst", "Fundamentals Analyst"],
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}

SECTION_TITLES = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Team Plan",
    "final_trade_decision": "Portfolio Management Decision",
}

MAX_MESSAGES = 300
_EOF = object()


class Run:
    """Mutable state for one analysis run plus its event fan-out."""

    def __init__(self, params: dict[str, Any]):
        self.id = uuid.uuid4().hex[:10]
        self.params = params
        self.created_at = datetime.datetime.now().isoformat(timespec="seconds")
        self.status = "queued"  # queued | running | completed | failed
        self.error: str | None = None
        self.agent_status: dict[str, str] = {}
        self.sections: dict[str, str | None] = {}
        self.messages: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}
        self.report_path: str | None = None
        self.decision: str | None = None
        self._processed_message_ids: set[str] = set()
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []

    # ---- snapshot / events -------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "params": self.params,
                "created_at": self.created_at,
                "status": self.status,
                "error": self.error,
                "agent_status": dict(self.agent_status),
                "sections": dict(self.sections),
                "messages": list(self.messages),
                "stats": dict(self.stats),
                "report_path": self.report_path,
                "decision": self.decision,
            }

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ticker": self.params["ticker"],
            "date": self.params["analysis_date"],
            "created_at": self.created_at,
            "status": self.status,
            "decision": self.decision,
        }

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(q)
            if self.status in ("completed", "failed"):
                q.put(_EOF)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            q.put({"event": event, "data": data})

    def _close(self) -> None:
        for q in list(self._subscribers):
            q.put(_EOF)

    # ---- state mutators (each emits its own event) -------------------------

    def set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.error = error
        self._emit("status", {"status": status, "error": error})

    def update_agent(self, agent: str, status: str) -> None:
        with self._lock:
            if self.agent_status.get(agent) == status:
                return
            self.agent_status[agent] = status
        self._emit("agent", {"agent": agent, "status": status})

    def update_section(self, name: str, content: str) -> None:
        with self._lock:
            if name not in self.sections or self.sections[name] == content:
                return
            self.sections[name] = content
        self._emit("section", {"name": name, "content": content})

    def add_message(self, kind: str, content: str) -> None:
        msg = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "type": kind,
            "content": content,
        }
        with self._lock:
            self.messages.append(msg)
            if len(self.messages) > MAX_MESSAGES:
                del self.messages[: len(self.messages) - MAX_MESSAGES]
        self._emit("message", msg)

    def update_stats(self, stats: dict[str, Any]) -> None:
        with self._lock:
            if stats == self.stats:
                return
            self.stats = dict(stats)
        self._emit("stats", self.stats)


class RunManager:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self._worker_lock = threading.Lock()

    def list(self) -> list[dict[str, Any]]:
        return [r.summary() for r in sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)]

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def start(self, params: dict[str, Any]) -> Run:
        run = Run(params)
        self.runs[run.id] = run
        threading.Thread(target=self._execute, args=(run,), daemon=True, name=f"run-{run.id}").start()
        return run

    def _execute(self, run: Run) -> None:
        with self._worker_lock:
            try:
                run.set_status("running")
                _run_analysis(run)
                run.set_status("completed")
            except Exception as exc:  # noqa: BLE001 - surface anything to the UI
                run.add_message("System", f"Error: {exc}\n{traceback.format_exc()}")
                run.set_status("failed", error=str(exc))
            finally:
                run._close()


def build_config(params: dict[str, Any]) -> dict[str, Any]:
    """Same precedence as ``cli.main._build_run_config`` minus the env-skip logic."""
    config = DEFAULT_CONFIG.copy()
    depth = int(params.get("research_depth", 1))
    config["max_debate_rounds"] = depth
    config["max_risk_discuss_rounds"] = depth
    provider = params["llm_provider"].lower()
    config["llm_provider"] = provider
    config["quick_think_llm"] = params["quick_think_llm"]
    config["deep_think_llm"] = params["deep_think_llm"]
    config["backend_url"] = (
        params.get("backend_url") or DEFAULT_CONFIG.get("backend_url") or provider_default_url(provider)
    )
    config["output_language"] = params.get("output_language") or "English"
    config["google_thinking_level"] = params.get("google_thinking_level") or None
    config["openai_reasoning_effort"] = params.get("openai_reasoning_effort") or None
    config["anthropic_effort"] = params.get("anthropic_effort") or None
    config["checkpoint_enabled"] = bool(params.get("checkpoint_enabled", False))
    return config


def _run_analysis(run: Run) -> None:
    params = run.params
    ticker = params["ticker"]
    analysis_date = params["analysis_date"]
    asset_type = params.get("asset_type", "stock")
    selected = [a for a in ANALYST_ORDER if a in set(params["analysts"])]
    if not selected:
        raise ValueError("Select at least one analyst")

    config = build_config(params)
    stats_handler = StatsCallbackHandler()

    # Seed status + section tables so the page can render the full board up front.
    for team_agents in AGENT_TEAMS.values():
        for agent in team_agents:
            if agent in ANALYST_AGENT_NAMES.values() and agent not in {
                ANALYST_AGENT_NAMES[k] for k in selected
            }:
                continue
            run.agent_status[agent] = "pending"
    for key in selected:
        run.sections[ANALYST_REPORT_MAP[key]] = None
    for key in ("investment_plan", "trader_investment_plan", "final_trade_decision"):
        run.sections[key] = None
    run._emit("init", {"agent_status": dict(run.agent_status), "sections": list(run.sections)})

    run.add_message("System", f"Selected ticker: {ticker}")
    if asset_type != "stock":
        run.add_message("System", f"Asset type: {asset_type}")
    run.add_message("System", f"Analysis date: {analysis_date}")
    run.add_message("System", f"Selected analysts: {', '.join(selected)}")
    run.add_message(
        "System",
        f"Provider: {config['llm_provider']} | quick={config['quick_think_llm']} deep={config['deep_think_llm']}",
    )

    graph = TradingAgentsGraph(selected, config=config, debug=True, callbacks=[stats_handler])
    run.update_agent(ANALYST_AGENT_NAMES[selected[0]], "in_progress")

    instrument_context = graph.resolve_instrument_context(ticker, asset_type)
    init_state = graph.propagator.create_initial_state(
        ticker, analysis_date, asset_type=asset_type, instrument_context=instrument_context
    )
    args = graph.propagator.get_graph_args(callbacks=[stats_handler])
    checkpoint_tid = graph.begin_checkpoint(ticker, analysis_date, asset_type)
    if checkpoint_tid is not None:
        args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_tid

    trace: list[dict[str, Any]] = []
    try:
        for chunk in graph.graph.stream(graph.checkpoint_input(init_state), **args):
            _process_chunk(run, chunk, selected)
            run.update_stats(stats_handler.get_stats())
            trace.append(chunk)
        graph.clear_checkpoint_on_success(ticker, analysis_date, asset_type)
    finally:
        graph.end_checkpoint()

    final_state: dict[str, Any] = {}
    for chunk in trace:
        final_state.update(chunk)

    for agent in list(run.agent_status):
        run.update_agent(agent, "completed")
    for section in list(run.sections):
        if final_state.get(section):
            run.update_section(section, final_state[section])

    # Persist the same report tree the CLI writes, under results_dir/web/.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(config["results_dir"]) / "web" / f"{ticker}_{analysis_date}_{timestamp}"
    try:
        report_file = write_report_tree(final_state, ticker, save_path)
        run.report_path = str(report_file)
        run.add_message("System", f"Report saved to {save_path}")
    except Exception as exc:  # noqa: BLE001
        run.add_message("System", f"Could not save report: {exc}")

    try:
        run.decision = graph.process_signal(final_state.get("final_trade_decision", ""))
    except Exception:  # noqa: BLE001 - signal extraction is best-effort
        run.decision = None
    run._emit("done", {"decision": run.decision, "report_path": run.report_path})
    run.update_stats(stats_handler.get_stats())
    run.add_message("System", f"Completed analysis for {analysis_date}")


def _process_chunk(run: Run, chunk: dict[str, Any], selected: list[str]) -> None:
    for message in chunk.get("messages", []):
        msg_id = getattr(message, "id", None)
        if msg_id is not None:
            if msg_id in run._processed_message_ids:
                continue
            run._processed_message_ids.add(msg_id)
        msg_type, content = classify_message_type(message)
        if content and content.strip():
            run.add_message(msg_type, content)
        tool_calls = getattr(message, "tool_calls", None) or []
        for tc in tool_calls:
            name = tc["name"] if isinstance(tc, dict) else tc.name
            targs = tc["args"] if isinstance(tc, dict) else tc.args
            run.add_message("Tool", f"{name}({targs})")

    # Analysts: completed once their report exists; first without one is active.
    found_active = False
    for key in selected:
        report_key = ANALYST_REPORT_MAP[key]
        if chunk.get(report_key):
            run.update_section(report_key, chunk[report_key])
        if run.sections.get(report_key):
            run.update_agent(ANALYST_AGENT_NAMES[key], "completed")
        elif not found_active:
            run.update_agent(ANALYST_AGENT_NAMES[key], "in_progress")
            found_active = True
        else:
            run.update_agent(ANALYST_AGENT_NAMES[key], "pending")
    if not found_active and run.agent_status.get("Bull Researcher") == "pending":
        run.update_agent("Bull Researcher", "in_progress")

    if chunk.get("investment_debate_state"):
        d = chunk["investment_debate_state"]
        bull, bear, judge = (d.get(k, "").strip() for k in ("bull_history", "bear_history", "judge_decision"))
        if bull or bear:
            for a in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                if run.agent_status.get(a) != "completed":
                    run.update_agent(a, "in_progress")
        parts = []
        if bull:
            parts.append(f"### Bull Researcher Analysis\n{bull}")
        if bear:
            parts.append(f"### Bear Researcher Analysis\n{bear}")
        if judge:
            parts.append(f"### Research Manager Decision\n{judge}")
        if parts:
            run.update_section("investment_plan", "\n\n".join(parts))
        if judge:
            for a in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                run.update_agent(a, "completed")
            run.update_agent("Trader", "in_progress")

    if chunk.get("trader_investment_plan"):
        run.update_section("trader_investment_plan", chunk["trader_investment_plan"])
        if run.agent_status.get("Trader") != "completed":
            run.update_agent("Trader", "completed")
            run.update_agent("Aggressive Analyst", "in_progress")

    if chunk.get("risk_debate_state"):
        r = chunk["risk_debate_state"]
        parts = []
        for agent, key, title in (
            ("Aggressive Analyst", "aggressive_history", "Aggressive Analyst Analysis"),
            ("Conservative Analyst", "conservative_history", "Conservative Analyst Analysis"),
            ("Neutral Analyst", "neutral_history", "Neutral Analyst Analysis"),
        ):
            hist = r.get(key, "").strip()
            if hist:
                if run.agent_status.get(agent) != "completed":
                    run.update_agent(agent, "in_progress")
                parts.append(f"### {title}\n{hist}")
        judge = r.get("judge_decision", "").strip()
        if judge:
            parts.append(f"### Portfolio Manager Decision\n{judge}")
        if parts:
            run.update_section("final_trade_decision", "\n\n".join(parts))
        if judge:
            for a in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"):
                run.update_agent(a, "completed")

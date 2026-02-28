import asyncio
import uuid
import json
import os
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from .schemas import AnalyzeRequest, AnalyzeResponse, EventEnvelope
from .ws import WSManager, TERMINAL_EVENT_TYPES
from .orchestrator import run_pipeline

load_dotenv()

app = FastAPI()

_FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip()
if _FRONTEND_ORIGIN:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_FRONTEND_ORIGIN],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

ws_manager = WSManager()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    analysis_id = f"a_{uuid.uuid4().hex[:12]}"

    # Kick off background pipeline
    asyncio.create_task(run_pipeline(
        analysis_id=analysis_id,
        prompt=req.prompt,
        publish=ws_manager.publish
    ))

    return AnalyzeResponse(
        analysis_id=analysis_id,
        ws_url=f"/ws/analysis/{analysis_id}",
        sse_url=f"/sse/analysis/{analysis_id}",
    )


@app.websocket("/ws/analysis/{analysis_id}")
async def ws_analysis(ws: WebSocket, analysis_id: str):
    await ws_manager.stream_to_ws(analysis_id, ws)


def _sse_data(event: dict) -> str:
    payload = json.dumps(event, separators=(",", ":"))
    return f"data: {payload}\n\n"


@app.get("/sse/analysis/{analysis_id}")
async def sse_analysis(request: Request, analysis_id: str):
    def _event_key(event: dict) -> str:
        return json.dumps(event, sort_keys=True, separators=(",", ":"))

    async def event_generator():
        seen: set[str] = set()

        # Replay history first.
        for event in ws_manager.get_history(analysis_id):
            seen.add(_event_key(event))
            yield _sse_data(event)
            if event.get("type") in TERMINAL_EVENT_TYPES:
                return

        # Stream future events from queue.
        q = ws_manager.get_queue(analysis_id)
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(q.get(), timeout=15)
            except asyncio.TimeoutError:
                # Keep connection warm through proxies/load balancers.
                yield ": keep-alive\n\n"
                continue

            event_key = _event_key(event)
            if event_key in seen:
                continue
            seen.add(event_key)
            yield _sse_data(event)
            if event.get("type") in TERMINAL_EVENT_TYPES:
                return

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )

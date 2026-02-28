import asyncio
import uuid
from fastapi import FastAPI, WebSocket
from .schemas import AnalyzeRequest, AnalyzeResponse, EventEnvelope
from .ws import WSManager
from .orchestrator import run_pipeline

app = FastAPI()
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

    return AnalyzeResponse(analysis_id=analysis_id)


@app.websocket("/ws/analysis/{analysis_id}")
async def ws_analysis(ws: WebSocket, analysis_id: str):
    await ws_manager.stream_to_ws(analysis_id, ws)
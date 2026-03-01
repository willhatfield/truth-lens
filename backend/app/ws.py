import asyncio
from typing import Dict, List
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

TERMINAL_EVENT_TYPES = ("DONE", "FATAL_ERROR")


class WSManager:
    def __init__(self, history_limit: int = 500) -> None:
        self.queues: Dict[str, "asyncio.Queue[dict]"] = {}
        self.history: Dict[str, List[dict]] = {}
        self.history_limit = history_limit

    def get_queue(self, analysis_id: str) -> "asyncio.Queue[dict]":
        if analysis_id not in self.queues:
            self.queues[analysis_id] = asyncio.Queue()
        return self.queues[analysis_id]

    def _append_history(self, analysis_id: str, event: dict) -> None:
        if analysis_id not in self.history:
            self.history[analysis_id] = []
        self.history[analysis_id].append(event)
        # keep only last N
        if len(self.history[analysis_id]) > self.history_limit:
            self.history[analysis_id] = self.history[analysis_id][-self.history_limit :]

    def get_history(self, analysis_id: str) -> List[dict]:
        return list(self.history.get(analysis_id, []))

    async def publish(self, analysis_id: str, event: dict) -> None:
        self._append_history(analysis_id, event)
        q = self.get_queue(analysis_id)
        await q.put(event)

    async def stream_to_ws(self, analysis_id: str, ws: WebSocket) -> None:
        await ws.accept()

        for event in self.history.get(analysis_id, []):
            try:
                await ws.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                return
            if event.get("type") in TERMINAL_EVENT_TYPES:
                await ws.close()
                return

        q = self.get_queue(analysis_id)
        try:
            while True:
                event = await q.get()
                await ws.send_json(event)
                if event.get("type") in TERMINAL_EVENT_TYPES:
                    break
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            try:
                await ws.close()
            except Exception:
                pass

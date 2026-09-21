"""HTTP proposal service: POST /predict with base64 camera images."""
import argparse
import base64
import io
import threading
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .format import messages, ordered_images


class Cameras(BaseModel):
    head: str
    left: str
    right: str


class Request(BaseModel):
    instruction: str = ""
    question: str | None = None
    memory: str = ""
    task_type: Literal["full_qa", "subtask_only"] = "full_qa"
    images: Cameras


def create_app(policy):
    app = FastAPI(title="Tau0 Proposal")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"ready": True}

    @app.post("/predict")
    def predict(request: Request):
        start = time.perf_counter()
        row = request.model_dump()
        images = []
        try:
            for value in ordered_images(row["images"]):
                if value.startswith("data:"):
                    value = value.split(",", 1)[1]
                with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as image:
                    images.append(image.convert("RGB").resize((640, 480), Image.Resampling.BILINEAR))
            messages(row, images)  # Validate the same prompt contract as CLI/SFT.
        except (ValueError, IndexError, UnidentifiedImageError, OSError) as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        # One generation at a time: the Qwen cache and CUDA memory are shared.
        with lock:
            result = policy.predict(row, images=images)
        return {**result, "task_type": request.task_type, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10089)
    args = parser.parse_args()
    import uvicorn
    from .runtime import Proposal
    uvicorn.run(create_app(Proposal(args.model, args.adapter, args.device)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

from contextlib import asynccontextmanager
import os
import uuid
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from src.models.multi_track_finder import MultiTrackFinder


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Q-Tracker API", lifespan=lifespan)


@app.get("/")
async def index():
    return {"message": "Welcome to the Q-Tracker API"}


@app.post("/evaluate-track-finder/")
async def evaluate_track_finder(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Upload a ROOT file to evaluate Track-Finder performance.",
    ),
):
    if not file.filename.lower().endswith(".root"):
        raise HTTPException(status_code=400, detail="Only ROOT files are allowed.")

    os.makedirs("tmp", exist_ok=True)
    file_id = uuid.uuid4().hex
    path = f"tmp/{file_id}.root"

    try:
        async with aiofiles.open(path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                await f.write(chunk)

        multi_track_finder = MultiTrackFinder()
        results = await run_in_threadpool(
            multi_track_finder.evaluate,
            path,
        )

        return results

    except IOError:
        raise HTTPException(status_code=500, detail="Error writing uploaded file.")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal processing error.")
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

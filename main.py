from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.tasks import repeat_every
from pydantic import BaseModel
from uuid import uuid4
from chat_memory_handler import ChatMemoryHandler
import asyncio
from contextlib import asynccontextmanager
import time
from fastapi import UploadFile, File, Form
from fastapi import Query
import shutil
import os
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


SESSION_TIMEOUT = 180


app = FastAPI()

# Allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or my deployed frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store memory sessions per user
sessions = {}
session_last_used = {}
locks = {}  # Lock for each session to handle concurrency

class MessageRequest(BaseModel):
    session_id: str
    message: str

class GeneratePDFRequest(BaseModel):
    session_id: str


@app.on_event("startup")
def start_clean():
    if os.path.exists("./inputs"):
        shutil.rmtree("./inputs")
    if os.path.exists("./outputs"):
        shutil.rmtree("./outputs")
    os.makedirs("./inputs")
    os.makedirs("./outputs")

@app.on_event("startup")
@repeat_every(seconds=180)
def cleanup_sessions():
    print("cleanup session attempted")
    now = time.time()
    to_delete = [sid for sid, last in session_last_used.items() if now - last > SESSION_TIMEOUT]  # 1/2 hr expiry
    for sid in to_delete:
        print("deleted id", sid, "remaining", sessions)
        path = "./inputs/" + sid + ".pdf"
        if os.path.exists(path):
            os.remove(path)
        path = "./outputs/" + sid + ".pdf"
        if os.path.exists(path):
            os.remove(path)
        sessions.pop(sid, None)
        locks.pop(sid, None)
        session_last_used.pop(sid, None)

@app.post("/chat")
async def chat_endpoint(data: MessageRequest):
    session_id = data.session_id

    # If the session doesn't exist, return an empty response
    if session_id not in sessions:
        return JSONResponse(content={"response": None})

    session_last_used[session_id] = time.gmtime()
    # Ensure that only one request is processed for the same session at a time
    lock = await get_session_lock(session_id)
    async with lock:
        response = sessions[session_id].ask(data.message)
        return {"response": response}

@app.post("/load")
async def retrieve_summary(
    session_id: str = Form(...),
    pdf_file: UploadFile = File(...)
):
    filename = f"{session_id}.pdf"
    pdf_path = "./inputs/" + filename

    # Save the uploaded file
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)
    
    # Initialize a new session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = ChatMemoryHandler()
        session_last_used[session_id] = time.time()
    print("session added:", session_id, "session list", sessions)

    # Ensure that only one request is processed for the same session at a time
    lock = await get_session_lock(session_id)
    async with lock:
        try:
            session_last_used[session_id] = time.time()
            sessions[session_id].load_pdf(pdf_path, delete=False)  # Load the PDF
            return JSONResponse(content=sessions[session_id].summary_json)
        except Exception as e:
            print("Failed to load PDF:", e)
            raise HTTPException(status_code=500, detail="Failed to load PDF")

@app.get("/generate_quiz")
async def generate_quiz_endpoint(session_id: str = Query(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    lock = await get_session_lock(session_id)
    async with lock:
        try:
            quiz_path = sessions[session_id].generate_quiz(f"./outputs/{session_id}.pdf")
            return FileResponse(
                path=quiz_path,
                media_type='application/pdf',
                filename="quiz.pdf"
            )
        except Exception as e:
            print("Failed to generate quiz:", e)
            raise HTTPException(status_code=500, detail="Quiz generation failed")


async def get_session_lock(session_id: str) -> asyncio.Lock:
    """Returns a lock object for a specific session, creating one if necessary."""
    if session_id not in locks:
        locks[session_id] = asyncio.Lock()
    return locks[session_id]

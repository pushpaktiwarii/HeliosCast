from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from app.api.predictions import router as predictions_router, background_prediction_task
import asyncio

app = FastAPI(
    title="HelioCast API",
    description="API for the AI-powered Space Weather Intelligence Platform",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_prediction_task())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Final force reload for the fully fixed model
app.include_router(predictions_router, prefix="/api", tags=["Predictions"])

frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")

if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    @app.get("/")
    async def root():
        return {"message": "HelioCast API is running. Frontend build not found."}

from fastapi import FastAPI
from app.core.config import settings
from app.infrastructure.api import health_router, sequential_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Focus Prediction API - GRU Sequential Model",
    version="1.0.0"
)

# Include routers
app.include_router(health_router)
app.include_router(sequential_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """Cargar modelo GRU al iniciar"""
    from app.core.dependencies import get_sequential_repo
    repo = get_sequential_repo()
    success = repo.load_model()
    if success:
        print("✅ GRU model loaded successfully on startup")
    else:
        print("⚠️ GRU model initialized (not trained)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
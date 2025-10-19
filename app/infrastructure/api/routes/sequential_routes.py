from fastapi import APIRouter
from app.application.controllers.sequential_controller import router as sequential_router

# Este archivo puede reexportar o extenderse según necesidades
router = APIRouter()
router.include_router(sequential_router)
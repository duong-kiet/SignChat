from fastapi import APIRouter

from app.api.routes import login, admin, register, dashboard, meeting, join

api_router = APIRouter()


api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(login.router, prefix="/login", tags=["login"])
api_router.include_router(register.router, prefix="/register", tags=["register"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(meeting.router, prefix="/meeting", tags=["meeting"])
api_router.include_router(join.router, prefix="/join", tags=["join"])


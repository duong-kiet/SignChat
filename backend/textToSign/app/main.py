from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status

from fastapi.responses import RedirectResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from app.dependencies import create_db_and_tables, host_port, host_ip
from app.config.settings import authenticatedUser
from app.api.main import api_router

def custom_generate_unique_id(route: APIRoute):
    return f"{route.tags[0]} - {route.name}"



@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(
    title="SignMeet Backend",
    lifespan=lifespan
)

@app.get("/")
async def home(request: Request):
    if (authenticatedUser):
        return RedirectResponse(f"http://{host_ip}:{host_port}/dashboard")
    return RedirectResponse(f"http://{host_ip}:{host_port}/login")

app.mount("/static", StaticFiles(directory="../Frontend/static"), name="static")
app.include_router(api_router)
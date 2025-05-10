from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session
from app.config.settings import authenticatedUser
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from app.dependencies import host_ip, host_port, getSession, templatePath

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def meeting(request: Request, response: Response, templates: Jinja2Templates = Depends(templatePath), session: Session = Depends(getSession)): 
    # if not authenticatedUser:
    #     raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authorized")
    # else:
    return templates.TemplateResponse("meeting.html", {"request": request}, status_code=status.HTTP_200_OK) 
    pass
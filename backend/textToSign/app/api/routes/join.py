from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session
from app.config.settings import authenticatedUser
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, Response, status
from app.dependencies import host_ip, host_port, getSession, templatePath
from app.utils.join_form import JoinForm

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def join(request: Request, response: Response, templates: Jinja2Templates = Depends(templatePath), session: Session = Depends(getSession)): 
    # if not authenticatedUser:
    #     raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authorized")
    # else:
    return templates.TemplateResponse("join.html", {"request": request}, status_code=status.HTTP_200_OK) 
    pass

@router.post("/")
async def joinMeeting(request: Request, response: Response, roomID: int = Form(), templates: Jinja2Templates = Depends(templatePath), session: Session = Depends(getSession)):
    form = JoinForm(request) 
    await form.load_data()

        #TODO: Check connectable or not, currently let the browser dump error
    if await form.is_valid():
        return RedirectResponse(f"http://{host_ip}:{host_port}/meeting?roomID={form.roomID}", status_code=status.HTTP_303_SEE_OTHER)
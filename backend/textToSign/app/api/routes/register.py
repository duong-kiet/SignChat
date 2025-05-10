from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session
from app.utils.registration_form import RegistrationForm
from app.models.models import User
from app.models.schemas import UserSignUpModel
from app.dependencies import getSession, templatePath
from sqlalchemy.exc import IntegrityError
from app.config.settings import authenticatedUser

router = APIRouter()

# @router.get('/request', response_class=HTMLResponse)
# async def signup(request: Request):
#     return template.TemplateResponse(request=request, name="/templates/register.html")

@router.get('/', response_class=HTMLResponse)
async def register(request: Request, response: Response, templates: Jinja2Templates = Depends(templatePath), session: Session = Depends(getSession)):
    # form = RegistrationForm()
    return templates.TemplateResponse("register.html", {"request": request}) # form=form

@router.post("/", status_code=status.HTTP_201_CREATED)
async def registerSubmit(request: Request, response: Response, first_name: str = Form(), last_name: str = Form(), username: str = Form(), email: str =  Form(), password: str = Form(), templates: Jinja2Templates = Depends(templatePath), session: Session = Depends(getSession)):
    form = RegistrationForm(request)
    await form.load_data()
    print(f"{username} + \n + {email} + \n + {password} ")
    if await form.is_valid():
        success = False
        try:
            available = True
            print(f"{username} + \n + {email} + \n + {password} ")
            inputEmail = session.query(User).filter(User.email == email).first()
            if inputEmail is not None:
                available = False
                response.status_code = status.HTTP_400_BAD_REQUEST
                # response.body = ""
                # TODO: Raise only for debugging
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already used!")
            
            inputUsername = session.query(User).filter(User.username == username).first()
            if inputUsername is not None:
                available = False
                # TODO: Raise only for debugging
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already used!")
            if available:
                newUser = User(
                    first_name = first_name,
                    last_name = last_name,
                    username = username,
                    email = email,
                    password = password,            # Add hashing functions if want
                    is_in_call = False
                )
                
                session.add(newUser)
                session.commit()
                session.refresh(newUser)
                success = True
                return templates.TemplateResponse("register.html", {"request": request, "success": success}, status_code=status.HTTP_201_CREATED) 
        except IntegrityError:
            return {"msg": "Error"}
    else:
        # TODO: Handle more exceptions (not only raise Exception but also can have a web interface report that)
        print("Error when creating form")
        errors = form.errors
        # response.status_code = status.HTTP_400_BAD_REQUEST
        return templates.TemplateResponse("register.html", {"request": request, "errors": errors}, status_code=status.HTTP_400_BAD_REQUEST)     #Usually hidden ...?
        # return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=errors)


    return templates.TemplateResponse("register.html", {
        "request": request,
        "errors": errors
    })
import os
from fastapi import APIRouter, Form, Request, Response, status, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import and_
from app.dependencies import getSession, templatePath, host_ip, host_port, getAuthenticatedUser
from app.models.models import User
from app.models.schemas import UserSignUpModel
from sqlmodel import Session
from sqlalchemy.future import select
from sqlalchemy.sql import and_
from app.utils.login_form import LoginForm
from sqlalchemy.exc import IntegrityError
from app.config.settings import authenticatedUser
# Route dựa trên cả tên file và tên hàm

# TODO: Check login credentials from db (password still store in plain text)
# TODO: It needed to report that which fields error, or simply notify errors (like username and/or password is not match) ?

router = APIRouter()
# frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Frontend'))
# domain_template_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/templates")

@router.get('/', response_class=HTMLResponse)
async def login(request: Request, response: Response, template: Jinja2Templates = Depends(templatePath), session: Session = Depends(getSession)):
    if (authenticatedUser):
        return RedirectResponse(f"http://{host_ip}:{host_port}/dashboard")
    return template.TemplateResponse("login.html", {'request': request})

@router.post('/', status_code=status.HTTP_200_OK)
async def loginSubmit(request: Request, response: Response, email: str = Form(), password: str = Form(), templates: Jinja2Templates = Depends(templatePath), session: Session = Depends(getSession), authenticatedUser: User = Depends(getAuthenticatedUser)):
    form = LoginForm(request) 
    await form.load_data()
    print(f"{email}\n{password}")

    if await form.is_valid():
        success = False
        try:
            #TODO: Maybe should reduce to one query instead of two (but cannot check which fields are correct or incorrect)
            inputEmail = session.exec(select(User).where(and_(User.email == email, User.password == password))).first()
            
            print(session.exec(select(User).where(and_(User.email == email, User.password == password))).first())
            if inputEmail is None:
                form.errors.append("Email not exist")
                response.status_code = status.HTTP_400_BAD_REQUEST

                print(form.errors)
                # TODO: Raise only for debugging
                return templates.TemplateResponse("login.html", {"request": request, "errors": form.errors}, status_code=status.HTTP_400_BAD_REQUEST)
            
                # raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already used!")
            else:
                # inputPassword = session.exec(select(User).where(User.password == password))
                # print(session.exec(select(User).where(User.password == password)))
                # if inputPassword is None:
                #     form.errors.append("Password incorrect")

                #     # TODO: Raise only for debugging
                #     return templates.TemplateResponse("login.html", {"request": request, "errors": form.errors}, status_code=status.HTTP_400_BAD_REQUEST)
                
                #     # raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password incorrect!")
                # else: 
                    success = True
                    print("Login successfully")
                    
                    # authenticatedUser = inputEmail
                    return RedirectResponse(f"http://{host_ip}:{host_port}/dashboard/", status_code=status.HTTP_303_SEE_OTHER)
                    # return RedirectResponse(request.url_for("../../../../Frontend/templates/dashboard.html"), status_code=status.HTTP_303_SEE_OTHER)
                    return templates.TemplateResponse("dashboard.html", {"request": request, "success": success}, status_code=status.HTTP_200_OK)
        except IntegrityError:
            return {"msg": "Error"}
    else:
        # TODO: Handle more exceptions (not only raise Exception but also can have a web interface report that)
        print("Error when login")
        errors = form.errors
        # response.status_code = status.HTTP_400_BAD_REQUEST
        return templates.TemplateResponse("login.html", {"request": request, "errors": errors}, status_code=status.HTTP_400_BAD_REQUEST)     #Usually hidden ...?
        # return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=errors)


    return templates.TemplateResponse("register.html", {
        "request": request,
        "errors": errors
    })
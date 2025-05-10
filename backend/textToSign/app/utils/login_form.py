from typing import List, Optional, Required
from fastapi import Request, Response, status

#TODO: Refactoring to integrate login form and registration form
#TODO: Add login through username if needed

class LoginForm:
    def __init__(self, request:Request) -> None:
        self.request: Request = request
        self.errors: List = []
        # self.username: str = ""
        self.email: str = ""
        self.password: str = ""

    async def load_data(self):
        form = await self.request.form()
        # self.username = form.get("username")
        self.email = form.get("email")
        self.password = form.get("password")

    async def is_valid(self):
        await self.load_data()
        # if not self.username:
        #     self.errors.append(f"Username is blank")
        if not self.email or not (self.email.__contains__("@")):
            self.errors.append("Email is required or not in correct format")
        if not self.password:
            self.errors.append(f"Password is blank")
        if not self.errors:
            return True
        return False


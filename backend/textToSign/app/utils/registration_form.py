from typing import List, Optional, Required
from fastapi import Request, Response, status

MAXIMUM_USERNAME_LENGTH = 20
MINIMUM_PASSWORD_LENGTH = 6

class RegistrationForm:
    def __init__(self, request:Request) -> None:
        self.request: Request = request
        self.errors: List = []
        self.first_name: str = ""
        self.last_name: str = ""
        self.username: str = ""
        self.email: str = ""
        self.password: str = ""

    async def load_data(self):
        form = await self.request.form()
        self.first_name = form.get("first_name")
        self.last_name = form.get("last_name")
        self.username = form.get("username")
        self.email = form.get("email")
        self.password = form.get("password")

    async def is_valid(self):
        await self.load_data()
        if not self.first_name:
            self.errors.append("First name is required")
        if not self.last_name:
            self.errors.append("Last name is required")
        if not self.username or len(self.username) > MAXIMUM_USERNAME_LENGTH:
            self.errors.append(f"Username should be < {MAXIMUM_USERNAME_LENGTH} chars")
        if not self.email or not (self.email.__contains__("@")):
            self.errors.append("Email is required or not in correct format")
        if not self.password or not len(self.password) >= MINIMUM_PASSWORD_LENGTH:
            self.errors.append(f"Password must be > {MINIMUM_PASSWORD_LENGTH} chars")
        if not self.errors:
            return True
        return False


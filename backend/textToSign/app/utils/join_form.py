from typing import List, Optional, Required
from fastapi import Request, Response, status

#TODO: Refactoring to integrate login form and registration form

class JoinForm:
    def __init__(self, request:Request) -> None:
        self.request: Request = request
        self.errors: List = []
        self.roomID: int = -1

    async def load_data(self):
        form = await self.request.form()
        self.roomID = form.get("roomID")

    async def is_valid(self):
        await self.load_data()
        if self.roomID == -1: 
            self.errors.append("No roomID is entered")
        if not self.errors:
            return True
        return False


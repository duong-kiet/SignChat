from pydantic import BaseModel
from typing import Optional

class UserSignUpModel(BaseModel):
    id: int | None = None
    username: str
    email: str
    password: str
    is_active: bool | None = None
    is_in_call: bool | None = None

    class Config:
        from_attribute = True
        # json_schema_extra = {
        #     'example': {
        #         "username": "USERNAME",
        #         "email": "email@email.com",
        #         "password": "PASSWORD",
        #         "is_active": True,
        #         "is_in_call": False
        #     }
        # }

    
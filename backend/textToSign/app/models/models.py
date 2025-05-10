import uuid, json

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import Column, Integer, Boolean, Text, String, ARRAY, ForeignKey
from sqlalchemy.orm import relationship

MAX_USERNAME_LENGTH = 20
MIN_PASSWORD_LENGTH = 20

# TODO: Implement password hashing feature

class User(SQLModel, table=True):
    __tablename__ = 'User'
    id: int | None = Field(default=None, primary_key=True)       # Để là None vì do được generate bởi database, chứ không phải từ input 
    first_name: str = Field(nullable=False)
    last_name: str = Field(nullable=False)
    username: str = Field(max_length=MAX_USERNAME_LENGTH, unique=True, nullable=False)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, nullable=False)
    email: EmailStr = Field(unique=True, nullable=False)
    is_in_call: bool | None = Field(default=False)
    # records = relationship("Record", back_populates='User')

    def __repr__(self):
        return f"User {self.username} with {self.id}"

# Dùng JSON để lưu dạng array 

class Record(SQLModel, table=True):
    __tablename__ = "Record"
    id: int | None = Field(default=None, primary_key=True)
    # Thêm link với User nào tham gia cuộc trò chuyện này
    messages: str = Field()
    user_ids: str = Field()
    # users = relationship("User", back_populates="Record")

    def __repr__(self):
        return f"Record {self.user_ids}"
    

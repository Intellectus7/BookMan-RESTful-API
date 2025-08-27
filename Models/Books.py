# --------- IMPORTS --------------------------------------------------------------------------------------
import sqlalchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, ForeignKey, Text, Float, JSON, Table, exists
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import faker, random, uuid, datetime, requests, re
from faker import Faker
from passlib.context import CryptContext
from .Run import Base, session
import uuid
#-------------------------------------------------------------------------------------------------------



#------------------ BOOK MODEL--------------------------------------
class Book(Base):
    __tablename__ = "books"
    BookId = Column(Integer, primary_key=True)
    Title = Column(String(100), nullable=False)
    Description = Column(Text)
    Author = Column(String(100), nullable=False, default="Anonymous")
    def to_dict(self):
        return {
            "BookId": BookId,
            "Title": Title,
            "Description": Description,
            "Author": Author
        }
#--------------------------------------------------------------------
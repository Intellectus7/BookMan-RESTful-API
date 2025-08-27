# --------- IMPORTS --------------------------------------------------------------------------------------
import sqlalchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, ForeignKey, Text, Float, JSON, Table, exists
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import  random, uuid, datetime, requests, re
from passlib.context import CryptContext
from .Run import session, Base
#-------------------------------------------------------------------------------------------------------



#------------------ PERSON MODEL--------------------------------------
class Person(Base):
    __tablename__ = "people"
    PersonId = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), unique=True)
    Username = Column(String, unique=True, nullable=False)
    Password = Column(String, nullable=False)
#--------------------------------------------------------------------
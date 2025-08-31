# --------- IMPORTS --------------------------------------------------------------------------------------
import sqlalchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, ForeignKey, Text, Float, JSON, Table, exists
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import faker, random, uuid, datetime, requests, re
from faker import Faker
from passlib.context import CryptContext
#-------------------------------------------------------------------------------------------------------


#--------- CONFIGURATION ------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Database setup ---
engine = create_engine("sqlite:///bookman.db", echo=True, future=True)
Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
session = Session()
Base = declarative_base()
faker = Faker()

#---------------------------------------------------------------------------------
def update(obj, **kwargs):
    for key, value in kwargs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
    session.commit()
    return obj

def delete(obj):
    session.delete(obj)
    session.commit()

def bool_table_check(modelClass, column, value):
    return session.query(exists().where(column == value)).scalar()

def find_table_by_key(tableClass, keyName, value):
    return session.query(tableClass).filter(getattr(tableClass, keyName) == value).first()


def find_table(tableClass, primaryKey, theUser):
    return session.query(tableClass).filter_by(primaryKey=theUser).first()
def add(*args):
    try:
        session.add_all(args)
        session.commit()
    except Exception as e:
        session.add(args)
        session.commit()



def run():
    Base.metadata.create_all(engine)
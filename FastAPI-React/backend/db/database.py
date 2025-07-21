from sqlalchemy import create_engine # Creates engine that connects to the database
from sqlalchemy.orm import sessionmaker # Creates a session that interacts with the engine
from sqlalchemy.ext.declarative import declarative_base # Base class that data models inherit from

from core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db(): # ensures only one session is used per request
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    Base.metadata.create_all(bind=engine)  # Creates all tables in the database based on the models defined        


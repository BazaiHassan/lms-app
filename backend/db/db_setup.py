from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql+psycopg2://hbazai:6099990917@localhost/lms_db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={},
    future=True
)

Sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()

def get_db():
    db = Sessionmaker()
    try:
        yield db
    finally:
        db.close()
from fastapi import FastAPI
from backend.api import users, ml_endpoints
from backend.db.db_setup import engine, async_engine
from backend.db.models import user

user.Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Fast API LMS",
    description="LMS for managing students and courses.",
    version="0.0.1",
    contact={
        "name": "HBazai",
        "email": "bazaee.hassan@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
)

app.include_router(users.router)
app.include_router(ml_endpoints.router)
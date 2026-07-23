from fastapi import FastAPI

from .routes import admin, bitrix, logistics
from .startup import lifespan

app = FastAPI(title="AI Server", version="0.1.0", lifespan=lifespan)

app.include_router(admin.router)
app.include_router(bitrix.router)
app.include_router(logistics.router)

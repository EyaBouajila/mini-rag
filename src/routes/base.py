from fastapi import FastAPI, APIRouter, Depends
import os
from helpers.config import get_settings, Settings


base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"],
)

@base_router.get("/")
def welcome(app_settings : Settings=Depends(get_settings)):
    
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION

    # app_name = os.environ.get("APP_NAME")
    # app_version = os.environ.get("APP_VERSION")
    return {
        "app_name": app_name,
        "app_version": app_version,
        }
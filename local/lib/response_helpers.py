#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from pydantic import BaseModel, Field

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class ErrorResponse(BaseModel):
    message: str = Field(description="Simple message describing the error")
    details: dict | None = Field(description="Optional dictionary containing extra details about the error")


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def server_error_exception(message = "unknown server error",
                          details: dict | None = None, status_code: int = 500) -> ErrorResponse:
    return HTTPException(status_code, jsonable_encoder({"error": message, "details": details}))

def client_error_exception(message = "unknown client error",
                          details: dict | None = None, status_code: int = 400) -> ErrorResponse:
    return server_error_exception(message, details, status_code)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from pydantic import BaseModel, Field
from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from local.lib.response_helpers import server_error_exception

from local.global_server_resources import DBACCESS, MODEL_LOADER
from local.lib.cpugpu_device import check_device_availability


# ---------------------------------------------------------------------------------------------------------------------
#%% Route request Types

class ModelSelectRequest(BaseModel):
    # Note: We can't use the name 'model_select' because 'model_' is used internally by fastapi...?
    dmodel_select: str = Field(description="Name of the detection model to switch to using")
    
class ModelDeviceRequest(BaseModel):
    use_gpu: bool = Field(description="Determines whether the detection model should use the GPU or CPU")
    dtype: str | None = Field(None, description="Data type to use when running detection model")

class DBServerURLRequest(BaseModel):
    dbserver_url: str = Field(description="This server will use the provided URL to communicate with the dbserver",
                              examples=["http://192.168.0.5:8050"])


# ---------------------------------------------------------------------------------------------------------------------
#%% Route response types

class ModelSelectResponse(BaseModel):
    dmodel_select: str = Field(description="Name of the currently selected detection model")
    dmodel_list: list[str] = Field(description="List of the names of all available detection models")

class ModelDeviceResponse(BaseModel):
    has_gpu: bool = Field(description="True if the server has support for GPU usage")
    device: str = Field(description="Name of device being used by the server", examples=["cpu", "cuda"])
    dtype: str = Field(description="Name describing data type used by the detection model",
                       examples=["float32", "float16", "bfloat16"])

class DBServerURLResponse(BaseModel):
    dbserver_url: str = Field(description="URL that this server uses to communicate with the dbserver")
    ok_connection: bool = Field(description="True if this server can make requests to the dbserver")


# ---------------------------------------------------------------------------------------------------------------------
#%% Helper functions

def get_model_select_info(model_loader_ref):
    
    ''' Function used to provide consistently formatted model selection results '''
    
    active_file = model_loader_ref.get_active_file()
    has_model_select = active_file is not None
    file_lut = model_loader_ref.get_file_name_to_path_lut()
    model_files_list = list(file_lut.keys())
    
    return has_model_select, active_file, model_files_list


# ---------------------------------------------------------------------------------------------------------------------
#%% Routes

settings_router = APIRouter(prefix="/v0/settings" , tags=["Settings"])

# .....................................................................................................................

@settings_router.get("", include_in_schema=False)
def v0_settings_page_route():
    return RedirectResponse("/static/v0/pages/settings.html")

# .....................................................................................................................

@settings_router.get("/get-dbserver-url")
def v0_settings_get_dbserver_base_url_route() -> DBServerURLResponse:
    
    '''
    Route used to check the current dbserver url used by this server,
    as well as whether the connection is available
    '''
    
    dbserver_url = DBACCESS.get_base_url()
    ok_url = DBACCESS.check_connection(feedback_on_error=False)
    
    return DBServerURLResponse(dbserver_url=dbserver_url, ok_connection=ok_url)

# .....................................................................................................................

@settings_router.post("/set-dbserver-url", status_code=201)
def v0_settings_set_dbserver_base_url_route(post_body: DBServerURLRequest) -> DBServerURLResponse:
    
    '''
    Route used to change the dbserver url used by this server.
    Can be used to redirect requests to a remote dbserver!
    '''
    
    new_dbserver_url = post_body.dbserver_url
    DBACCESS.set_url(new_dbserver_url)
    ok_url = DBACCESS.check_connection(feedback_on_error=False)
    
    return DBServerURLResponse(dbserver_url=new_dbserver_url, ok_connection=ok_url)

# .....................................................................................................................

@settings_router.get("/get-model")
def v0_settings_get_current_model_route() -> ModelSelectResponse:
    
    '''
    Route used to check the currently select detection model
    as well as list out all other model options
    '''
    
    has_model_select, model_select, model_files_list = get_model_select_info(MODEL_LOADER)
    if not has_model_select:
        raise server_error_exception("Error! No model selected (missing model files?)", 503)
    
    return ModelSelectResponse(dmodel_select=model_select, dmodel_list=model_files_list)

# .....................................................................................................................

@settings_router.post("/set-model", status_code=201)
def v0_settings_set_model_route(post_body: ModelSelectRequest) -> ModelSelectResponse:
    
    ''' Route used to switch to a different detection model '''
    
    # Bail if we can't set the model for whatever reason
    new_model_file_name = post_body.dmodel_select
    ok_set = MODEL_LOADER.set_active_file(new_model_file_name)
    if not ok_set:
        raise server_error_exception("Error setting new model!")
    
    has_model_select, model_select, model_files_list = get_model_select_info(MODEL_LOADER)
    if not has_model_select:
        raise server_error_exception("Error! No model selected (missing model files?)", 503)
    
    return ModelSelectResponse(dmodel_select=model_select, dmodel_list=model_files_list)

# .....................................................................................................................

@settings_router.get("/get-model/device")
def v0_settings_get_model_device_route() -> ModelDeviceResponse:
    
    ''' Route used to check the device usage (cpu vs gpu) of the model in use '''
    
    # Make sure we have a model to use
    ok_model, model = MODEL_LOADER.get_model()
    if not ok_model:
        raise server_error_exception("Error! Model is unavailable", status_code=503)
    
    model_device_str, model_dtype_str = model.get_device_str()
    has_gpu, _ = check_device_availability()
    
    return ModelDeviceResponse(has_gpu=has_gpu, device=model_device_str, dtype=model_dtype_str)

# .....................................................................................................................

@settings_router.post("/set-model/device", status_code=201)
def v0_settings_set_model_device_route(post_body: ModelDeviceRequest) -> ModelDeviceResponse:
    
    ''' Route used to change the device usage (i.e. cpu vs gpu) of the currently selected model '''
    
    # Make sure we have a model to use
    ok_model, model = MODEL_LOADER.get_model()
    if not ok_model:
        raise server_error_exception("Error! Model is unavailable", status_code=503)
    
    # Get gpu device (if server has one)
    has_gpu, fastest_device_str = check_device_availability()
    
    # Set device according to user preference
    device_str_to_use = fastest_device_str if post_body.use_gpu else "cpu"
    model_device_str, model_dtype_str = model.set_device(device_str_to_use, post_body.dtype)
    
    return ModelDeviceResponse(has_gpu=has_gpu, device=model_device_str, dtype=model_dtype_str)

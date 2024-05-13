#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os


# ---------------------------------------------------------------------------------------------------------------------
#%% Pathing env

def get_storage_path():
    return os.environ.get("STORAGE_FOLDER_PATH", "storage")

def get_models_path():
    storage_path = get_storage_path()
    return os.path.join(storage_path, "models")


# ---------------------------------------------------------------------------------------------------------------------
#%% Server env

def get_todserver_host() -> str:
    return os.environ.get("UVICORN_HOST", "0.0.0.0")

def get_todserver_port() -> int:
    return int(os.environ.get("UVICORN_PORT", 3834))

def get_todserver_url() -> str:
    host = get_todserver_host()
    port = get_todserver_port()
    return f"http://{host}:{port}"


# ---------------------------------------------------------------------------------------------------------------------
#%% Cleanup env

def get_auto_unload_timeout_sec() -> int:
    return int(os.environ.get("MODEL_AUTO_UNLOAD_TIMEOUT_SEC", 10 * 60))


# ---------------------------------------------------------------------------------------------------------------------
#%% DBServer env

def get_dbserver_base_url() -> str:
    return os.environ.get("DBSERVER_URL", "http://localhost:8050")


# ---------------------------------------------------------------------------------------------------------------------
#%% Demo

if __name__ == "__main__":
    
    print("", "Environment variables:", sep = "\n")
    print("STORAGE_FOLDER_PATH", get_storage_path())
    print("(models)", get_models_path())
    print("")
    print("TODSERVER_HOST", get_todserver_host())
    print("TODSERVER_PORT", get_todserver_port())
    print("")
    print("MODEL_AUTO_UNLOAD_TIMEOUT_SEC", get_auto_unload_timeout_sec())
    print("")
    print("DBSERVER_URL", get_dbserver_base_url())
    print("")

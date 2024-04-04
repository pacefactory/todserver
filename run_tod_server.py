#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from local.routes.detection import detection_router
from local.routes.demo import demo_router
from local.routes.settings import settings_router
from local.routes.misc import misc_router


# ---------------------------------------------------------------------------------------------------------------------
#%% Configure

# Create asgi application so we can start adding routes
asgi_app = FastAPI()
asgi_app.mount("/static", StaticFiles(directory="server_static_files"), name="static")

# Setup CORs support
origins = ["http://localhost", "https://localhost", "http://localhost:3834"]
asgi_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add sub-routes (order here affects documentation page!)
asgi_app.include_router(detection_router)
asgi_app.include_router(settings_router)
asgi_app.include_router(demo_router)
asgi_app.include_router(misc_router)


# ---------------------------------------------------------------------------------------------------------------------
#%% *** Launch server in debug mode ***

# Launch server for development if the script is run directly
if __name__ == "__main__":
    import os.path as osp
    import uvicorn
    from local.environment import get_todserver_host, get_todserver_port
    
    # Get server parameters
    host = get_todserver_host()
    port = get_todserver_port()
    this_file_name = osp.splitext(osp.basename(__file__))[0]
    uvicorn_launch_command = f"{this_file_name}:asgi_app"
    
    # Helpful feedback
    print(
        "",
        "Running Text-based Object Detection Server in debug mode!",
        "If you meant to run this outside of debug mode, use the command:",
        f"uvicorn {uvicorn_launch_command} --host {host} --port {port}",
        "",
        sep="\n",
    )
    
    # Run server!
    uvicorn.run(uvicorn_launch_command, host=host, port=port, reload=True)

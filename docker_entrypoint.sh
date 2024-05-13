#! /bin/sh

# Command to run when launching as a docker container (Blocking!)
exec uvicorn run_todserver:asgi_app


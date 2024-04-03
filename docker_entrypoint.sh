#! /bin/sh

# Command to run when launching as a docker container (Blocking!)
exec uvicorn run_tod_server:asgi_app


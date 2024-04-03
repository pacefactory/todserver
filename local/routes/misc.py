#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from fastapi import APIRouter
from fastapi.responses import RedirectResponse


# ---------------------------------------------------------------------------------------------------------------------
#%% Routes

misc_router = APIRouter(tags=["Miscellaneous"])

# .....................................................................................................................

@misc_router.get("/", include_in_schema=False)
def home_route():
    return RedirectResponse("/static/v0/pages/index.html")

# .....................................................................................................................

@misc_router.get("/favicon.ico", include_in_schema=False)
async def favicon_route():
    ''' Included to get rid of annoying 'missing favicon' errors! '''
    return RedirectResponse("/static/images/favicon.ico")

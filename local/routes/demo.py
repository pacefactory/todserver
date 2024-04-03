#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from fastapi import APIRouter
from fastapi.responses import RedirectResponse, Response

from local.lib.response_helpers import server_error_exception

from local.global_server_resources import DBACCESS


# ---------------------------------------------------------------------------------------------------------------------
#%% Route globals

JPEG_MIME_TYPE = "image/jpeg"


# ---------------------------------------------------------------------------------------------------------------------
#%% Routes

demo_router = APIRouter(prefix="/v0/demo" , tags=["Demo"])

# .....................................................................................................................

@demo_router.get("", include_in_schema=False)
def v0_demo_page_route():
    return RedirectResponse("/static/v0/pages/demo.html")

# .....................................................................................................................

@demo_router.get("/check-camera-list")
def v0_demo_check_camera_list_route() -> list[str]:
    
    ''' Returns a list of available cameras, based on data available from the dbserver '''
    
    ok_resp, camera_list = DBACCESS.get_all_camera_names()
    if not ok_resp:
        dbserver_url = DBACCESS.get_base_url()
        error_msg = "unable to access camera list"
        details_dict = {"dbserver_url": dbserver_url}
        raise server_error_exception(error_msg, details_dict, status_code=502)
    
    return camera_list

# .....................................................................................................................

@demo_router.get("/get-snapshot-bounding-ems/{camera_select}")
def v0_demo_get_snapshot_bounding_times_route(camera_select: str) -> tuple[int, int]:
    
    ''' Returns the earliest and latest snapshot ems values for a given camera '''
    
    ok_resp, min_max_ems = DBACCESS.get_snapshot_bounding_ems(camera_select)
    if not ok_resp:
        dbserver_url = DBACCESS.get_base_url()
        error_msg = "unable to retrieve snapshot bounding ems values"
        details_dict = {"dbserver_url": dbserver_url}
        raise server_error_exception(error_msg, details_dict, status_code=502)
    
    return min_max_ems

# .....................................................................................................................

@demo_router.get("/check-snapshot-image/{camera_select}/{snapshot_ems}",
                 responses={200: {"content": {JPEG_MIME_TYPE: {}}, "description": "Returns a jpeg image"}},
                 response_class=Response)
def v0_demo_check_snapshot_image_route(camera_select: str, snapshot_ems: int) -> Response:
    
    ''' Returns a single snapshot (jpeg) image from the dbserver '''
    
    ok_snap, snap_jpeg_bytes = DBACCESS.get_snapshot_bytes(camera_select, snapshot_ems)
    if not ok_snap:
        error_msg = "couldn't retrieve snapshot image"
        details_dict = {"camera_select": camera_select, "snapshot_ems": snapshot_ems}
        raise server_error_exception(error_msg, details_dict, status_code=502)
    
    return Response(content=snap_jpeg_bytes, media_type=JPEG_MIME_TYPE)

# .....................................................................................................................

@demo_router.get("/get-snapshot-ems-list/{camera_select}/{start_ems}/{end_ems}")
def v0_demo_get_snapshot_ems_list_route(camera_select: str, start_ems: int, end_ems: int) -> list[int]:
    
    ''' Returns a list of all available snapshot ems values between a given start/end range '''
    
    ok_snap, ems_list = DBACCESS.get_snapshot_ems_list(camera_select, start_ems, end_ems)
    if not ok_snap:
        error_msg = "couldn't retrieve snapshot ems list"
        details_dict = {"camera_select": camera_select, "start_ems": start_ems, "end_ems": end_ems}
        raise server_error_exception(error_msg, details_dict, status_code=502)
    
    return ems_list

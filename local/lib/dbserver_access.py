#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from typing import Any
from numpy.typing import NDArray

import requests

from local.lib.image_helpers import image_bytes_to_cvimage
from local.lib.request_helpers import get_bytes_request, get_json_request
from local.lib.timekeeper_utils import timestamped_str

from local.lib.cache import RAMCache


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class _BaseDBServerCameralessRetrieval:
    
    # .................................................................................................................
    
    def __init__(self, base_server_url: str):
        self.base_url = str(base_server_url)
    
    # .................................................................................................................
    
    def _make_url(self, *url_joins) -> str:
        url_joins_strs = (str(each_component) for each_component in url_joins)
        return "/".join([self.base_url, *url_joins_strs])
    
    # .................................................................................................................
    
    def get_json_request(self, *url_joins,
                         timeout_sec = 5.0, expected_status_code = 200, use_gzip = False) -> tuple[bool, Any]:
        
        request_url = self._make_url(*url_joins)
        ok_req, response_data = get_json_request(request_url, timeout_sec, expected_status_code, use_gzip)        
        
        return ok_req, response_data
    
    # .................................................................................................................


# =====================================================================================================================


class _BaseDBServerCollectionsRetrieval:
    
    # .................................................................................................................
    
    def __init__(self, base_server_url: str, collection_name: str):
        self.base_url = str(base_server_url)
        self.collection_name = str(collection_name)
    
    # .................................................................................................................
    
    def _make_url(self, camera_select: str, *url_joins) -> str:
        url_joins_strs = (str(each_component) for each_component in url_joins)
        return "/".join([self.base_url, str(camera_select), self.collection_name, *url_joins_strs])
    
    # .................................................................................................................
    
    def get_json_request(self, camera_select: str, *url_joins,
                         timeout_sec = 5.0, expected_status_code = 200, use_gzip = False) -> tuple[bool, Any]:
        
        request_url = self._make_url(camera_select, *url_joins)
        ok_req, response_data = get_json_request(request_url, timeout_sec, expected_status_code, use_gzip)        
        
        return ok_req, response_data
    
    # .................................................................................................................
    
    def get_jpeg_bytes_request(self, camera_select: str, *url_joins,
                                timeout_sec = 5.0, expected_status_code = 200) -> tuple[bool, Any]:
        
        request_url = self._make_url(camera_select, *url_joins)
        ok_req, response_data = get_bytes_request(request_url, timeout_sec, expected_status_code)
        
        return ok_req, response_data
    
    # .................................................................................................................


# =====================================================================================================================


class _DBServerSnapshotsCollection(_BaseDBServerCollectionsRetrieval):
    
    # .................................................................................................................
    
    def __init__(self, base_server_url):
        super().__init__(base_server_url, "snapshots")
    
    # .................................................................................................................
    
    def get_one_jpeg_by_ems(self, camera_select: str, snapshot_ems: int):
        ''' Returns: ok_request, jpeg_bytes '''
        return self.get_jpeg_bytes_request(camera_select, "get-one-image", "by-ems", snapshot_ems)
    
    # .................................................................................................................
    
    def get_one_metadata_by_ems(self, camera_select: str, snapshot_ems: int):        
        ''' Returns: ok_request, metadata_dict '''
        return self.get_json_request(camera_select, "get-one-metadata", "by-ems", snapshot_ems)
    
    # .................................................................................................................
    
    def get_ems_list_by_time_range(self, camera_select: str, start_time_ems: int, end_time_ems: int):
        ''' Returns: ok_request, snapshots_ems_list '''
        return self.get_json_request(camera_select, "get-ems-list", "by-time-range", start_time_ems, end_time_ems)
    
    # .................................................................................................................
    
    def get_bounding_times(self, camera_select: str):
        ''' Returns: ok_request, snapshot_bounding_times_dict '''
        return self.get_json_request(camera_select, "get-bounding-times")
    
    # .................................................................................................................
    
    def get_closest_metadata_by_time_target(self, camera_select: str, target_ems: int):
        ''' Returns: ok_request, closest_metadata (dict) '''
        return self.get_json_request(camera_select, "get-closest-metadata", "by-time-target", target_ems)
    
    # .................................................................................................................    


# =====================================================================================================================


class _DBServerMiscellaneousCollection(_BaseDBServerCameralessRetrieval):
    
    # .................................................................................................................
    
    def __init__(self, base_server_url):
        super().__init__(base_server_url)
    
    # .................................................................................................................
    
    def is_alive(self):
        ''' Returns: ok_request, alive_info_dict '''
        return self.get_json_request("is-alive")
    
    # .................................................................................................................
    
    def get_all_camera_names(self):
        ''' Returns: ok_request, camera_list '''
        return self.get_json_request("get-all-camera-names")


# =====================================================================================================================


class DBServerAccess:
    
    # .................................................................................................................
    
    def __init__(self, dbserver_url: str, jpeg_cache_size = 250):
        
        # Set up access to snapshot data
        self._base_url = None
        self._misc = None
        self._snapshots = None
        self.set_url(dbserver_url)
        
        # Set up cache to hold on to snapshot request results, to prevent over-requesting data
        self._jpeg_cache = RAMCache(max_cache_items = jpeg_cache_size)
    
    # .................................................................................................................
    
    def get_base_url(self) -> str:
        return self._base_url
    
    # .................................................................................................................
    
    def set_url(self, dbserver_url: str, check_connection = False) -> None:
        self._base_url = dbserver_url
        self._misc = _DBServerMiscellaneousCollection(self._base_url)
        self._snapshots = _DBServerSnapshotsCollection(self._base_url)
        return
    
    # .................................................................................................................
    
    def get_all_camera_names(self) -> tuple[bool, list[str]]:
        ok_resp, camera_list = self._misc.get_all_camera_names()
        return ok_resp, camera_list
    
    # .................................................................................................................
    
    def get_snapshot_bytes(self, camera_select: str, snapshot_ems: int) -> tuple[bool, bytes | None]:
        
        # Return data from cache, if possible
        ok_cache, jpeg_bytes = self._jpeg_cache.retrieve(camera_select, snapshot_ems)
        if ok_cache:
            return ok_cache, jpeg_bytes
        
        # Get data from dbserver if it wasn't in cache
        ok_resp, jpeg_bytes = self._snapshots.get_one_jpeg_by_ems(camera_select, snapshot_ems)
        if ok_resp:
            self._jpeg_cache.store(jpeg_bytes, camera_select, snapshot_ems)
        
        return ok_resp, jpeg_bytes
    
    # .................................................................................................................
    
    def get_snapshot(self, camera_select: str, snapshot_ems: int) -> tuple[bool, NDArray | None]:
        
        # Get raw jpeg data (from if possible)
        ok_resp, jpeg_bytes = self.get_snapshot_bytes(camera_select, snapshot_ems)
        
        # Convert to opencv compatible image, if we got a response
        snap_image = image_bytes_to_cvimage(jpeg_bytes) if ok_resp else None
        
        return ok_resp, snap_image
    
    # .................................................................................................................
    
    def get_snapshot_ems_list(self, camera_select: str, first_snapshot_ems:int, last_snapshot_ems: int,
                              return_ascending_order = True) -> tuple[bool, list[int]]:
        
        # Make sure first/last ems are ordered properly
        first_ems, last_ems = sorted((first_snapshot_ems, last_snapshot_ems))
        
        # Sort results as needed
        ok_resp, snap_ems_list = self._snapshots.get_ems_list_by_time_range(camera_select, first_ems, last_ems)
        if ok_resp:
            snap_ems_list = sorted(snap_ems_list, reverse = not return_ascending_order)
        
        return ok_resp, snap_ems_list
    
    # .................................................................................................................
    
    def get_snapshot_bounding_ems(self, camera_select: str) -> tuple[bool, tuple[int, int] | None]:
        
        '''
        Returns:
            ok_response, [min_ems, max_ems]
        '''
        
        bounding_ems = None
        ok_resp, snap_bounding_times_dict = self._snapshots.get_bounding_times(camera_select)
        if ok_resp:
            min_ems = snap_bounding_times_dict.get("min_epoch_ms", None)
            max_ems = snap_bounding_times_dict.get("max_epoch_ms", None)
            bounding_ems = [min_ems, max_ems] if None not in {min_ems, max_ems} else None
        
        return ok_resp, bounding_ems
    
    # .................................................................................................................
    
    def get_closest_snapshot_ems(self, camera_select: str, target_ems: int) -> tuple[bool, int | None]:
        
        # Make request to dbserver and then pull ems from metadata result, if possible
        ok_resp, closest_md = self._snapshots.get_closest_metadata_by_time_target(camera_select, target_ems)
        closest_ems = closest_md.get("epoch_ms", None) if ok_resp else None
        
        return ok_resp, closest_ems
    
    # .................................................................................................................
    
    def check_connection(self, feedback_on_error = True) -> bool:
        
        ''' Function used to test connection to the dbserver '''
        
        # Initialize outputs
        ok_resp = False
        
        try:
            ok_resp, _ = self._misc.is_alive()
            
        except requests.ConnectionError:
            if feedback_on_error:
                print("", timestamped_str("Error connecting to dbserver"), sep = "\n")
        
        except requests.exceptions.ReadTimeout:
            if feedback_on_error:
                print("", timestamped_str("Timeout error connecting to dbserver"), sep = "\n")
        
        except Exception as err:
            if feedback_on_error:
                print("", timestamped_str("Unknown error connecting to dbserver"), err, sep = "\n")
        
        return ok_resp
    
    # .................................................................................................................
    
    def clear_cache(self) -> None:
        
        ''' Function used to clear out cached jpeg data. Returns None '''
        
        self._jpeg_cache.clear()
        
        return
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions


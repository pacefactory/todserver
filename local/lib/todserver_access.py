#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from local.lib.request_helpers import get_json_request


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class TodServerAccessV0:
    
    ''' Class used to make requests to the todserver itself '''
    
    # .................................................................................................................
    
    def __init__(self, server_url: str):
        self._base_url = server_url
    
    # .................................................................................................................
    
    def _make_url(self, *url_joins) -> str:
        url_joins_strs = (str(each_component) for each_component in url_joins)
        return "/".join([self._base_url, "v0", *url_joins_strs])
    
    # .................................................................................................................
    
    def unload_model(self):
        
        req_url = self._make_url("settings", "reclaim-resources")
        ok_req, is_unloaded = get_json_request(req_url, expected_status_code=201)
        
        return ok_req, is_unloaded
    
    # .................................................................................................................

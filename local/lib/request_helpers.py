#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from typing import Any

import requests
import requests.exceptions as reqexcept

from local.lib.timekeeper_utils import timestamped_str


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def get_request(url, params = None, **kwargs) -> tuple[bool, Any]:
    
    '''
    Wrapper around requests library GET request, with error handling for connection errors
    Returns:
        ok_response, response_data
    '''
    
    # Initialize output
    ok_response = False
    response_data: Any = None
    
    try:
        response_data = requests.get(url, params, **kwargs)
        ok_response = True
    
    except (ConnectionError, ConnectionRefusedError,
            reqexcept.ConnectionError, reqexcept.ConnectTimeout, reqexcept.RetryError) as err:
        response_data = err
    
    except Exception as err:
        print("",
              timestamped_str("Unknown error with GET request"),
              "@ {}".format(url),
              err, sep = "\n")
        response_data = err
    
    return ok_response, response_data

# .....................................................................................................................

def get_bytes_request(request_url, timeout_sec = 5.0, expected_status_code = 200) -> tuple[bool, Any]:
    
    # Initialize output
    ok_response = False
    response_data = None
        
    # Make the actual get-request
    ok_get, get_response = get_request(request_url, timeout = timeout_sec)
    if not ok_get:
        return ok_response, response_data
    
    # Convert response to image data
    ok_response = (get_response.status_code == expected_status_code)
    try:
        response_data = get_response.content
        
    except ValueError as err:
        ok_response = False
        response_data = err
        print("",
              timestamped_str("DEBUG: GET bytes request error"),
              err, sep = "\n")
    
    return ok_response, response_data

# .....................................................................................................................

def get_json_request(request_url, timeout_sec = 5.0, expected_status_code = 200, use_gzip = False) -> tuple[bool, Any]:
    
    # Initialize output
    ok_response = False
    response_data = None
        
    # Make the actual get-request
    headers = {"Accept-Encoding": "gzip"} if use_gzip else {"Accept-Encoding": "identity"}
    
    # Try the get request (bail on connection errors, which implies server is down)
    ok_get, get_response = get_request(request_url, headers = headers, timeout = timeout_sec)
    if not ok_get:
        return ok_response, response_data
    
    # Convert json response data to python data type
    ok_response = (get_response.status_code == expected_status_code)
    try:
        response_data = get_response.json()
        
    except ValueError as err:
        ok_response = False
        response_data = err
        print("",
              timestamped_str("DEBUG: GET json request parsing error"),
              "@ {}".format(request_url),
              "STATUS:", get_response.status_code,
              "Response:", get_response,
              "Error:", err, sep = "\n")
    
    return ok_response, response_data

# .....................................................................................................................

def post_json_request(request_url, post_data, timeout_sec = 5.0, expected_status_code = 201, use_gzip = False,
                      is_json_response = True) -> tuple[bool, Any]:
    
    '''
    Function used to make a POST request to a given url. By default, assumes a json response, if the
    'is_json_response' flag is set to False, then a binary response is assumed instead
    Returns:
        ok_response, response_data
    '''
    
    # Initialize output
    ok_response = False
    response_data = None
    
    # Make the actual post-request
    headers = {"Accept-Encoding": "gzip"} if use_gzip else {"Accept-Encoding": "identity"}
    
    try:
        post_data = jsonable_encoder(post_data)
        post_response = requests.post(request_url, json = post_data, headers = headers, timeout = timeout_sec)
        
    except Exception as err:
        print("",
              timestamped_str("ERROR - Could not complete post request!"),
              "Data:", post_data,
              "", "Exception:", err, sep = "\n", flush = True)
        return ok_response, response_data
    
    # Convert json response data to python data type
    ok_response = (post_response.status_code == expected_status_code)
    try:
        response_data = post_response.json() if is_json_response else post_response.content
        
    except ValueError as err:
        ok_response = False
        response_data = err
        print("",
              timestamped_str("DEBUG: POST request response parsing error"),
              "Response:", post_response,
              "Error:", err, sep = "\n")
    
    return ok_response, response_data

# .....................................................................................................................

def check_connection(request_url, timeout_sec = 5.0, expected_status_code = 200) -> tuple[bool, bool, bool]:
    
    '''
    Helper used to check a connection by requesting a given url
    and looking for connection or timeout errors.
    Note: This function doesn't care about the data returned from the request
    Returns:
        ok_status, connect_error, timeout_error
    '''
    
    # Initialize outputs
    ok_status = False
    connect_error = False
    timeout_error = False
    
    # Try a get request to see if connection succeeds
    try:
        get_response = requests.get(request_url, timeout = timeout_sec)
        ok_status = (get_response.status_code == expected_status_code)
    
    except (ConnectionError, reqexcept.ConnectionError):
        connect_error = True
        
    except reqexcept.ConnectTimeout:
        timeout_error = True
    
    return ok_status, connect_error, timeout_error

# .....................................................................................................................




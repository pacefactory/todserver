#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from time import sleep
from local.lib.timekeeper_utils import (
    get_utc_datetime, get_time_delta_sec, add_seconds_to_datetime, timestamped_str
)

from local.environment import get_todserver_url, get_auto_unload_timeout_sec
from local.lib.todserver_access import TodServerAccessV0


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def get_next_unload_dt():
    curr_dt = get_utc_datetime()
    unload_timeout_sec = get_auto_unload_timeout_sec()
    return add_seconds_to_datetime(curr_dt, unload_timeout_sec)


# ---------------------------------------------------------------------------------------------------------------------
#%% Setup

# Create access to todserver for making unload requests
todserver = TodServerAccessV0(get_todserver_url())

# Set initial event timing
next_model_unload_dt = get_next_unload_dt()


# ---------------------------------------------------------------------------------------------------------------------
#%% Daemon loop

# Feedback/logging for start timing
print("", timestamped_str("Resource clean-up daemon started!"), sep = "\n", flush=True)

try:
    
    while True:
        
        # Check for event timing
        curr_dt = get_utc_datetime()
        
        # Request model unload periodically
        need_unload = (curr_dt > next_model_unload_dt)
        if need_unload:
            ok_req, is_unloaded = todserver.unload_model()
            if ok_req and is_unloaded:
                print(timestamped_str("Unloaded model"))
            elif not ok_req:
                print(timestamped_str("Unable to request model unload (server is down?)"))
            next_model_unload_dt = get_next_unload_dt()
        
        # Treat sleep loop as 'sleep till nearest next event'
        dts_list = [next_model_unload_dt]
        sec_to_event_list = [get_time_delta_sec(curr_dt, each_dt, force_positive_value=False) for each_dt in dts_list]
        sec_to_sleep = min(sec_to_event_list) + 5
        sec_to_sleep = max(sec_to_sleep, 60)
        
        sleep(sec_to_sleep)
    
except KeyboardInterrupt:
    print(timestamped_str("Keyboard cancel!"))

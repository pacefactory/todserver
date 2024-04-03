#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import time
import datetime as dt


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes


# ---------------------------------------------------------------------------------------------------------------------
#%% Types

DATETIME_TYPE = dt.datetime
DATE_TYPE = dt.date
TIME_TYPE = dt.time
TIMEDELTA_TYPE = dt.timedelta
TIMEZONE_TYPE = dt.timezone


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................
    
def get_utc_datetime() -> DATETIME_TYPE:

    ''' Returns a datetime object based on UTC time, with timezone information included '''

    return dt.datetime.utcnow().replace(tzinfo = get_utc_tzinfo())
    
# .....................................................................................................................
    
def get_local_datetime() -> DATETIME_TYPE:

    ''' Returns a datetime object based on the local time, with timezone information included '''

    return dt.datetime.now(tz = get_local_tzinfo())

# .....................................................................................................................

def get_local_tzinfo() -> TIMEZONE_TYPE:
    
    ''' Function which returns a local tzinfo object. Accounts for daylight savings '''
    
    # Figure out utc offset for local time, accounting for daylight savings
    is_daylight_savings = time.localtime().tm_isdst
    utc_offset_sec = time.altzone if is_daylight_savings else time.timezone
    utc_offset_delta = dt.timedelta(seconds = -utc_offset_sec)
    
    return dt.timezone(offset = utc_offset_delta)
    
# .....................................................................................................................

def get_utc_tzinfo() -> TIMEZONE_TYPE:
    
    ''' Convenience function which returns a utc tzinfo object '''
    
    return dt.timezone.utc

# .....................................................................................................................

def timestamped_str(message, separator = "  |  ") -> str:
    
    # Get current time
    locat_dt = get_local_datetime()
    timestamp_str = datetime_to_human_readable_string(locat_dt)
    
    # Prefix message with timestamp
    print_str =  "{}{}{}".format(timestamp_str, separator, message)
    
    return print_str

# .....................................................................................................................

def get_time_delta_sec(start_datetime, end_datetime, force_positive_value = True) -> float:
    
    '''
    Convenience function, used to get time delta between 2 datetimes, in seconds
    Calculated as: end_dt - start_dt
    Returns:
        total_seconds_between_start_and_end
    '''
    
    time_delta = end_datetime - start_datetime
    total_sec = time_delta.total_seconds()
    
    return abs(total_sec) if force_positive_value else total_sec

# .....................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Datatime conversion functions

# .....................................................................................................................

def add_seconds_to_datetime(input_datetime: DATETIME_TYPE, seconds_to_add: int) -> DATETIME_TYPE:
    ''' Adds the specified number of seconds to the given datetime '''
    return input_datetime + dt.timedelta(seconds = seconds_to_add)

# .....................................................................................................................

def datetime_to_human_readable_string(input_datetime) -> str:
    
    '''
    Converts a datetime object into a 'human friendly' string
    Example:
        "2019-01-30 05:11:33 PM (-0400 UTC)"
    
    Note: This function assumes the datetime object has timezone information (tzinfo)
    '''
    
    return input_datetime.strftime("%Y-%m-%d %I:%M:%S %p (%z UTC)")

# .....................................................................................................................

def datetime_to_isoformat_string(input_datetime) -> str:
    
    '''
    Converts a datetime object into an isoformat string
    Example:
        "2019-01-30T11:22:33+00:00.000000"
    
    Note: This function assumes the datetime object has timezone information (tzinfo)
    '''
    
    return input_datetime.isoformat()

# .....................................................................................................................

def datetime_to_epoch_ms(input_datetime) -> int:
    
    ''' Function which converts a datetime to the number of milliseconds since the 'epoch' (~ Jan 1970) '''
    
    return int(round(1000 * input_datetime.timestamp()))

# .....................................................................................................................

def datetime_convert_to_day_start(input_datetime) -> DATETIME_TYPE:
    
    ''' Function which takes in a datetime and returns a datetime as of the start of that day '''
    
    return input_datetime.replace(hour = 0, minute = 0, second = 0, microsecond = 0)

# .....................................................................................................................

def datetime_convert_to_day_end(input_datetime) -> DATETIME_TYPE:
    
    ''' Function which takes in a datetime and returns a datetime as of the end of that day (minus 1 second) '''
    
    return input_datetime.replace(hour = 23, minute = 59, second = 59, microsecond = 0)

# .....................................................................................................................

def local_datetime_to_utc_datetime(local_datetime) -> DATETIME_TYPE:
    
    ''' Convenience function for converting datetime objects from local timezones to utc '''
    
    return (local_datetime - local_datetime.utcoffset()).replace(tzinfo = get_utc_tzinfo())

# .....................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Demo

if __name__ == "__main__":
    
    # Make aligned printing helpers
    print_indent = lambda indent_amount, arg0, *args: print(str(arg0).rjust(indent_amount), *args)
    print10 = lambda arg0, *args: print_indent(10, arg0, *args)
    print16 = lambda arg0, *args: print_indent(16, arg0, *args)
    
    # Get base datetimes for re-use
    local_dt = get_local_datetime()
    utc_dt = get_utc_datetime()
    
    # Print examples
    print("")
    print("*** Base datetimes ***")
    print10("Local DT:", local_dt)
    print10("UTC DT:", utc_dt)
    print("")
    print("*** Local datetime conversions ***")
    print16("Human readable:", datetime_to_human_readable_string(local_dt))
    print16("isoformat:", datetime_to_isoformat_string(local_dt))
    print16("epoch ms:", datetime_to_epoch_ms(local_dt))
    print16("day start:", datetime_convert_to_day_start(local_dt))
    print16("day end:", datetime_convert_to_day_end(local_dt))
    print16("to utc:", local_datetime_to_utc_datetime(local_dt))
    print("")
    print(timestamped_str("End of demo"))
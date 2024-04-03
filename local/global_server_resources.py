#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''
This script does something hacky-ish!
It sets up global variables which are used by the server.

For example, access to the DBServer is handled through a
single instance of a class, which contains all access logic.
If the 'settings' page is used to update the 'dbserver_url',
used by the access variable, then these changes will persist
across to the detection page since they share the underlying
instance of the access variable.

Doing things this way guarantees consistency of behavior across
different routes, which may otherwise be written in separate scripts!
It also avoids re-instantiating variables within each routing script,
like the detection models, which would be wasteful, 
But it does feel hacky...
'''


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from local.environment import get_dbserver_base_url, get_models_path

from local.lib.dbserver_access import DBServerAccess
from local.lib.gdino.make_gdino import GDINOLoader


# ---------------------------------------------------------------------------------------------------------------------
#%% Create globals

# # Get (global!) settings
DBACCESS = DBServerAccess(get_dbserver_base_url())
MODEL_LOADER = GDINOLoader(get_models_path())


# Route response documentation
DBSERVER_ACCESS_ERROR_RESPONSE = {502: {"description": "Error requestig data from dbserver"}}
MODEL_ERROR_RESPONSE = {503: {"description": "Error accessing model"}}

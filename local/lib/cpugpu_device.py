#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import torch
import gc


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def clear_cpugpu_memory() -> bool:
    
    '''
    Helper used to free memory usage (python garbage collect + cuda cache)
    Returns:
        has_cuda (bool)
    '''
    
    gc.collect()
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.empty_cache()
    
    return has_cuda

# .....................................................................................................................

def check_device_availability() -> tuple[bool, str]:
    
    '''
    Helper used to check if a GPU (cuda or mps) is available
    Returns:
        has_gpu, device_string
    '''
    
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    has_gpu = has_cuda or has_mps
    
    fastest_device = "cpu"
    if has_mps:
        fastest_device = "mps"
    if has_cuda:
        fastest_device = "cuda"
    
    return has_gpu, fastest_device

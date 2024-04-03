#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports


import datetime as dt

from collections import OrderedDict

from inspect import isfunction

import gc


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes

class RAMCache:
    
    '''
    Class used to maintain a cache, using some input arguments as keys
    Note: data is only stored in RAM, there is no file IO
    
    Example usage:
        data_cache = RAMCache()
        data_cache.store([1,2,3,4,5], "some", "identifier")
        ...
        is_in_cache, data = data_cache.retrieve("some", "identifier") 
        # -> returns: (True, [1, 2, 3, 4, 5])
    '''
    
    # .................................................................................................................
    
    def __init__(self, max_cache_items = 50):
                
        # Allocate storage for keeping track of access times
        self._last_access_dt = dt.datetime.utcnow()
        
        # Initialize storage
        self._max_cache_items = max_cache_items
        self._storage_dict = OrderedDict()
        self.clear()
        
        # Set the default keymaker function
        self._keymaker = None
        self.set_cache_keymaker_fn(self._default_keymaker)
    
    # .................................................................................................................
    
    def __len__(self):
        ''' Allows len(...) call on cache, which returns number of cached elements '''
        return len(self._storage_dict)
    
    # .................................................................................................................
    
    def __iter__(self):
        ''' Allows cache to be used in a for loop, where it returns all cached data (newest to oldest) '''
        return (value for value in reversed(self._storage_dict.values()))
    
    # .................................................................................................................
    
    def get_last_access_seconds(self) -> int:
        ''' Get the number of seconds elapsed since the last time the cache was used (storage or retrieval) '''
        access_timedelta = dt.datetime.utcnow() - self._last_access_dt
        return int(access_timedelta.total_seconds())
    
    # .................................................................................................................
    
    def set_cache_keymaker_fn(self, custom_keymaker_function):
        
        '''
        Function used to set a custom keymaker function for cache storage. The keys created by the keymaker 
        are used to uniquely indentify (for storage and retrival) the cached data
        The keymaker function should have the form:
            
            def keymaker(arg1, arg2, arg3,...):
                unique_key_made_from_args = ...
                return unique_key_made_from_args
        
        The default keymaker (i.e. if this function isn't used to set a custom keymaker),
        forms an underscore-separated string from the args, eg. "arg1_arg2_arg3_...".
        
        The main reason to use a custom keymaker is if the key arguments, when combined in the default
        (underscore-separated) way would not be unique, or if the keys cannot be converted to strings
        
        Returns self
        '''
        
        # Warn about bad inputs
        if not isfunction(custom_keymaker_function):
            raise TypeError("Must provide a function for cache keymaker!")
        
        self._keymaker = custom_keymaker_function
        
        return self
    
    # .................................................................................................................
    
    def is_stored(self, *key_args) -> bool:
        
        '''
        Function used to check if data is already stored in cache
        Inputs:
            key_args - arguments used to generate cache key (using keymaker function)
        
        Returns:
            is_stored_in_cache (boolean)
        '''
        
        cache_key = self._keymaker(*key_args)
        is_in_storage = cache_key in self._storage_dict.keys()
        
        return is_in_storage
    
    # .................................................................................................................
    
    def store(self, data_to_store, *key_args) -> bool:
        
        '''
        Function used to store data in the cache. If there is already data with the same cache key,
        the new data will NOT be stored (i.e. old data is not overwritten!)
        Inputs:
            data_to_store - data to be stored in cache
            key_args - arguments used to generate a cache key (using the keymaker function)
        
        Returns:
            already_in_storage (boolean)
        '''
        
        # Record access timing
        self._refresh_access_time()
        
        # Build key for storage (bail if we've already stored it!)
        cache_key = self._keymaker(*key_args)
        already_in_storage = cache_key in self._storage_dict.keys()
        if already_in_storage:
            return already_in_storage
        
        # Store data
        self._trim_cache()
        self._storage_dict[cache_key] = data_to_store
        
        return already_in_storage
    
    # .................................................................................................................
    
    def retrieve(self, *key_args):
        
        '''
        Function used to get data out of the cache. If data is retrieved, it's position in the cache is reset
        (i.e. it is considered 'recently used')
        Inputs:
            key_args - arguments used to generate a cache key (using keymaker function)
        
        Returns:
            is_stored_in_cache (boolean), stored_data (data or None)
        '''
        
        # Record access timing
        self._refresh_access_time()
        
        # Try to retrieve entry from cache
        cache_key = self._keymaker(*key_args)
        stored_data = self._storage_dict.get(cache_key, None)
        
        # Refresh position in cache, if the key was valid
        is_in_storage = cache_key in self._storage_dict.keys()
        if is_in_storage:
            self._storage_dict.move_to_end(cache_key, last = True)
        
        return is_in_storage, stored_data
    
    # .................................................................................................................
    
    def delete(self, *key_args) -> bool:
        
        '''
        Function used to manually remove entries from the cache. This normally shouldn't be needed,
        since the cache automatically removes old entries as it fills up. The main use case for
        manual deletion is to be able to replace an entry that would otherwise have the same key
        
        Returns:
            data_existed (True if there was an entry, False if there was nothing to delete)
        '''
        
        # Only remove data if we have a key entry
        cache_key = self._keymaker(*key_args)
        data_existed = cache_key in self._storage_dict.keys()
        if data_existed:
            self._storage_dict.pop(cache_key, None)
            gc.collect()
        
        return data_existed
    
    # .................................................................................................................
    
    def clear(self):
        
        ''' Function which completely wipes out all cache data. Returns self '''
        
        # Reset storage and force garbage collection
        self._storage_dict = OrderedDict()
        gc.collect()
        
        return self
    
    # .................................................................................................................
    
    def _trim_cache(self, trimming_before_storing = True) -> bool:
        
        '''
        Helper which clears the oldest data from cache until it is at it's maximum allowable size
        Inputs:
            trimming_before_storing - If True, the cache will be trimmed to 1 element less than the max
                                      allowable size, to make space for an element to be stored.
                                      The prevents the cache from ever exceeding it's max sizing!
        Returns:
            data_was_removed_from_cache (boolean)
        '''
        
        # Remove an additional cache element if we are trimming before storage
        max_cache_items = self._max_cache_items
        if trimming_before_storing:
            max_cache_items = max_cache_items - 1
        
        data_was_removed_from_cache = False
        while True:
            
            # Bail if we don't need to trim the cache
            need_trim_cache = (len(self._storage_dict) > max_cache_items)
            if not need_trim_cache:
                break
            
            # Remove 'oldest' (i.e. least recent) entry from storage
            self._storage_dict.popitem(last = False)
            data_was_removed_from_cache = True
        
        return data_was_removed_from_cache
    
    # .................................................................................................................
    
    def _refresh_access_time(self) -> None:
        self._last_access_dt = dt.datetime.utcnow()
        return
    
    # .................................................................................................................
    
    @staticmethod
    def _default_keymaker(*key_args) -> str:
        return "_".join((str(each_arg) for each_arg in key_args))

    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

#%% Demo

if __name__ == "__main__":
    
    # Create example cache which stores squared values 0-to-15
    ex = RAMCache(10)
    print("-> Adding numbers 0 to 9 (squared)")
    for k in range(10):
        ex.store(k*k, "item", k)

    # Example getting data out of cache
    item_idx = 0
    ok_item, item_value = ex.retrieve("item", item_idx)
    if ok_item:
        print("Retrieved item {} (value: {})".format(item_idx, item_value))
    
    # Add more data
    print("-> Adding numbers 100 to 105")
    for k in range(6):
        ex.store(100 + k, "other", k)

    # Example iterating over all cached data (note ordering after previous retrival)
    print("", "Showing cache data, top is most recent", sep = "\n")
    for data in ex:
        print(data)

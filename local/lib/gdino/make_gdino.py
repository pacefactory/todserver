#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import os
import os.path as osp

import torch

from local.lib.cpugpu_device import clear_cpugpu_memory, check_device_availability

from local.lib.gdino.gdino_model import GDinoModel
from local.lib.gdino.text_processing import TextEncoder
from local.lib.gdino.image_processing import ImageEncoder
from local.lib.gdino.text_image_transformer import TextAndImageTransformer


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

class GDINOLoader:
    
    _allowable_file_exts = {".pt", ".pth"}
    
    # .................................................................................................................
    
    def __init__(self, models_folder_path, load_on_start = False):
        
        # Create pathing to where GDINO model files are stored
        self._folder_path = osp.join(models_folder_path, "gdino")
        os.makedirs(self._folder_path, exist_ok=True)
        
        # Initialize device settings
        has_gpu, fastest_device_str = check_device_availability()
        self._has_gpu = has_gpu
        self._device_on_load_str = fastest_device_str 
        
        self._model = None
        if load_on_start:
            self.load_active_model()
        
        pass
    
    # .................................................................................................................
    
    def get_model(self) -> tuple[bool, GDinoModel | None]:
        
        # Re-load model if needed
        if self._model is None:
            self.load_active_model()
        
        ok_model = self._model is not None
        return ok_model, self._model
    
    # .................................................................................................................
    
    def get_file_name_to_path_lut(self) -> dict[str, str]:
        
        '''
        Helper function used to build lookup table of file names
        (without extensions) to full file paths
        
        Example:
            {
                "file_a": "/path/to/FILE_a.ext",
                "info": "/a/Pathing/to/a/File/called/info.jpg"
            }
        '''
        
        # Get listing of model files
        files_list = os.listdir(self._folder_path)
        model_files_list = [f for f in files_list if osp.splitext(f)[1] in self._allowable_file_exts]
        
        # Get listing of lowercase filenames (without ext) with mapping to file pathing for output
        model_files_no_ext_list = [self._clean_file_name(f) for f in model_files_list]
        model_paths_list = [osp.join(self._folder_path, f) for f in model_files_list]
        return {fname_no_ext: fpath for fname_no_ext, fpath in zip(model_files_no_ext_list, model_paths_list)}
    
    # .................................................................................................................
    
    def load_model(self, file_name) -> bool:
        
        '''
        Function used to load a model based on a given file name
        Returns True/False if loading succeeds/fails.
        '''
        
        ok_file, model_file_path = self.build_file_path(file_name)
        if ok_file:
            self.unload_model()
            self._model = make_gdino_from_model_path(model_file_path)
            self._model.set_device(self._device_on_load_str)
        
        return ok_file
    
    # .................................................................................................................
    
    def load_active_model(self) -> bool:
        active_file_name = self.get_active_file()
        return self.load_model(active_file_name)
    
    # .................................................................................................................
    
    def unload_model(self) -> None:
        
        ''' Function used to recover resources in use by the model '''
        
        if self._model is not None:
            self._device_on_load_str, _ = self._model.get_device_str()
            self._model.unload_resources()
            self._model = None
            clear_cpugpu_memory()
        
        return
    
    # .................................................................................................................
    
    def get_active_file(self) -> str:
        
        '''
        Function which gets the file name (without ext) of the 'active model file',
        which is used as the default model to load. This is determined based on the
        most recently 'modified' file.
        '''
        
        # Get the model file with the most recent modification time
        file_lut = self.get_file_name_to_path_lut()
        file_names = file_lut.keys()
        models_exist = len(file_names) > 0
        if not models_exist:
            print("", "ERROR! Cannot load active model, no files exist!", sep="\n", flush=True)
        
        get_file_mod_time = lambda file_name: osp.getmtime(file_lut.get(file_name,0))
        active_name = max(file_names, key = get_file_mod_time) if models_exist else None
        
        return active_name
    
    # .................................................................................................................
    
    def set_active_file(self, file_name) -> bool:
        
        '''
        Tags a file as being 'active' so that it will be loaded
        as the default file if the server were to restart.
        
        This is done by updating the file access time!
        '''
        
        file_lut = self.get_file_name_to_path_lut()
        file_path = file_lut.get(file_name, None)
        ok_file = file_path is not None
        if ok_file:
            os.utime(file_path)
            self.load_model(file_name)
        
        return ok_file
    
    # .................................................................................................................
    
    def build_file_path(self, file_name) -> tuple[bool, str]:
        
        '''
        Helper used to build a full file path to a given file
        This works by checking the name against allowable files,
        if the given name isn't valid, a path will not be given
        Returns:
            ok_file, file_path
        '''
        
        target_name = self._clean_file_name(file_name)
        file_lut = self.get_file_name_to_path_lut()
        
        file_path = file_lut.get(target_name, None)
        ok_file = file_path is not None
        
        return ok_file, file_path
    
    # .................................................................................................................
    
    @staticmethod
    def _clean_file_name(file_name) -> str:
        return osp.splitext(str(file_name).lower())[0]
    
    # .................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

def make_gdino_from_model_path(path_to_state_dict, strict_load = True):
    
    '''
    Function used to make a gdino model directly from a path to weights
    Also handles mapping to cpu if weights are in cuda format, but cuda isn't available
    '''
    
    # Load model weights with fail check in case weights are in cuda format and user doesn't have cuda
    try:
        state_dict = torch.load(path_to_state_dict)
    except RuntimeError:
        state_dict = torch.load(path_to_state_dict, map_location="cpu")
    
    return make_gdino_from_state_dict(state_dict, strict_load)

# .....................................................................................................................

def make_gdino_from_state_dict(state_dict, strict_load = True):
    
    '''
    Helper used to create a full gdino model instatiated with the given weights (state dict)
    '''
    
    # Feedback on using non-strict loading
    if not strict_load:
        print("",
              "WARNING:",
              "  Loading model weights without 'strict' mode enabled!",
              "  Some weights may be missing or unused!", sep = "\n", flush = True)
    
    # Get model weights separated by component for loading
    txtenc_sd, imgenc_sd, crossenc_sd = convert_state_dict_keys(state_dict)
    is_tiny_swin = check_is_tiny_swin(imgenc_sd)
    
    # Load weights for model components
    gdino_model = make_gdino(is_tiny_swin)
    gdino_model.txtenc_model.load_state_dict(txtenc_sd, strict_load)
    gdino_model.imgenc_model.load_state_dict(imgenc_sd, strict_load)
    gdino_model.crossenc_model.load_state_dict(crossenc_sd, strict_load)
    
    return gdino_model

# .....................................................................................................................

def make_gdino(use_tiny_swin = True):
    
    ''' Helper used to instantiate a full gdino model, without loading weights '''
    
    # Construct model components
    txtenc_model = TextEncoder()
    imgenc_model = ImageEncoder(use_tiny_swin)
    crossenc_model = TextAndImageTransformer()
    
    # Build combined model!
    gdino_model = GDinoModel(txtenc_model, imgenc_model, crossenc_model)
    
    return gdino_model


# ---------------------------------------------------------------------------------------------------------------------
#%% State dictionary conversions

def check_is_tiny_swin(image_encoder_state_dict):
    
    '''
    Simple hacky helper used to infer which sized swin model we're working with
    This is based on a simple count of the number of model weight entries. The original
    GDino models have 203 keys for the 'tiny' (file name calls this 'swint_ogc') model
    and 371 keys for the 'base' model (file name 'swinb_cogcoor').
    
    This function uses a threshold check on the key count to figure out the model size
    (with a bit of slop in case of modified models)
    '''
    
    swin_tiny_key_count_threshold = 225
    num_weight_keys = len(image_encoder_state_dict.keys())
    is_tiny_swin = num_weight_keys < swin_tiny_key_count_threshold
    
    return is_tiny_swin

# .....................................................................................................................

def convert_state_dict_keys(original_state_dict):
    
    '''
    Helper function used to break the GDino weights into separate groupings,
    based on model component (i.e. text-encoder, image-encoder, output-decoder),
    so that they can be loaded into separate models
    '''
    
    # Set up listing of layer prefixes we're targeting (and the ones we don't need)
    textenc_prefixes = {"bert", "feat_map"}
    imgenc_prefixes = {"backbone", "input_proj"}
    outdec_prefixes = {"bbox_embed", "transformer"}
    unused_key_prefixes = {"bert.pooler", "bert.embeddings.position_ids", "label_enc"}
    
    # Set up fixes for swin keys. The new implementation has slight structural differences
    # -> Original is prefixed 'backbone.0' we need just 'backbone'
    # -> Original norm layers are individually numbered (e.g. 'norm1') we want a list format: 'norm.0'
    fix_backbone_key = lambda k: k.replace("backbone.0", "backbone")
    fix_norm1_key = lambda k: k.replace("backbone.norm1", "backbone.norm.0")
    fix_norm2_key = lambda k: k.replace("backbone.norm2", "backbone.norm.1")
    fix_norm3_key = lambda k: k.replace("backbone.norm3", "backbone.norm.2")
    
    # Setup storage for outputs, by model component
    textenc_state_dict = {}
    imgenc_state_dict = {}
    crossenc_state_dict = {}
    
    # The original model components are bundled under a top-level model key, so get rid of that
    orig_model_state_dict = original_state_dict["model"]
    for k, v in orig_model_state_dict.items():
        
        # The tiny GDino model includes a 'module.' prefix on all keys that we need to get rid of...
        k = k.removeprefix("module.")
        
        # Skip weights we don't need
        ignore_prefix = any(k.startswith(prefix) for prefix in unused_key_prefixes)
        if ignore_prefix:
            # print("DEBUG - Skipping unused key:", k)
            continue
        
        # Handle text-encoder components
        is_textenc = any(k.startswith(prefix) for prefix in textenc_prefixes)
        if is_textenc:
            textenc_state_dict[k] = v
        
        # Handle image-encoder components
        elif any(k.startswith(prefix) for prefix in imgenc_prefixes):
            new_key = fix_backbone_key(k)
            new_key = fix_norm1_key(new_key)
            new_key = fix_norm2_key(new_key)
            new_key = fix_norm3_key(new_key)
            imgenc_state_dict[new_key] = v
        
        # Handle output-decoder components
        elif any(k.startswith(prefix) for prefix in outdec_prefixes):
            crossenc_state_dict[k] = v
        
        # Report any unknown keys (this shouldn't happen!)
        else:
            print("",
                  "WARNING:",
                  "Unrecognized state_dict entry:", k, sep="\n")
        pass
    
    return textenc_state_dict, imgenc_state_dict, crossenc_state_dict

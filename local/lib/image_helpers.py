#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

import base64
import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------------------------------
#%% Classes


# ---------------------------------------------------------------------------------------------------------------------
#%% Functions

# .....................................................................................................................

def image_bytes_to_cvimage(image_bytes, dtype = np.uint8) -> np.ndarray:
    
    image_as_buffer = np.frombuffer(image_bytes, dtype = dtype)
    image = cv2.imdecode(image_as_buffer, cv2.IMREAD_UNCHANGED)
    
    return image

# .....................................................................................................................

def image_to_base64(image_nparray, add_html_data_prefix = True) -> tuple[bool, str | None]:
    
    ok_encode, image_bytes = cv2.imencode(".png", image_nparray)
    b64_str = base64.b64encode(image_bytes).decode("utf-8") if ok_encode else None
    
    if add_html_data_prefix:
        b64_str = ",".join(("data:image/png;base64", b64_str))
    
    return ok_encode, b64_str

# .....................................................................................................................

def base64_to_image(base_64_str) -> tuple[bool, np.ndarray | None]:
    
    # Initialize outputs
    image = None
    ok_decode = False
    
    # Remove html data prefix, if detected
    contains_html_prefix = base_64_str.startswith("data:image")
    if contains_html_prefix:
        _, base_64_str = base_64_str.split(",")
    
    try:
        image_bytes = base64.b64decode(base_64_str.encode("utf-8"))
        image = image_bytes_to_cvimage(image_bytes)
        ok_decode = True
        
    except Exception:
        pass
    
    return ok_decode, image

# .....................................................................................................................


# ---------------------------------------------------------------------------------------------------------------------
#%% Demo

if __name__ == "__main__":
    
    import numpy as np
    
    ex_img = np.random.randint(0,255, (128,256, 3), dtype = np.uint8)
    
    ok_encode, b64_img = image_to_base64(ex_img)
    ok_decode, image_from_b64 = base64_to_image(b64_img)
    
    cv2.imshow("BEFORE (Original image)", ex_img)
    cv2.imshow("AFTER (from base64)", image_from_b64)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
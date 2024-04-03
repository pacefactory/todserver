#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
#%% Imports

from time import perf_counter

from pydantic import BaseModel, Field
from fastapi import APIRouter

from local.lib.response_helpers import server_error_exception, client_error_exception

from local.global_server_resources import DBACCESS, MODEL_LOADER


# ---------------------------------------------------------------------------------------------------------------------
#%% Route request Types

class DetectionRequest(BaseModel):
    camera_select: str = Field(description="Name of camera to run detection on", examples=["ip100"])
    snapshot_ems: int = Field(
        description="Exact epoch ms value of snapshot to use for detection request",
        examples=[946686599007, 1712100090980],
    )
    text_prompts_list: list[str] = Field(
        description="List of names of objects to detect in the snapshot",
        examples=[["person", "forklift"]],
    )
    box_confidence: float = Field(
        default=0.35,
        description="Confidence threshold for detections. Detections below this value will not be returned",
    )
    text_confidence: float = Field(
        default=1.0,
        description="Additional threshold applied after bounding box threshold. This may just be an optimization parameter of the model! Feel free to play with it",
    )


# ---------------------------------------------------------------------------------------------------------------------
#%% Route response types

class DetectionResponse(BaseModel):
    
    text_prompt: str = Field(description="Text prompt that was given to the detection model", examples=["person"])
    boxes_xywh_norm: list[list[float]] = Field(
        description="List of detected bounding boxes in x-center/y-center/width/height format, normalized",
        examples=[[[0.5, 0.5, 0.25, 0.25], [0.7, 0.4, 0.15, 0.35]]],
    )
    labels: list[str] = Field(
        description="List of classification labels associated with each detection",
        examples=[["person", "person"]],
    )
    scores: list[float] = Field(
        description="List of confidence scores associated with each detection",
        examples=[[0.233211, 0.4448367]],
    )
    
    time_taken_ms: int = Field(
        description="Amount of time taken (in milliseconds) to run the detection model",
        examples=[84],
    )
    

# ---------------------------------------------------------------------------------------------------------------------
#%% Routes

detection_router = APIRouter(prefix="/v0/detect" , tags=["Detection"])

# .....................................................................................................................

@detection_router.post("")
def v0_detect_route(post_body: DetectionRequest) -> DetectionResponse:
    
    '''
    Main route of this server!
    Performs text-based object detection on a target snapshot image from the dbserver.
    '''
    
    # For clarity
    camera_select = post_body.camera_select
    snapshot_ems = post_body.snapshot_ems
    text_prompts_list = post_body.text_prompts_list
    box_conf = post_body.box_confidence
    text_conf = post_body.text_confidence
    
    # Warn about missing inputs
    no_prompts = len(text_prompts_list) == 0
    if no_prompts:
        raise client_error_exception(
            "Must provide text prompts for detections!",
            details = {"text_prompts_list": text_prompts_list},
        )
    
    # Get snapshot image from database
    ok_resp, image_bgr = DBACCESS.get_snapshot(camera_select, snapshot_ems)
    if not ok_resp:
        raise server_error_exception(
            "Unable to retrieve snapshot image",
            details={"camera_select": camera_select, "snapshot_ems": snapshot_ems},
            status_code=502,
        )
    
    # Make sure we have a model to use
    ok_model, model = MODEL_LOADER.get_model()
    if not ok_model:
        raise server_error_exception("Error! Model is unavailable", status_code=503)
    
    # Run model (with timing for feedback)
    t1 = perf_counter()
    det_result = model.detect(text_prompts_list, image_bgr, box_conf, text_conf).cpu()
    t2 = perf_counter()
    
    time_taken_ms = round(1000 * (t2 - t1))
    result_as_dict = det_result.as_dict()
    return DetectionResponse(
        text_prompt = result_as_dict["text_prompt"],
        boxes_xywh_norm = result_as_dict["boxes_xywh_norm"],
        labels = result_as_dict["labels"],
        scores = result_as_dict["scores"],
        time_taken_ms = time_taken_ms,
    )

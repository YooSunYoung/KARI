import os
import argparse
import tensorflow as tf
from datetime import datetime

def get_log_path(model_type, backbone="vgg16", custom_postfix=""):
    """Generating log path from model_type value for tensorboard.
    inputs:
        model_type = "rpn", "faster_rcnn"
        backbone = "vgg16", "mobilenet_v2"
        custom_postfix = any custom string for log folder name
    outputs:
        log_path = tensorboard log path, for example: "logs/rpn_mobilenet_v2/{date}"
    """
    return "logs/{}_{}{}/{}".format(model_type, backbone, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))

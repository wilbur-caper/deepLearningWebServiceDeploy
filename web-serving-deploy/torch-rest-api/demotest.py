# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time
import os
import numpy as np

# initialize the Keras REST API endpoint URL along with the input
# image path
# KERAS_REST_API_URL = "http://localhost:6602/predict"
KERAS_REST_API_URL = "http://localhost:40001/private/v1/push_paltoon"

#IMAGE_PATH = "/home/hs/caffedl/hs-serving-master_pos/caffe-rest-api/test2.jpg"

# load the input image and construct the payload for the request
# image = open(IMAGE_PATH, "rb").read()


# payload = {
#     "image": image
# }

payload = {
    "batch_number" : "ailab_shanghai_947759",
    "local_image_path" : "/opt/imgs/test2.jpg"
    }
# r = requests.post(KERAS_REST_API_URL,files=payload).json()
# r = requests.post(KERAS_REST_API_URL,json=payload).json()
r = requests.post(KERAS_REST_API_URL,json=payload).json()
bboxs = []
# ensure the request was sucessful
if r["success"]:
    print("[INFO] thread {} OK".format(1))
    print("r[\"predictions\"]：", r["predictions"])
    # for (i, result) in enumerate(r["predictions"]):
    #     bboxs.append(result["x"])
    #     bboxs.append(result["y"])
    #     bboxs.append(result["xr"])
    #     bboxs.append(result["yr"])
    # print("bboxs:", str(bboxs))
    #     # print("{}: {:.4f}".format(i + 1,  result["probability"]))
    #     print("{}: box_lx:{},box_ly:{},box_rx:{},box_ry:{},probability:{:.4f}".format(i + 1, result["box_lx"], result["box_ly"], result["box_rx"],result["box_ry"],result["probability"]))
    #     # print("{}if_occlusion?:{}".format(i + 1, result["occlusion"]))
    # # print("ss：",r["ss"])

# otherwise, the request failed
else:
    print("sorry request filed")
    print("[INFO] thread {} FAILED".format(1))

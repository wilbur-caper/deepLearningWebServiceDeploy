# import the necessary packages
import numpy as np  
import sys,os  

import settings
import helpers
import redis
import time
import json

import torch
import torch.backends.cudnn as cudnn
import time
from pathlib import Path
from numpy import random
from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import *
import copy, cv2

CLASSES = ('background', 'mono_oos')
# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

class RetailF(object):
    def __init__(self, imgsz=800, weights='', device='CPU', logger=None, vis_img=False, myriad_model_xml = None, num_myriad_model_xml = None):
        self.vis_img = vis_img
        self.logger = logger
        self.iou_thres = 0.5
        self.conf_thres=0.25
        self.agnostic_nms = 0.5
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]
        # self.num_classic = NumClass(device, num_myriad_model_xml)
        # if myriad_model_xml:
        #     # self.run_myriad = True
        #     # self.predictor = myriad_init(myriad_model_xml, device)
        # else:
        # self.run_myriad = False
        # self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(device)
        self.model = attempt_load(weights, map_location=self.device)
        # self.model = attempt_load("weights/yolov5s.pt")  # load FP32 model
        self.model.eval()
        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
    
    def predict(self, org_img):
        img = letterbox(org_img, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms)
        # t2 = torch_utils.time_synchronized()
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]

        # Process detections
        result_list = list()
        for i, det in enumerate(pred):
            s = ''
            # p, s, im0 = path, '', im0s

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], org_img.shape).round()

                # Print results
                # print(det)
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %s, ' % (n, self.classes[int(c)])  # add to string

                det = list(det)
                # for i in det:
                #     i[5] = i[3] - i[1]
                # det.sort(key = self.sort_by_conf, reverse=False)

                for *xyxy, conf, cls in det:
                    # result = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # print(xyxy, self.classes[int(cls)], conf)
                    # result = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]-xyxy[0]),int(xyxy[3]-xyxy[1])]
                    result = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]
                    # result.append(self.classes[int(cls)])
                    result.append(conf)
                    result_list.append(result)
                    # x y x y classes conf

        return result_list
    
    def print_img(self, info, org_img, is_true):
        if self.vis_img:
            try:
                label = '%s' % (self.classes[int(info[4])])
                self.plot_one_box([info[0], info[1], info[0]+info[2], info[1]+info[3]], org_img, label=label, color=self.colors[int(info[4])], line_thickness=1, label_num=int(info[4]), is_true=is_true)
            except:
                pass


# ---------------------------------------------------------
def classify_process():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	# print("* Loading model...")

	detecModel = "/opt/torch-rest-api/best.pt"

	if not os.path.exists(detecModel):
	    print(detecModel + " does not exist")
	    exit()
	retailDetect = RetailF(weights=detecModel, vis_img=True)
	# -----------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------
	# continually pool for new images to classify
	while True:
		try:
			# attempt to grab a batch of images from the database, then
			# initialize the image IDs and batch of images themselves
			queue = db.lrange(settings.IMAGE_QUEUE, 0,
				settings.BATCH_SIZE - 1)
			imageIDs = []
			# batch = None
			batch = []
			# print("hereh")

			# loop over the queue
			for q in queue:
				# deserialize the object and obtain the input image
				# {"id": k,"batch_number":json_data["batch_number"], "local_image_path": json_data["local_image_path"]}
				q = json.loads(q.decode("utf-8"))
				print("**********", q["local_image_path"])
				# image = cv2.imread('/opt/imgs/'+q["image"])
				image = cv2.imread(q["local_image_path"])
				batch.append(image)
				imageIDs.append(q["id"])
			# check to see if we need to process the batch
			if len(imageIDs) > 0:
				results = []
				for i in range(len(batch)):	
					img = batch[i]
					# height, width, _ = img.shape
					# if height <= 2500 or width <= 2500: 
					# 	small_img_hw_length = 600
					# else:
					# 	small_img_hw_length = 1000
					results.append(retailDetect.predict(img))									
					# box, conf = prediction(imageIDs[i], img, small_img_hw_length)
					# results.append((box, conf))
					# results.append((box, conf))
				# loop over the image IDs and their corresponding set of
				# results from our model
				#---------------------------------------------------------------
				#---------------------------------------------------------------
				for (imageID, resultSet) in zip(imageIDs, results):
					# initialize the list of output predictions
					output = []

					# loop over the results and add them to the list of
					# output predictions
					resultSet = np.array(resultSet)
					print(resultSet.shape[0])
					# for (box, label, prob) in resultSet:
					for i in range(resultSet.shape[0]):
						# cv2.rectangle(image, (int(resultSet[i][0]), int(resultSet[i][1])), (int(resultSet[i][2]), int(resultSet[i][3])), (0, 255, 0), 5)
						# confidence = round(float(resultSet[i][4]),1)
						# cv2.putText(img, str(confidence), (int((resultSet[i][0]+resultSet[i][2])/2), int(resultSet[i][3])), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(200,255,155),thickness=3)
						# cv2.imwrite("/opt/imgs/{}".format(str(q["image"])), img) 
						r = {
							"x1": int(resultSet[i][0]), 
							"y1": int(resultSet[i][1]), 
							"width": int(resultSet[i][2])-int(resultSet[i][0]), 
							"height": int(resultSet[i][3])-int(resultSet[i][1]), 
							"prob": float(resultSet[i][4])}
						output.append(r)
					finalResult = {"batch_number":q["batch_number"], "bbox_info":output, "platoonInfo":"1.0.3"}
					# print(output)
					# store the output predictions in the database, using
					# the image ID as the key so we can fetch the results
					db.set(imageID, json.dumps(finalResult))

				# remove the set of images from our queue
				db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

			# sleep for a small amount
			time.sleep(settings.SERVER_SLEEP)
		except Exception as e:
			print(e)
			continue

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
	classify_process()

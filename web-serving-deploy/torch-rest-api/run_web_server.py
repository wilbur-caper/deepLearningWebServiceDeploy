#!/usr/bin/python3
# import the necessary packages
from PIL import Image
import numpy as np
import settings
import helpers
import flask
import redis
import uuid
import time
import json
import io

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)

def prepare_image(image):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	# image = image.resize(target)
	image = np.asarray(image)
	print(image.shape)
	# image = np.expand_dims(image, axis=0)

	# return the processed image
	return image

@app.route("/")
def homepage():
	return "Welcome to the torch REST API!"

@app.route("/private/v1/push_paltoon", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		# print("get post")
		json_data = flask.request.json
		if json_data["batch_number"] and json_data["local_image_path"]:
		# if flask.request.files.get("image"):
			# read the image in PIL format and prepare it for
			# classification
			# image = flask.request.files["image"].read()
			# image = Image.open(io.BytesIO(image))
			# image = prepare_image(image)
			# h,w,c = image.shape

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			# image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			# image = helpers.base64_encode_image(image)
			# d = {"id": k, "image": image, "width": w, "height":h}
			d = {"id": k,"batch_number":json_data["batch_number"], "local_image_path": json_data["local_image_path"]}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

			# keep looping until our model server returns the output
			# predictions
			while True:
				try:
					# attempt to grab the output predictions
					output = db.get(k)

					# check to see if our model has classified the input
					# image
					if output is not None:
						# add the output predictions to our data
						# dictionary so we can return it to the client
						output = output.decode("utf-8")
						data["predictions"] = json.loads(output)

						# delete the result from the database and break
						# from the polling loop
						db.delete(k)
						break

					# sleep for a small amount to give the model a chance
					# to classify the input image
					time.sleep(settings.CLIENT_SLEEP)
				except:
					continue
			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# for debugging purposes, it's helpful to start the Flask testing
# server (don't use this for production
if __name__ == "__main__":
	# print("* Starting web service...")
	app.run('127.0.0.1', port=40001)

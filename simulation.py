import base64
from io import BytesIO

import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model
import utils

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 5

speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):

    steering_angle = data["steering_angle"]
    throttle = float(data["throttle"])
    speed = float(data["speed"])
    imgString = data["image"]
    
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    image_array = np.asarray(image)
    image_array = utils.crop(image_array)
    image_array = utils.resize(image_array)
    image_array = image_array/127.5-1.0
    transformed_image_array = image_array[None, :, :, :]

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    global speed_limit
    if speed > speed_limit:
        speed_limit = MIN_SPEED  # slow down
    else:
        speed_limit = MAX_SPEED
    throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

    print('{:.5f}, {:.1f}'.format(steering_angle, throttle))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':

    model = load_model('./model_new.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
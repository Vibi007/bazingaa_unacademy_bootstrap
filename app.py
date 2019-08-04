import os
from flask import Flask, render_template, Response
from keras.models import load_model
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from camera import VideoCamera
# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
import cv2
import tensorflow as tf

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
class_names = ['person']
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
app = Flask(__name__)


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

global graph
graph = tf.get_default_graph()


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    # cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        # for skimagee in camera.get_frame():
        for skimagee in camera.get_frame():
        # Handles the mirroring of the current frame
        # skimagee = cv2.flip(skimagee,1)
            skimagee = skimagee[:, :, ::-1]

            segImg = get_frame(skimagee)

            ret, segImg = cv2.imencode('.jpg', segImg)

            segImg = segImg.tobytes()
            yield (b'--frame\r\n' 
                b'Content-Type: image/jpeg\r\n\r\n' + segImg + b'\r\n\r\n')
        
        # skimagee = skimagee.tobytes()
        # yield (b'--frame\r\n'
        #     b'Content-Type: image/jpeg\r\n\r\n' + skimagee + b'\r\n')

        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()

def gen1():
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, skimagee = cap.read()
        # Handles the mirroring of the current frame
        # frame = cv2.flip(frame,1)
        skimagee = frame[:, :, ::1]
        # skimagee = xy_to_binary2d(skimagee)
        # print("frame dtype 1: " + str((frame.shape)))

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # print("frame dtype 2: " + str((frame.shape)))
        # skimagee = skimage.img_as_float(frame)
        # print("skimage dtype 1: " + str((skimagee.shape)))
        # get_frame(frame)

        # segImg = get_frame(skimagee)
        # segImg = segImg.tobytes()
        # yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + segImg + b'\r\n\r\n')
        
        skimagee = skimagee.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + skimagee + b'\r\n\r\n')

        # Saves image of the current frame in jpg file
        # name = 'frame_test' + '.jpg'
        # cv2.imwrite(name, frame)

        # Display the resulting frame
        # cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # To stop duplicate images
        # currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    # while True:
    #     frame = camera.get_frame()
    #     ret, jpeg = cv2.imencode('.jpg', frame)
    #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # cv2.imshow('frame',gray)
    #     jpeg = get_frame(jpeg)
    #     # jpeg = get_frame(frame)
    #     # yield(frame1)
    #     jpeg = jpeg.tobytes()

    #     yield (b'--frame\r\n'
    #         b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')

def get_frame(skimagee):
    # print("skimage dtype 2: " + str(type(skimagee)))
    with graph.as_default():
        r = model.detect([skimagee], verbose=0)
        frame_cropped = segment(skimagee, r[0])
        return frame_cropped

#   vcapture  = cv2.VideoCapture('/content/small.mp4')
# #   s,i = vcapture.read()
# #   height , width =  img.shape
#   width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#   height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#   fps = vcapture.get(cv2.CAP_PROP_FPS)
  # Define codec and create video writer
#   file_name = "UnAcademy_test.avi"
        # vwriter = cv2.VideoWriter(file_name,
        #                             cv2.VideoWriter_fourcc(*'MJPG'),
        #                             fps, (width,height))
#   count = 0
#   success = True
#   while success:
#   while count < 10:
    #   print("frame: ", count)
      
    #   success, image = vcapture.read()
    # if image:
    #   if success:
        # rows,cols = image.shape[:2]
        # M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        # image = cv2.warpAffine(image,M,(cols,rows))
          
#                 # OpenCV returns images as BGR, convert to RGB
        # for image1 in image:
            # image1 = image1[..., ::-1]
            # Detect objects
            # print(image)
            
            # r = model.detect([image1], verbose=0)
            # Color splash
            # frame_cropped = segment(image1, r)
            # blob = cv2.dnn.blobFromImage(frame_cropped, swapRB=True, crop=False)
            # return frame_cropped
        # yield frame_cropped
        
          # Add image to video writer
        # vwriter.write(segmentation)
#           vwriter.write(image)
        #   count += 1
#   vwriter.release()
#   print("Saved to ", file_name)
def segment(image, r):
  idx = r['scores'].argmax()
  mask = r['masks'][:,:,idx]
  mask = np.stack((mask,)*3, axis=-1)
  mask = mask.astype('uint8')
  bg = 255 - mask * 255
  mask_img = image*mask
  result = mask_img+ bg
  return result

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
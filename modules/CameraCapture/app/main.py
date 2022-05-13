# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import os
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

from azure.iot.device import IoTHubModuleClient, Message

import CameraCapture
from CameraCapture import CameraCapture
import VideoStream
from VideoStream import VideoStream


# global counters
SEND_CALLBACKS = 0

def send_to_Hub_callback(strMessage):
    message = Message(bytearray(strMessage, 'utf8'))
    hubManager.send_message_to_output(message, "output1")

# Callback received when the message that we're forwarding is processed.

class HubManager(object):

    def __init__(
            self,
            messageTimeout,
            verbose):
        '''
        Communicate with the Edge Hub

        :param int messageTimeout: the maximum time in milliseconds until a message times out. The timeout period starts at IoTHubClient.send_event_async. By default, messages do not expire.
        :param IoTHubTransportProvider protocol: Choose HTTP, AMQP or MQTT as transport protocol.  Currently only MQTT is supported.
        :param bool verbose: set to true to get detailed logs on messages
        '''
        self.messageTimeout = messageTimeout
        self.client = IoTHubModuleClient.create_from_edge_environment()
        #self.client.set_option("messageTimeout", self.messageTimeout)
        #self.client.set_option("product_info", "edge-camera-capture")
        #if verbose:
        #    self.client.set_option("logtrace", 1)  # enables MQTT logging

    def send_message_to_output(self, event, outputQueueName):
        self.client.send_message_to_output(event, outputQueueName)
        global SEND_CALLBACKS
        SEND_CALLBACKS += 1

def __IsInt(string):
    try: 
        int(string)
        return True
    except ValueError:
        return False


def runDetect(model: str, videoPath: str, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, showVideo: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    videoPath: The camera id/path to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()
  vs = None
  isWebcam = False

  with CameraCapture(showVideo) as cameraCapture:
    cameraCapture.start()

  if (__IsInt(videoPath)):
    isWebcam = True
    vs = VideoStream(int(videoPath), width, height).start()
    time.sleep(1.0)#needed to load at least one frame into the VideoStream class
  else:
    cap = cv2.VideoCapture(videoPath)
  # Start capturing video input from the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while True:
    image = None
    if isWebcam:
        image = vs.read()
    else:
        image = cap.read()[1]
    if (image is None):
        if (not isWebcam):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            counter = 0
            continue
        sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    
    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    for detection in detection_result.detections:
      category = detection.classes[0]
      class_name = category.class_name
      probability = round(category.score, 2)
      result_text = class_name + ' (' + str(probability) + ')'
      print(result_text)

    cameraCapture.put_display_frame(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    print(fps_text)
    print(counter)
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break

#  if (__IsInt(videoPath)):
#    cameraCapture.imageServer.close()
#    cap.release()

def main(
    debugy = False,
    model = "",
    videoPath="0",
    frameWidth = 0,
    frameHeight = 0,
    numThreads = 0,
    enableEdgeTPU = False,
    showVideo = False,
    verbose = False,
    bypassIot = False
):
    #if debugy:
    #    print("Wait for debugger!!!")
    #    import debugpy
    #    debugpy.listen(5678)
    #    debugpy.wait_for_client()  # blocks execution until client is attached
    '''
    Capture a camera feed, send it to processing and forward outputs to EdgeHub

    :param str model: model file. Example: "efficientdet_lite0.tflite".
    :param int cameraId: camera device path such as /dev/video0 or a test video file such as /TestAssets/myvideo.avi. /dev/video0 by default ("0")
    '''
    try:
        print("\nPython %s\n" % sys.version)
        print("Camera Capture Azure IoT Edge Module. Press Ctrl-C to exit.")
        print("Initialising the camera capture with the following parameters: ")
        print("   - Model file: " + model)
        print("   - Video path: " + videoPath)
        print("   - Frame width: " + str(frameWidth))
        print("   - Frame height: " + str(frameHeight))
        print("   - Num Threads: " + str(numThreads))
        print("   - Enable TPU: " + str(enableEdgeTPU))
        print("   - Show video: " + str(showVideo))
        print("   - Verbose: " + str(verbose))
        print("   - Send processing results to hub: " + str(bypassIot))
        print()
        try:
            if not bypassIot:
                global hubManager
                hubManager = HubManager(
                    10000, verbose)
        except Exception as iothub_error:
            print("Unexpected error %s from IoTHub" % iothub_error)
            return

        runDetect(model,videoPath,frameWidth, frameHeight, numThreads,enableEdgeTPU, showVideo)

    except KeyboardInterrupt:
        print("Camera capture module stopped")

def __convertStringToBool(env):
    if env in ['True', 'TRUE', '1', 'y', 'YES', 'Y', 'Yes']:
        return True
    elif env in ['False', 'FALSE', '0', 'n', 'NO', 'N', 'No']:
        return False
    else:
        raise ValueError('Could not convert string to bool.')


if __name__ == '__main__':
  try:
    DEBUGY = __convertStringToBool(os.getenv('DEBUG', 'False'))
    MODEL = os.getenv('MODEL', "efficientdet_lite0.tflite")
    VIDEO_PATH = os.getenv('VIDEO_PATH', "./AppleAndBanana.mp4")
#    VIDEO_PATH = os.getenv('VIDEO_PATH', "0")
    FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', 640))
    FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', 480))
    NUM_THREADS = int(os.getenv('NUM_THREADS', 4))
    ENABLE_TPU = __convertStringToBool(os.getenv('ENABLE_TPU', 'False'))
    SHOW_VIDEO = __convertStringToBool(os.getenv('SHOW_VIDEO', 'True'))
    VERBOSE = __convertStringToBool(os.getenv('VERBOSE', 'False'))
    BYPASS_IOT = __convertStringToBool(os.getenv('BYPASS_IOT', 'True'))

  except ValueError as error:
    print(error)
    sys.exit(1)

main(DEBUGY, MODEL, VIDEO_PATH, FRAME_WIDTH, FRAME_HEIGHT, NUM_THREADS, ENABLE_TPU,
      SHOW_VIDEO, VERBOSE, BYPASS_IOT)

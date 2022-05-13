#Imports
from typing import List

import cv2

import numpy as np
from tflite_support.task import processor

import ImageServer
from ImageServer import ImageServer

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


class CameraCapture(object):

    def __init__(
            self,
            showVideo = False
            ):
        self.showVideo = showVideo

        self.displayFrame = None
        if self.showVideo:
            self.imageServer = ImageServer(5012, self)
            self.imageServer.start()

    def __enter__(self):
        return self

    def get_display_frame(self):
        return self.displayFrame

    def put_display_frame(self, image, detection_result: processor.DetectionResult):
        if self.showVideo:
            for detection in detection_result.detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

                # Draw label and score
                category = detection.classes[0]
                class_name = category.class_name
                probability = round(category.score, 2)
                result_text = class_name + ' (' + str(probability) + ')'
                text_location = (_MARGIN + bbox.origin_x,
                                _MARGIN + _ROW_SIZE + bbox.origin_y)
                cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
            self.displayFrame =  cv2.imencode('.jpg', image)[1].tobytes()

    def __exit__(self, exception_type, exception_value, traceback):
        if self.showVideo:
            self.imageServer.close()

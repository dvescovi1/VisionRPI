#Imports
from typing import List

import cv2
import time

import ImageServer
from ImageServer import ImageServer

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

    def put_display_frame(self, image):
        self.displayFrame = image

    def start(self):
        frameCounter = 0
        perfForOneFrameInMs = None
        while False:
            frameCounter +=1

            #Display frames
            if self.showVideo:
                try:
                    if self.nbOfPreprocessingSteps == 0:
                        if self.verbose and (perfForOneFrameInMs is not None):
                            cv2.putText(frame, "FPS " + str(round(1000/perfForOneFrameInMs, 2)),(10, 35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255), 2)
                        if self.annotate:
                            #TODO: fix bug with annotate function
                            self.__annotate(frame, response)
                        self.displayFrame = cv2.imencode('.jpg', frame)[1].tobytes()
                    else:
                        if self.verbose and (perfForOneFrameInMs is not None):
                            cv2.putText(preprocessedFrame, "FPS " + str(round(1000/perfForOneFrameInMs, 2)),(10, 35),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255), 2)
                        self.displayFrame = cv2.imencode('.jpg', preprocessedFrame)[1].tobytes()
                except Exception as e:
                    print("Could not display the video to a web browser.") 
                    print('Excpetion -' + str(e))
                if self.verbose:
                    if 'startDisplaying' in locals():
                        print("Time to display frame: " + self.__displayTimeDifferenceInMs(time.time(), startDisplaying))
                    elif 'startSendingToEdgeHub' in locals():
                        print("Time to display frame: " + self.__displayTimeDifferenceInMs(time.time(), startSendingToEdgeHub))
                    else:
                        print("Time to display frame: " + self.__displayTimeDifferenceInMs(time.time(), startEncodingForProcessing))
                perfForOneFrameInMs = int((time.time()-startOverall) * 1000)
                if not self.isWebcam:
                    waitTimeBetweenFrames = max(int(1000 / self.capture.get(cv2.CAP_PROP_FPS))-perfForOneFrameInMs, 1)
                    print("Wait time between frames :" + str(waitTimeBetweenFrames))
                    if cv2.waitKey(waitTimeBetweenFrames) & 0xFF == ord('q'):
                        break

            if self.verbose:
                perfForOneFrameInMs = int((time.time()-startOverall) * 1000)
                print("Total time for one frame: " + self.__displayTimeDifferenceInMs(time.time(), startOverall))

    def __exit__(self, exception_type, exception_value, traceback):
        if self.showVideo:
            self.imageServer.close()
            cv2.destroyAllWindows()

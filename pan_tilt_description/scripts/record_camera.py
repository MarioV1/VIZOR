#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_camera.py — PTZ camera video recorder
=============================================
Shows the live camera feed and records to MP4 on demand.

Keys
----
  r     Start / stop recording
  ESC / q   Quit
"""

import os
import threading
import time
from datetime import datetime

import rospy
import rospkg
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

FPS        = 30
RESOLUTION = (1280, 720)


class CameraRecorder:

    def __init__(self):
        rospy.init_node("camera_recorder", anonymous=True)

        self.bridge    = CvBridge()
        self.image     = None
        self.lock      = threading.Lock()

        self.recording = False
        self.writer    = None

        # Save to pan_tilt_description/videos if available, else home
        rospack = rospkg.RosPack()
        try:
            pkg_path = rospack.get_path("pan_tilt_description")
        except Exception:
            pkg_path = os.path.expanduser("~")
        self.out_dir = os.path.join(pkg_path, "videos")
        os.makedirs(self.out_dir, exist_ok=True)

        rospy.Subscriber("/datavideo/video", Image, self._img_cb)

        cv2.namedWindow("Camera Recorder", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Recorder", 1280, 720)

        print("\n📷  Camera Recorder")
        print("   r       →  start / stop recording")
        print("   ESC/q   →  quit\n")

        self._run()

    def _img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.image = frame
                if self.recording and self.writer is not None:
                    self.writer.write(frame)
        except Exception:
            pass

    def _start_recording(self):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        path     = os.path.join(self.out_dir, f"recording_{ts}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer    = cv2.VideoWriter(path, fourcc, FPS, RESOLUTION)
        self.recording = True
        print(f"⏺  Recording started → {path}")

    def _stop_recording(self):
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        print("⏹  Recording stopped")

    def _run(self):
        rate = rospy.Rate(FPS)
        while not rospy.is_shutdown():
            with self.lock:
                img = self.image.copy() if self.image is not None else None

            if img is not None:
                display = img.copy()
                # Red border while recording (display only)
                if self.recording:
                    cv2.rectangle(display, (0, 0), (RESOLUTION[0]-1, RESOLUTION[1]-1),
                                  (0, 0, 220), 6)
                # Optical centre cross (display only, not recorded)
                cv2.drawMarker(display, (643, 388), (255, 0, 255),
                               cv2.MARKER_CROSS, 30, 2)
                cv2.imshow("Camera Recorder", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord("r"):
                if self.recording:
                    self._stop_recording()
                else:
                    self._start_recording()

            rate.sleep()

        if self.recording:
            self._stop_recording()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        CameraRecorder()
    except rospy.ROSInterruptException:
        pass

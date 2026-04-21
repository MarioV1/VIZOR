#!/usr/bin/env python3
"""
zoom_sweep.py
=============
Shows a clean live camera feed and optionally sweeps from ×1 to ×20 in one
continuous call.  No overlay text on the video window.

Keyboard controls (window must be focused):
  s  : start automatic sweep ×1 → ×20
  1  : jump to zoom ×1
  +  : zoom in  one step
  -  : zoom out one step
  q  : quit
"""

import rospy
import cv2
import threading

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from zoom_msgs.srv import SetZoom

bridge      = CvBridge()
latest_frame = None
frame_lock   = threading.Lock()

def image_cb(msg):
    global latest_frame
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    with frame_lock:
        latest_frame = frame.copy()

def set_zoom(level):
    rospy.wait_for_service('/set_zoom', timeout=5.0)
    svc = rospy.ServiceProxy('/set_zoom', SetZoom)
    svc(level)

def sweep():
    rospy.loginfo("Sweep started: ×1 → ×20")
    for z in range(1, 21):
        if rospy.is_shutdown():
            break
        rospy.loginfo(f"  Zoom ×{z}")
        set_zoom(z)
        rospy.sleep(1.5)
    rospy.loginfo("Sweep complete")

def main():
    rospy.init_node('zoom_sweep', anonymous=False)
    rospy.Subscriber('/datavideo/video', Image, image_cb, queue_size=1,
                     buff_size=2**24)
    rospy.sleep(1.0)

    current_zoom = 1
    set_zoom(current_zoom)

    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Feed', 1280, 720)

    while not rospy.is_shutdown():
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is not None:
            cv2.imshow('Camera Feed', frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            t = threading.Thread(target=sweep, daemon=True)
            t.start()
        elif key == ord('1'):
            current_zoom = 1
            set_zoom(current_zoom)
        elif key == ord('+') or key == ord('='):
            current_zoom = min(20, current_zoom + 1)
            set_zoom(current_zoom)
        elif key == ord('-'):
            current_zoom = max(1, current_zoom - 1)
            set_zoom(current_zoom)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orb_correct_tracking.py
=======================
Live PTZ zoom corrector — ORB, continuous tracking.

Correct approach (same as orb_correct_once):
  1. User draws a ROI around the target at zoom ×1.
  2. The ROI crop is saved as the reference template.
  3. Press 'z' → system applies the optimal zoom level.
  4. Every TRACK_INTERVAL seconds: ORB finds the template IN the zoomed frame,
     measures pixel error vs active cursor, converts to angles, sends correction.
  5. Press 'z' again to stop tracking.

Keyboard:
  Draw ROI   : left-click drag
  s          : save ROI crop as template
  z          : start / stop zoom + continuous tracking
  r          : reset
  1 / 2 / 3  : select cursor (image centre / camera centre / extra point)
  0          : show all cursors
  ESC        : quit
"""

import rospy
import cv2
import numpy as np
import os
import subprocess
import threading
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pan_tilt_msgs.msg import PanTiltCmdDeg
import rospkg

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720

CURSORS = {
    1: (640, 360),
    2: (643, 415),
    3: (643, 387),
}
CURSOR_LABELS = {1: "Centro imagen", 2: "Centro camara", 3: "Punto extra"}
CURSOR_COLORS = {1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 0, 255)}

ZOOM_FACTORES = {
    1.0: (1.00, 1.00), 2.0: (1.12, 1.10), 3.0: (1.28, 1.25), 4.0: (1.39, 1.33),
    5.0: (1.61, 1.53), 6.0: (1.74, 1.67), 7.0: (1.98, 1.89), 8.0: (2.19, 2.10),
    9.0: (2.59, 2.43), 10.0: (3.09, 2.84), 11.0: (3.64, 3.25), 12.0: (4.37, 3.79),
    13.0: (6.29, 5.15), 14.0: (7.95, 6.48), 15.0: (9.89, 7.95), 16.0: (12.16, 9.35),
    17.0: (14.77, 10.86), 18.0: (16.82, 12.10), 19.0: (18.23, 12.85), 20.0: (18.56, 13.00),
}
ZOOM_FOVS = {
    1.0: (63.7, 35.84), 2.0: (56.9, 31.2), 3.0: (50.7, 27.3), 4.0: (45.9, 24.5),
    5.0: (40.5, 21.6), 6.0: (37.4, 19.6), 7.0: (32.2, 17.2), 8.0: (29.1, 15.2),
    9.0: (25.3, 13.0), 10.0: (21.7, 11.1), 11.0: (18.3, 9.3), 12.0: (15.2, 7.7),
    13.0: (10.0, 6.2), 14.0: (7.8, 4.8), 15.0: (6.2, 3.6), 16.0: (5.2, 2.9),
    17.0: (4.1, 2.3), 18.0: (3.5, 1.9), 19.0: (2.9, 1.7), 20.0: (2.3, 1.3)
}

LOWE_RATIO     = 0.75
MIN_INLIERS    = 6
STABILISE_S    = 1.5
TRACK_INTERVAL = 0.5   # seconds between corrections
DEAD_BAND_PX   = 3.0   # pixel error below which no command is sent


# ── ORB: find ROI template centre inside a scene ─────────────────────────────

def find_template_centre(template_gray, scene_gray):
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(scene_gray, None)

    if des1 is None or des2 is None:
        return None
    if len(kp1) < MIN_INLIERS or len(kp2) < MIN_INLIERS:
        return None

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_INLIERS:
        return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None

    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < MIN_INLIERS:
        return None

    th, tw    = template_gray.shape
    centre    = np.float32([[tw / 2, th / 2]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(centre, H)
    cx = float(projected[0, 0, 0])
    cy = float(projected[0, 0, 1])

    return cx, cy, n_inliers, H


# ── Main node ─────────────────────────────────────────────────────────────────

class OrbCorrectTracking:

    def __init__(self):
        rospy.init_node('orb_correct_tracking', anonymous=True)

        self.bridge     = CvBridge()
        self.image      = None
        self.image_lock = threading.Lock()

        self.drawing        = False
        self.start_x        = self.start_y = -1
        self.current_x      = self.current_y = -1
        self.roi            = None
        self.roi_selected   = False
        self.template_gray  = None
        self.template_saved = False
        self.optimal_zoom   = 1.0
        self.current_zoom   = 1.0

        self.cursor_sel     = 1
        self.is_busy        = False   # initial setup in progress
        self.tracking       = False   # continuous loop active
        self._stop_track    = threading.Event()

        # Overlay telemetry
        self.last_target_px = None
        self.last_error_px  = None
        self.last_inliers   = 0
        self.consec_fail    = 0

        self.status = "Draw a ROI around the target, then press 's'"

        rospack  = rospkg.RosPack()
        pkg_path = rospack.get_path("pan_tilt_description")
        self.image_dir = os.path.join(pkg_path, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.pub_cmd = rospy.Publisher('/pan_tilt_cmd_deg', PanTiltCmdDeg, queue_size=1)
        rospy.Subscriber('/datavideo/video', Image, self._img_cb)

        cv2.namedWindow("ORB Correct Tracking")
        cv2.setMouseCallback("ORB Correct Tracking", self._mouse_cb)

        print("\n🟢 orb_correct_tracking started")
        print("   Draw ROI → 's' save template → 'z' start/stop tracking")
        print("   1/2/3 cursor   0 all   r reset   ESC quit\n")

        self._main_loop()

    # ── ROS ───────────────────────────────────────────────────────────────────

    def _img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.image = frame
        except Exception as e:
            rospy.logwarn(f"Image error: {e}")

    def _get_frame(self):
        with self.image_lock:
            return self.image.copy() if self.image is not None else None

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        if self.is_busy or self.tracking:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_x, self.current_y = x, y
            self.roi_selected   = False
            self.template_saved = False
            self.template_gray  = None
            self.last_target_px = self.last_error_px = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x, self.current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x0 = min(self.start_x, x)
            y0 = min(self.start_y, y)
            w  = abs(x - self.start_x)
            h  = abs(y - self.start_y)
            if w > 10 and h > 10:
                self.roi          = (x0, y0, w, h)
                self.roi_selected = True
                self.optimal_zoom = self._best_zoom(w, h)
                self.status = (f"ROI {w}×{h}px → zoom ×{self.optimal_zoom}. "
                               f"Press 's' to save template")
                print(f"📐 ROI {w}×{h}px  optimal zoom ×{self.optimal_zoom}")

    # ── Zoom selection ────────────────────────────────────────────────────────

    def _best_zoom(self, w, h):
        for zoom in sorted(ZOOM_FACTORES.keys(), reverse=True):
            fx, fy = ZOOM_FACTORES[zoom]
            if w * fx <= IMG_W * 0.9 and h * fy <= IMG_H * 0.9:
                return zoom
        return 1.0

    # ── PTZ ───────────────────────────────────────────────────────────────────

    def _set_zoom(self, level):
        subprocess.call(f'rosservice call /set_zoom "level: {int(level)}"', shell=True)

    def _read_pose(self):
        out = subprocess.getoutput("rostopic echo -n 1 /pan_tilt_status")
        yaw = pitch = 0.0
        for line in out.splitlines():
            if 'yaw_now:' in line and 'yaw_now_' not in line:
                try: yaw   = float(line.split('yaw_now:')[1].strip())
                except: pass
            if 'pitch_now:' in line and 'pitch_now_' not in line:
                try: pitch = float(line.split('pitch_now:')[1].strip())
                except: pass
        return yaw, pitch

    def _send_cmd(self, yaw, pitch, speed=20):
        cmd = PanTiltCmdDeg()
        cmd.yaw, cmd.pitch, cmd.speed = yaw, pitch, speed
        self.pub_cmd.publish(cmd)

    # ── Correction from pixel error ───────────────────────────────────────────

    def _pixel_error_to_angles(self, ex, ey, zoom):
        fov_h, fov_v = ZOOM_FOVS[zoom]
        delta_yaw   = -(ex / (IMG_W / fov_h))
        delta_pitch =  (ey / (IMG_H / fov_v))
        return delta_yaw, delta_pitch

    # ── Tracking thread ───────────────────────────────────────────────────────

    def _tracking_thread(self):
        zoom = self.optimal_zoom
        try:
            # 1. Go to ×1, snapshot template
            self.status = "Returning to zoom ×1…"
            self._set_zoom(1)
            time.sleep(0.8)
            frame1 = self._get_frame()
            if frame1 is None:
                self.status = "❌ No frame at ×1"; return

            x, y, w, h    = self.roi
            template_gray = cv2.cvtColor(frame1[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            print(f"📸 Template: {w}×{h}px")

            # 2. Apply target zoom
            self.status = f"Applying zoom ×{zoom}…"
            print(f"🔍 Zoom ×{zoom}")
            self._set_zoom(zoom)
            time.sleep(STABILISE_S)
            self.is_busy = False  # setup done

            self.consec_fail = 0
            print(f"🔄 Tracking started  interval={TRACK_INTERVAL}s  "
                  f"deadband={DEAD_BAND_PX}px")

            # 3. Continuous correction loop
            while not self._stop_track.is_set() and not rospy.is_shutdown():
                t0 = time.time()

                frame_z = self._get_frame()
                if frame_z is None:
                    time.sleep(TRACK_INTERVAL)
                    continue

                scene_gray = cv2.cvtColor(frame_z, cv2.COLOR_BGR2GRAY)
                result     = find_template_centre(template_gray, scene_gray)

                cx_cursor, cy_cursor = CURSORS[self.cursor_sel]

                if result is None:
                    self.consec_fail += 1
                    self.status = f"⚠️ ORB failed ({self.consec_fail} consecutive)"
                    rospy.logwarn_throttle(5.0, "ORB failed during tracking")
                else:
                    cx_found, cy_found, n_inliers, H = result
                    self.last_target_px = (cx_found, cy_found)
                    self.last_inliers   = n_inliers
                    self.consec_fail    = 0

                    ex = cx_found - cx_cursor
                    ey = cy_found - cy_cursor
                    self.last_error_px = (ex, ey)

                    if abs(ex) > DEAD_BAND_PX or abs(ey) > DEAD_BAND_PX:
                        d_yaw, d_pitch = self._pixel_error_to_angles(ex, ey, zoom)
                        yaw_now, pitch_now = self._read_pose()
                        self._send_cmd(yaw_now + d_yaw, pitch_now + d_pitch)
                        self.status = (f"🔄 Tracking  err=({ex:+.0f},{ey:+.0f})px  "
                                       f"Δyaw={d_yaw:+.3f}°  Δpitch={d_pitch:+.3f}°  "
                                       f"inliers={n_inliers}")
                    else:
                        self.status = (f"✅ On-target  err=({ex:+.0f},{ey:+.0f})px  "
                                       f"inliers={n_inliers}")

                elapsed = time.time() - t0
                time.sleep(max(0.0, TRACK_INTERVAL - elapsed))

        except Exception as e:
            self.status = f"❌ Tracking error: {e}"
            rospy.logerr(f"Tracking error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.tracking = False
            self.is_busy  = False
            print("⏹ Tracking stopped")
            self.status = "Stopped. Press 'z' to restart or 'r' to reset."

    def _start_tracking(self):
        self._stop_track.clear()
        self.tracking     = True
        self.is_busy      = True
        self.current_zoom = self.optimal_zoom
        self.last_target_px = self.last_error_px = None
        threading.Thread(target=self._tracking_thread, daemon=True).start()

    def _stop_tracking(self):
        self._stop_track.set()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_overlay(self, img):
        if self.drawing:
            cv2.rectangle(img,
                          (self.start_x, self.start_y),
                          (self.current_x, self.current_y),
                          (0, 255, 255), 1)

        if self.roi_selected and self.roi:
            x, y, w, h = self.roi
            color = (0, 0, 255) if self.tracking else \
                    (0, 255, 0) if self.template_saved else (0, 255, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.drawMarker(img, (x+w//2, y+h//2),
                           (255,165,0), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, f"Zoom ×{self.optimal_zoom}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Where target is in zoomed frame
        if self.last_target_px is not None:
            tx, ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.drawMarker(img, (tx, ty), (0,165,255), cv2.MARKER_CROSS, 24, 3)
            cv2.putText(img, f"Target ({tx},{ty})", (tx+8, ty-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)

        # Error arrow: target → cursor
        if self.last_target_px is not None and self.last_error_px is not None:
            cx_c, cy_c = CURSORS[self.cursor_sel]
            tx, ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.arrowedLine(img, (tx, ty), (cx_c, cy_c),
                            (255,255,0), 2, tipLength=0.15)
            ex, ey = self.last_error_px
            cv2.putText(img, f"err ({ex:+.0f},{ey:+.0f})px",
                        (cx_c+8, cy_c+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        # Tracking indicator dot
        if self.tracking:
            cv2.circle(img, (IMG_W-30, 30), 12, (0,0,220), -1)
            cv2.putText(img, "REC", (IMG_W-60, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,220), 1)

        # Cursors
        for k, (cx, cy) in CURSORS.items():
            if self.cursor_sel in (0, k):
                cv2.drawMarker(img, (cx, cy), CURSOR_COLORS[k],
                               cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img, CURSOR_LABELS[k], (cx+10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CURSOR_COLORS[k], 1)

        cv2.putText(img, self.status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(img,
                    f"Zoom: ×{self.current_zoom}  |  Cursor: {self.cursor_sel}  "
                    f"|  s=save  z=start/stop  r=reset  ESC=quit",
                    (10, IMG_H-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _main_loop(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            frame = self._get_frame()
            if frame is not None:
                vis = frame.copy()
                self._draw_overlay(vis)
                cv2.imshow("ORB Correct Tracking", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                self._stop_tracking()
                break

            elif key == ord('r'):
                self._stop_tracking()
                time.sleep(0.3)
                self.roi = None
                self.roi_selected = self.template_saved = False
                self.template_gray = None
                self.last_target_px = self.last_error_px = None
                self.status = "Reset. Draw a new ROI."
                print("🔄 Reset")

            elif key == ord('s') and self.roi_selected and not self.tracking and not self.is_busy:
                frame = self._get_frame()
                if frame is not None and self.roi is not None:
                    x, y, w, h = self.roi
                    crop = frame[y:y+h, x:x+w]
                    self.template_gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    self.template_saved = True
                    self.status = "Template saved. Press 'z' to start tracking."
                    print(f"💾 Template saved: {w}×{h}px")

            elif key == ord('z'):
                if self.tracking:
                    self._stop_tracking()
                elif self.template_saved and not self.is_busy:
                    self._start_tracking()

            elif key in (ord('0'), ord('1'), ord('2'), ord('3')):
                if not self.tracking:
                    self.cursor_sel = int(chr(key))
                    print(f"🎯 Cursor → {self.cursor_sel} "
                          f"({CURSOR_LABELS.get(self.cursor_sel, 'all')})")

            rate.sleep()

        cv2.destroyAllWindows()
        print("👋 Bye")


if __name__ == '__main__':
    try:
        OrbCorrectTracking()
    except rospy.ROSInterruptException:
        pass
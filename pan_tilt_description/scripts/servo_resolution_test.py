#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
servo_resolution_test.py
========================
Interactive live window to find the true minimum servo resolution.

Method:
  - Locks a small reference patch (ORB template) at the current position.
  - Sends a delta command of the current test step size.
  - Waits for the camera to settle, then measures actual pixel displacement
    using ORB matching against the reference patch.
  - Displays: commanded delta, measured pixel shift, and whether the servo
    actually moved (above a 1px noise threshold).

Keyboard:
  SPACE     : send one step in the current direction and measure
  UP / DOWN : increase / decrease step size
  W/A/S/D   : change direction (pan+/pan-/tilt-/tilt+)
  c         : capture new reference patch at current position
  r         : return to origin pose
  ESC       : quit
"""

import rospy
import cv2
import numpy as np
import subprocess
import threading
import time
import math

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pan_tilt_msgs.msg import PanTiltCmdDeg

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_W, IMG_H   = 1280, 720
PATCH_SIZE     = 200          # reference patch side length (px)
STABILISE_S    = 1.2          # wait after command before measuring
MIN_INLIERS    = 5
LOWE_RATIO     = 0.80

STEP_SIZES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]  # degrees

PAN_MIN, PAN_MAX   = -60.0, 60.0
TILT_MIN, TILT_MAX = -60.0, 60.0

MOVED_THRESHOLD_PX = 1.5   # pixel displacement considered "real movement"

# ── ORB match: returns pixel displacement (dx, dy) or None ───────────────────

def measure_shift(ref_gray, scene_gray):
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(scene_gray, None)

    if des1 is None or des2 is None: return None
    if len(kp1) < MIN_INLIERS or len(kp2) < MIN_INLIERS: return None

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_INLIERS: return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask is None: return None
    if mask.sum() < MIN_INLIERS: return None

    # Project patch centre through H
    cx, cy = ref_gray.shape[1] / 2, ref_gray.shape[0] / 2
    pt  = np.float32([[cx, cy]]).reshape(-1, 1, 2)
    dst_pt = cv2.perspectiveTransform(pt, H)
    dx = float(dst_pt[0, 0, 0]) - cx
    dy = float(dst_pt[0, 0, 1]) - cy
    return dx, dy, int(mask.sum())


# ── Main node ─────────────────────────────────────────────────────────────────

class ServoResTest:

    def __init__(self):
        rospy.init_node('servo_resolution_test', anonymous=True)
        self.bridge     = CvBridge()
        self.image      = None
        self.image_lock = threading.Lock()

        self.ref_gray   = None          # reference patch
        self.ref_pos    = None          # (yaw, pitch) when ref was captured
        self.origin_pos = None          # pose at start, for 'r' key

        self.step_idx   = 3             # index into STEP_SIZES → 0.10°
        self.direction  = 'pan+'        # current direction
        self.results    = []            # list of result dicts
        self.last_msg   = "Press 'c' to capture reference, then SPACE to test"
        self.measuring  = False

        rospy.Subscriber('/datavideo/video', Image, self._img_cb)
        self.pub_cmd = rospy.Publisher('/pan_tilt_cmd_deg', PanTiltCmdDeg, queue_size=1)

        cv2.namedWindow("Servo Resolution Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Servo Resolution Test", IMG_W, IMG_H)

        print("\n🔬 Servo Resolution Test")
        print("   SPACE=test  UP/DOWN=step  W/A/S/D=direction  c=capture  r=return  ESC=quit\n")

        # Read initial pose
        yaw, pitch = self._read_pose()
        self.origin_pos = (yaw, pitch)
        print(f"   Origin pose: yaw={yaw:.3f}°  pitch={pitch:.3f}°")

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
        yaw   = max(PAN_MIN,  min(PAN_MAX,  round(yaw,   4)))
        pitch = max(TILT_MIN, min(TILT_MAX, round(pitch, 4)))
        cmd = PanTiltCmdDeg()
        cmd.yaw, cmd.pitch, cmd.speed = yaw, pitch, speed
        self.pub_cmd.publish(cmd)
        return yaw, pitch

    # ── Actions ───────────────────────────────────────────────────────────────

    def _capture_ref(self):
        frame = self._get_frame()
        if frame is None:
            self.last_msg = "❌ No frame"
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = IMG_W // 2, IMG_H // 2
        x1 = cx - PATCH_SIZE // 2
        y1 = cy - PATCH_SIZE // 2
        self.ref_gray = gray[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE].copy()
        self.ref_pos  = self._read_pose()
        self.last_msg = f"📸 Reference captured at yaw={self.ref_pos[0]:.3f}° pitch={self.ref_pos[1]:.3f}°"
        print(self.last_msg)

    def _test_step(self):
        if self.ref_gray is None:
            self.last_msg = "⚠ Capture reference first ('c')"
            return

        self.measuring = True
        step = STEP_SIZES[self.step_idx]
        yaw_now, pitch_now = self._read_pose()

        # Compute delta based on direction
        dyaw, dpitch = 0.0, 0.0
        if   self.direction == 'pan+':  dyaw   =  step
        elif self.direction == 'pan-':  dyaw   = -step
        elif self.direction == 'tilt+': dpitch =  step
        elif self.direction == 'tilt-': dpitch = -step

        new_yaw   = yaw_now   + dyaw
        new_pitch = pitch_now + dpitch

        print(f"\n▶ Sending {self.direction} Δ={step:+.4f}°  "
              f"({yaw_now:.3f},{pitch_now:.3f}) → ({new_yaw:.3f},{new_pitch:.3f})")

        self._send_cmd(new_yaw, new_pitch)
        time.sleep(STABILISE_S)

        # Measure
        frame = self._get_frame()
        if frame is None:
            self.last_msg = "❌ No frame after command"
            self.measuring = False
            return

        scene_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = IMG_W // 2, IMG_H // 2
        x1 = cx - PATCH_SIZE // 2
        y1 = cy - PATCH_SIZE // 2
        scene_patch = scene_gray[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]

        result = measure_shift(self.ref_gray, scene_patch)

        if result is None:
            moved = False
            px_shift = None
            inliers  = 0
            self.last_msg = (f"Δ={step:+.4f}°  → ORB failed (low texture or too small shift)")
        else:
            dx, dy, inliers = result
            px_shift = math.hypot(dx, dy)
            moved    = px_shift >= MOVED_THRESHOLD_PX
            icon     = "✅ MOVED" if moved else "❌ NO MOVE"
            self.last_msg = (f"Δ={step:+.4f}° {self.direction}  "
                             f"shift={px_shift:.2f}px  inliers={inliers}  {icon}")

        print(f"   {self.last_msg}")

        self.results.append(dict(
            step=step, direction=self.direction,
            px_shift=px_shift, moved=moved, inliers=inliers
        ))

        # Return to reference position after measuring
        self._send_cmd(*self.ref_pos)
        time.sleep(STABILISE_S)

        self.measuring = False

    # ── Overlay ───────────────────────────────────────────────────────────────

    def _draw_overlay(self, img):
        # Patch outline in centre
        cx, cy = IMG_W // 2, IMG_H // 2
        x1 = cx - PATCH_SIZE // 2
        y1 = cy - PATCH_SIZE // 2
        col = (0, 200, 255) if self.ref_gray is None else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x1+PATCH_SIZE, y1+PATCH_SIZE), col, 2)
        cv2.drawMarker(img, (cx, cy), col, cv2.MARKER_CROSS, 20, 1)

        step = STEP_SIZES[self.step_idx]

        # Top status bar
        status_color = (0, 200, 255) if self.measuring else (255, 255, 255)
        cv2.putText(img, self.last_msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Info panel
        info = [
            f"Step: {step:.4f}°  (UP/DOWN to change)",
            f"Direction: {self.direction}  (W=tilt+ S=tilt- A=pan- D=pan+)",
            f"Ref captured: {'YES' if self.ref_gray is not None else 'NO — press c'}",
        ]
        for j, line in enumerate(info):
            cv2.putText(img, line, (10, 65 + j * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        # Results table (last 10)
        if self.results:
            cv2.putText(img, "── Recent results ──", (10, IMG_H - 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            for i, r in enumerate(self.results[-10:]):
                px  = f"{r['px_shift']:.2f}px" if r['px_shift'] is not None else "ORB fail"
                icon = "✅" if r['moved'] else "❌"
                line = (f"{icon}  Δ={r['step']:.4f}° {r['direction']:<6}  "
                        f"shift={px}  in={r['inliers']}")
                color = (80, 255, 80) if r['moved'] else (80, 80, 255)
                cv2.putText(img, line, (10, IMG_H - 235 + i * 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        # Bottom bar
        cv2.putText(img,
                    "SPACE=test  UP/DOWN=step  WASD=dir  c=capture  r=return  ESC=quit",
                    (10, IMG_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _main_loop(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            frame = self._get_frame()
            if frame is not None:
                vis = frame.copy()
                self._draw_overlay(vis)
                cv2.imshow("Servo Resolution Test", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            elif key == ord('c') and not self.measuring:
                self._capture_ref()

            elif key == ord(' ') and not self.measuring:
                threading.Thread(target=self._test_step, daemon=True).start()

            elif key == 82:  # UP arrow
                self.step_idx = min(len(STEP_SIZES) - 1, self.step_idx + 1)
                print(f"   Step → {STEP_SIZES[self.step_idx]:.4f}°")

            elif key == 84:  # DOWN arrow
                self.step_idx = max(0, self.step_idx - 1)
                print(f"   Step → {STEP_SIZES[self.step_idx]:.4f}°")

            elif key == ord('w') and not self.measuring:
                self.direction = 'tilt+'
            elif key == ord('s') and not self.measuring:
                self.direction = 'tilt-'
            elif key == ord('a') and not self.measuring:
                self.direction = 'pan-'
            elif key == ord('d') and not self.measuring:
                self.direction = 'pan+'

            elif key == ord('r') and not self.measuring and self.origin_pos:
                self._send_cmd(*self.origin_pos)
                self.last_msg = f"↩ Returned to origin {self.origin_pos}"
                print(self.last_msg)

            rate.sleep()

        cv2.destroyAllWindows()

        # Print summary
        if self.results:
            print("\n── Summary ──────────────────────────────")
            for r in self.results:
                px  = f"{r['px_shift']:.2f}px" if r['px_shift'] is not None else "ORB fail"
                icon = "✅" if r['moved'] else "❌"
                print(f"  {icon}  Δ={r['step']:.4f}° {r['direction']:<6}  shift={px}")
            moved_steps = [r['step'] for r in self.results if r['moved']]
            if moved_steps:
                print(f"\n  Minimum step that produced movement: {min(moved_steps):.4f}°")
            print("─────────────────────────────────────────\n")


if __name__ == '__main__':
    try:
        ServoResTest()
    except rospy.ROSInterruptException:
        pass

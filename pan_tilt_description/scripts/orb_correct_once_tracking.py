#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orb_correct_once_tracking.py
============================
ORB zoom-correction pipeline with integrated continuous tracking mode.

Workflow:
  1. Draw ROI → 'c' save template
  2. 'z' → zoom + correct (existing pipeline)
  3. 't' → start ORB tracking at current zoom level
  4. 't' again or 'q' → stop tracking
  5. Everything else unchanged (WASD, r, k, 1/2/3, ESC)

Tracking design:
  - Runs in a background thread (does not block the display loop)
  - Periodically refreshes the ORB keyframe to fight feature drift
  - Proportional controller for smooth pan/tilt corrections
  - Deadband to suppress micro-jitter when target is already centred
  - Lost-target detection: holds last position and displays warning
  - 'q' key stops tracking cleanly
"""

import rospy
import cv2
import numpy as np
import os
import threading
import time
import math
import csv
import subprocess
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pan_tilt_msgs.msg import PanTiltCmdDeg, PanTiltStatus
from video_stream.srv import SetZoomLevel
import rospkg

# ── Image geometry ─────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720

CURSORS = {
    1: (640, 360),
    2: (643, 415),
    3: (643, 387),
}
CURSOR_LABELS = {1: "Centro imagen", 2: "Centro camara", 3: "Punto extra"}
CURSOR_COLORS = {1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 0, 255)}
THRESHOLD_RECOMENDED_ZOOM = 0.30

# ── ORB / matching parameters ──────────────────────────────────────────────────
LOWE_RATIO   = 0.75
MIN_INLIERS  = 6
STABILISE_S  = 1.5
EPICENTER_SIZE = 450 

MICRO_CORRECT_PX = 15   # px threshold for correction during zoom pipeline
METHOD_NAMES = ["ORB"]

# ── Tracking parameters ────────────────────────────────────────────────────────
TRACK_HZ                  = 45    # control-loop frequency (Hz)
TRACK_DEADBAND_DEG        = 0.05  # angular deadband — no command below this error (degrees)
TRACK_KP                  = 0.20  # proportional gain — reduced to avoid overshoot
TRACK_MAX_STEP_DEG        = 2.0   # max angular correction per cycle (degrees) — hard cap
TRACK_SPEED_FAST          = 30    # motor speed when error > 5°
TRACK_SPEED_SLOW          = 12    # motor speed for fine corrections
TRACK_KEYFRAME_S          = 1     # seconds between forced keyframe refreshes
TRACK_LOST_FRAMES         = 8     # consecutive ORB failures before "target lost"
TRACK_EPICENTER_ROI_RATIO = 0.70  # search window as fraction of ROI size (each axis)
# Re-acquisition search window sizes (pixels, half-width each axis).
# Each step is tried after TRACK_LOST_FRAMES consecutive failures.
# The last entry uses the full image so the target is always findable.
TRACK_REACQUIRE_SIZES     = [0, 200, 400, max(IMG_W, IMG_H)]  # 0 = use epicenter
TRACK_MIN_INLIERS         = 12    # minimum inliers to accept a tracking match
TRACK_MAX_JUMP_PX         = 80    # max px the target can move between cycles (distractor rejection)

# ── Kalman filter (constant-velocity, pixel space) ─────────────────────────────
KALMAN_PROCESS_NOISE      = 1.0   # process noise — lower = velocity changes slowly
KALMAN_MEASURE_NOISE      = 20.0  # measurement noise — higher = trust ORB less
KALMAN_PREDICT_ON_LOSS    = True  # send predictive motor cmds while target is lost
KALMAN_MAX_VELOCITY_PX    = 15.0  # hard clamp on vx/vy (px/frame) after each update
KALMAN_PREDICT_DECAY      = 0.80 # velocity decay per lost frame (0.0=instant stop, 1.0=no decay)

# ── Motor sign convention ──────────────────────────────────────────────────────
# Set to +1 or -1 to match the physical motor response.
# If the camera moves opposite to the target, flip the relevant axis.
PAN_SIGN  = -1   # +1: positive yaw cmd = camera right;  -1: positive yaw cmd = camera left
TILT_SIGN = +1   # +1: positive pitch cmd = camera up;   -1: positive pitch cmd = camera down


# ══════════════════════════════════════════════════════════════════════════════
# Standalone ORB matching helper (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
def find_template_point(template_gray, scene_gray, target_x, target_y):
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(scene_gray, None)

    if des1 is None or des2 is None: return None
    if len(kp1) < MIN_INLIERS or len(kp2) < MIN_INLIERS: return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_INLIERS: return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None: return None

    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < MIN_INLIERS: return None

    pt = np.float32([[target_x, target_y]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pt, H)
    cx_found = float(projected[0, 0, 0])
    cy_found = float(projected[0, 0, 1])

    return cx_found, cy_found, n_inliers, H, mask, kp1, kp2, good


def find_template_point_masked(template_gray, scene_gray, target_x, target_y,
                               template_mask=None):
    """Like find_template_point but restricts keypoint detection on the template
    to the foreground mask (binary, same size as template_gray).
    The scene is always detected without mask so the object can be found anywhere
    within the search window regardless of its new position."""
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(template_gray, template_mask)
    kp2, des2 = orb.detectAndCompute(scene_gray, None)

    if des1 is None or des2 is None: return None
    if len(kp1) < MIN_INLIERS or len(kp2) < MIN_INLIERS: return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

    if len(good) < MIN_INLIERS: return None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None: return None

    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < MIN_INLIERS: return None

    pt = np.float32([[target_x, target_y]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pt, H)
    cx_found = float(projected[0, 0, 0])
    cy_found = float(projected[0, 0, 1])

    return cx_found, cy_found, n_inliers, H, mask, kp1, kp2, good


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════
class OrbCorrectOnce:

    def __init__(self):
        rospy.init_node('orb_correct_once', anonymous=True)
        self.bridge = CvBridge()
        self.image = None
        self.image_lock = threading.Lock()

        self.PAN_MIN_DEG,  self.PAN_MAX_DEG  = -60, 60
        self.TILT_MIN_DEG, self.TILT_MAX_DEG = -60, 60

        self.current_yaw   = 0.0
        self.current_pitch = 0.0

        rospy.sleep(1.0)

        # ── ROI / template state ──
        self.drawing        = False
        self.start_x = self.start_y   = -1
        self.current_x = self.current_y = -1
        self.roi            = None
        self._roi_orig_wh   = (0, 0)  # (w, h) at zoom x1, never scaled
        self.roi_selected   = False
        self.template_gray  = None
        self.template_saved = False
        self.optimal_zoom   = 1.0
        self.current_zoom   = 1.0
        self.current_target_zoom = 1.0

        self.cursor_sel   = 1
        self.is_busy      = False   # True during zoom pipeline

        self.last_target_px = None
        self.last_error_px  = None
        self.last_inliers   = 0

        self.benchmark_results = {m: [] for m in METHOD_NAMES}

        # ── Tracking state ─────────────────────────────────────────────────────
        self.tracking_active  = False   # flag read by both main and track thread
        self._track_thread    = None
        # Shared tracking info written by track thread, read by display thread
        self._track_target_px = None    # (cx, cy) of target in current frame
        self._track_error_px  = None    # (ex, ey) vs cursor
        self._track_lost      = False   # True when consecutive failures exceed limit
        self._track_lock      = threading.Lock()
        self._new_frame_event = threading.Event()  # set by image callback, wakes tracking loop
        self._track_epicenter_px = 250  # updated when tracking starts from ROI size
        self._track_mask      = None   # Canny foreground mask (uint8 255=fg), for overlay
        self._track_mask_x1   = 0      # absolute top-left x of mask in full image
        self._track_mask_y1   = 0      # absolute top-left y of mask in full image
        # ──────────────────────────────────────────────────────────────────────

        # ── Recording state ────────────────────────────────────────────────────
        self._recording       = False
        self._ffmpeg_proc     = None   # subprocess.Popen handle
        self._rec_lock        = threading.Lock()
        # ──────────────────────────────────────────────────────────────────────

        self.status = "Draw a ROI around the target, then press 'c'"

        rospack  = rospkg.RosPack()
        pkg_path = rospack.get_path("pan_tilt_description")
        self.image_dir = os.path.join(pkg_path, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.video_dir = os.path.join(pkg_path, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self.pub_cmd = rospy.Publisher('/pan_tilt_cmd_deg', PanTiltCmdDeg, queue_size=10)
        rospy.Subscriber('/datavideo/video',  Image,         self._img_cb)
        rospy.Subscriber('/pan_tilt_status',  PanTiltStatus, self._status_cb)

        rospy.loginfo("Connecting to /set_zoom service...")
        try:
            rospy.wait_for_service('/set_zoom', timeout=3.0)
            self.zoom_srv = rospy.ServiceProxy('/set_zoom', SetZoomLevel)
        except rospy.ROSException:
            rospy.logwarn("/set_zoom service not found.")
            self.zoom_srv = None

        cv2.namedWindow("ORB Correct Once")
        cv2.setMouseCallback("ORB Correct Once", self._mouse_cb)

        print("\n🟢 orb_correct_once_tracking started")
        print("   Draw ROI → 'c' save template → 'z' zoom + correct → 't' track")
        print("   't' again or 'q' → stop tracking")
        print("   'g' → start/stop recording")
        print("   W/A/S/D → Move camera manually (1°)")
        print("   k → Auto-calibrate lens")
        print("   1/2/3 cursor   0 all   r reset   ESC quit\n")

        self._main_loop()

    # ══════════════════════════════════════════════════════════════════════════
    # Mathematical models (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def _get_fovs(self, target_z):
        z = max(1.0, min(20.0, float(target_z)))
        t = (z - 1.0) / 19.0
        k = 1.78
        fov_w_h, fov_t_h = 63.7, 2.3
        fov_w_v, fov_t_v = 35.84, 1.3
        fov_h = fov_w_h * ((1.0 - t)**k) + fov_t_h * t
        fov_v = fov_w_v * ((1.0 - t)**k) + fov_t_v * t
        return fov_h, fov_v

    def _get_zoom_factors(self, target_z):
        fov_h_1, fov_v_1 = self._get_fovs(1.0)
        fov_h_z, fov_v_z = self._get_fovs(target_z)
        k_escala = 1
        fx = (math.tan(math.radians(fov_h_1 / 2.0)) /
              math.tan(math.radians(fov_h_z / 2.0))) ** k_escala
        fy = (math.tan(math.radians(fov_v_1 / 2.0)) /
              math.tan(math.radians(fov_v_z / 2.0))) ** k_escala
        return fx, fy

    def _px_to_deg(self, ex, ey, zoom_level):
        fov_h, fov_v = self._get_fovs(zoom_level)
        return ex * fov_h / IMG_W, ey * fov_v / IMG_H

    # ══════════════════════════════════════════════════════════════════════════
    # ROS callbacks
    # ══════════════════════════════════════════════════════════════════════════
    def _img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.image = frame
            self._new_frame_event.set()
        except Exception as e:
            rospy.logwarn(f"Image error: {e}")

    def _status_cb(self, msg):
        self.current_yaw   = msg.yaw_now
        self.current_pitch = msg.pitch_now

    def _get_frame(self):
        with self.image_lock:
            return self.image.copy() if self.image is not None else None

    # ══════════════════════════════════════════════════════════════════════════
    # PTZ commands (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def _set_zoom(self, level):
        if self.zoom_srv:
            try:
                self.zoom_srv(float(level))
            except rospy.ServiceException as e:
                print(f"❌ Zoom error: {e}")
        else:
            print(f"⚠️  Zoom simulation: x{float(level):.1f}")

    def _send_cmd(self, yaw, pitch, speed=20):
        cmd = PanTiltCmdDeg()
        cmd.yaw, cmd.pitch, cmd.speed = yaw, pitch, speed
        self.pub_cmd.publish(cmd)

    def _calculate_and_send_cmd(self, ex, ey, zoom_level, gain=1.0):
        """
        Direct pixel error → absolute pan/tilt command, no TF2.
        Converts pixel error to angular offset using the FOV model, applies
        optional proportional gain, then adds the delta to the current motor pose.
        """
        deg_x, deg_y = self._px_to_deg(ex * gain, ey * gain, zoom_level)
        error_mag = math.sqrt(deg_x**2 + deg_y**2)
        speed_val = TRACK_SPEED_FAST if error_mag > 5.0 else TRACK_SPEED_SLOW

        new_yaw   = max(self.PAN_MIN_DEG,
                        min(self.PAN_MAX_DEG,  round(self.current_yaw   + PAN_SIGN  * deg_x)))
        new_pitch = max(self.TILT_MIN_DEG,
                        min(self.TILT_MAX_DEG, round(self.current_pitch + TILT_SIGN * deg_y)))

        self._send_cmd(new_yaw, new_pitch, speed=speed_val)
        return new_yaw, new_pitch

    # ══════════════════════════════════════════════════════════════════════════
    # ROI / zoom helpers (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def _update_roi_visual(self, cx, cy, zoom_level, orig_w, orig_h):
        fx, fy = self._get_zoom_factors(zoom_level)
        new_w  = orig_w * fx
        new_h  = orig_h * fy
        new_x  = cx - new_w / 2.0
        new_y  = cy - new_h / 2.0
        self.roi = (int(new_x), int(new_y), int(new_w), int(new_h))

    def _get_roi_crop(self, image_gray, cx, cy, crop_w, crop_h):
        x1 = max(0, int(cx - crop_w / 2.0))
        y1 = max(0, int(cy - crop_h / 2.0))
        x2 = min(IMG_W, int(cx + crop_w / 2.0))
        y2 = min(IMG_H, int(cy + crop_h / 2.0))
        crop_gray = image_gray[y1:y2, x1:x2]
        local_cx  = cx - x1
        local_cy  = cy - y1
        return crop_gray, local_cx, local_cy

    def _get_roi_crop_with_origin(self, image_gray, cx, cy, crop_w, crop_h):
        """Like _get_roi_crop but also returns the absolute top-left (x1, y1)
        of the crop so callers can remap local coords back to full-image coords."""
        x1 = max(0, int(cx - crop_w / 2.0))
        y1 = max(0, int(cy - crop_h / 2.0))
        x2 = min(IMG_W, int(cx + crop_w / 2.0))
        y2 = min(IMG_H, int(cy + crop_h / 2.0))
        crop_gray = image_gray[y1:y2, x1:x2]
        local_cx  = cx - x1
        local_cy  = cy - y1
        return crop_gray, local_cx, local_cy, x1, y1

    def _best_zoom(self, w, h):
        for z_int in range(200, 9, -1):
            z = round(z_int / 10.0, 1)
            fx, fy = self._get_zoom_factors(z)
            if w * fx <= IMG_W * THRESHOLD_RECOMENDED_ZOOM and h * fy <= IMG_H * THRESHOLD_RECOMENDED_ZOOM:
                return z
        return 1.0

    def _move_manual(self, dyaw, dpitch):
        if self.is_busy or self.tracking_active: return
        target_yaw   = max(self.PAN_MIN_DEG,  min(self.PAN_MAX_DEG,  round(self.current_yaw)   + dyaw))
        target_pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, round(self.current_pitch) + dpitch))
        self.status  = f"Manual: yaw={target_yaw} deg  pitch={target_pitch} deg"
        self._send_cmd(target_yaw, target_pitch, speed=15)

    # ══════════════════════════════════════════════════════════════════════════
    # Auto-calibration (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════════
    def auto_calibrate_lens(self):
        if self.is_busy: return
        self.is_busy = True
        try:
            self.status = "Starting chained auto-calibration..."
            print(f"\n{self.status}")
            fov_w, fov_t, z_max = 63.7, 2.3, 20.0
            zoom_steps = [1.0, 3.0, 5.0, 8.0, 11.0]
            valores_k  = []

            self._set_zoom(zoom_steps[0])
            time.sleep(3.0)

            frame_prev = self._get_frame()
            if frame_prev is None:
                print("❌ No video frame."); return
            gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
            cx_prev, cy_prev = IMG_W / 2.0, IMG_H / 2.0
            escala_acumulada = 1.0

            for i in range(1, len(zoom_steps)):
                z_target = zoom_steps[i]
                z_prev   = zoom_steps[i - 1]
                crop_size = 400
                template, lcx, lcy = self._get_roi_crop(gray_prev, cx_prev, cy_prev, crop_size, crop_size)
                self.status = f"Applying zoom x{z_target}..."
                print(self.status)
                self._set_zoom(z_target)
                time.sleep(2.5)
                frame_curr = self._get_frame()
                gray_curr  = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
                self.status = f"Computing deformation (x{z_prev} -> x{z_target})..."
                res = find_template_point(template, gray_curr, lcx, lcy)
                if res is None:
                    print(f"⚠️  ORB lost reference at x{z_target}."); break
                cx_found, cy_found, _, H, _, _, _, _ = res
                pts_orig = np.float32([[lcx - 50, lcy], [lcx + 50, lcy]]).reshape(-1, 1, 2)
                pts_proy = cv2.perspectiveTransform(pts_orig, H)
                dist_nueva      = np.linalg.norm(pts_proy[0, 0] - pts_proy[1, 0])
                escala_relativa = dist_nueva / 100.0
                escala_acumulada *= escala_relativa
                tan_w   = math.tan(math.radians(fov_w / 2.0))
                fov_real = 2.0 * math.degrees(math.atan(tan_w / escala_acumulada))
                t = (z_target - 1.0) / (z_max - 1.0)
                numerador = (fov_real - fov_t * t) / fov_w
                if numerador > 0 and (1.0 - t) > 0:
                    k_paso = math.log(numerador) / math.log(1.0 - t)
                    valores_k.append(k_paso)
                    print(f"  -> S_total={escala_acumulada:.2f}x | FOV={fov_real:.1f}° | k={k_paso:.3f}")
                gray_prev = gray_curr
                cx_prev   = cx_found
                cy_prev   = cy_found

            if valores_k:
                k_new = max(valores_k)
                self.status = f"Calibration done! k = {k_new:.3f}"
                print(f"\n{self.status}")
            else:
                self.status = "Calibration failed. Point at a textured wall."
                print(self.status)
        except Exception as e:
            print(f"Calibration error: {e}")
        finally:
            self._set_zoom(1.0)
            self.is_busy = False
            self.status  = "Manual mode ready."

    # ══════════════════════════════════════════════════════════════════════════
    # Zoom + correct pipeline (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════════
    def _zoom_and_correct(self):
        try:
            self.benchmark_results = {m: [] for m in METHOD_NAMES}
            target_zoom = self.current_target_zoom
            cx_cursor, cy_cursor = CURSORS[self.cursor_sel]
            x, y, orig_w, orig_h = [int(v) for v in self.roi]

            # Safety zoom limit
            max_safe_zoom = 1.0
            for z_int in range(10, 201):
                z = round(z_int / 10.0, 1)
                fx, fy = self._get_zoom_factors(z)
                if orig_w * fx > IMG_W * 0.95 or orig_h * fy > IMG_H * 0.95:
                    break
                max_safe_zoom = z
            if target_zoom > max_safe_zoom:
                print(f"⚠️  Optical limit: x{max_safe_zoom:.1f}")
                target_zoom = max_safe_zoom

            self._set_zoom(1.0)
            time.sleep(0.5)
            frame_orig = self._get_frame()
            if frame_orig is None:
                self.status = "No frame at x1.0"; return

            initial_cx = x + orig_w / 2.0
            initial_cy = y + orig_h / 2.0
            crop_w_init = max(int(orig_w * 1.5), 250)
            crop_h_init = max(int(orig_h * 1.5), 250)
            frame_orig_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            template_orig, local_cx_orig, local_cy_orig = self._get_roi_crop(
                frame_orig_gray, initial_cx, initial_cy, crop_w_init, crop_h_init)

            ex_init = initial_cx - cx_cursor
            ey_init = initial_cy - cy_cursor

            if abs(ex_init) > 40 or abs(ey_init) > 40:
                self.status = "Pre-centering target..."
                self._calculate_and_send_cmd(ex_init, ey_init, 1.0)
                time.sleep(2.0)
                frame1 = self._get_frame()
                scene_gray_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                res0 = find_template_point(template_orig, scene_gray_1,
                                           local_cx_orig, local_cy_orig)
                if res0 is None:
                    self.status = "Failed to re-acquire after pre-centering"; return
                cx_found, cy_found, _, _, _, _, _, _ = res0
                current_cx, current_cy = cx_found, cy_found
                frame_base = frame1
            else:
                current_cx, current_cy = initial_cx, initial_cy
                frame_base = frame_orig

            self._update_roi_visual(current_cx, current_cy, 1.0, orig_w, orig_h)
            scene_gray_base = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)
            template_gray, local_cx, local_cy = self._get_roi_crop(
                scene_gray_base, current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)

            MAX_SAFE_JUMP = 4.0
            total_distance = target_zoom - 1.0
            zoom_steps = []
            if total_distance > 0:
                num_steps    = math.ceil(total_distance / MAX_SAFE_JUMP)
                dynamic_step = total_distance / num_steps
                curr_z = 1.0
                for _ in range(num_steps):
                    curr_z += dynamic_step
                    zoom_steps.append(round(curr_z, 1))
                if zoom_steps:
                    zoom_steps[-1] = float(round(target_zoom, 1))

            H_final   = None
            n_inliers = 0

            for step_zoom in zoom_steps:
                self.status = f"Applying zoom x{step_zoom:.1f}..."
                self._set_zoom(step_zoom)
                time.sleep(STABILISE_S)

                frame_z = self._get_frame()
                if frame_z is None: return
                scene_gray = cv2.cvtColor(frame_z, cv2.COLOR_BGR2GRAY)
                self.status = f"Matching template at x{step_zoom:.1f}..."

                t_start  = time.time()
                result   = find_template_point(template_gray, scene_gray, local_cx, local_cy)
                match_dt = time.time() - t_start

                if result is None:
                    self.status = f"ORB failed at x{step_zoom:.1f}"; return

                cx_found, cy_found, n_inliers, H_final, mask, kp1, kp2, good = result
                ex_step  = cx_found - cx_cursor
                ey_step  = cy_found - cy_cursor
                err_px   = math.sqrt(ex_step**2 + ey_step**2)

                self.benchmark_results["ORB"].append({
                    "zoom": step_zoom, "label": f"x{step_zoom:.1f}",
                    "ex": ex_step, "ey": ey_step,
                    "err_px": err_px, "duration_s": match_dt
                })

                if abs(ex_step) > MICRO_CORRECT_PX or abs(ey_step) > MICRO_CORRECT_PX:
                    self.status = f"Micro-centering at x{step_zoom:.1f}..."
                    self._calculate_and_send_cmd(ex_step, ey_step, step_zoom)
                    time.sleep(2.0)
                    frame_z    = self._get_frame()
                    scene_gray = cv2.cvtColor(frame_z, cv2.COLOR_BGR2GRAY)
                    res_c = find_template_point(template_gray, scene_gray, local_cx, local_cy)
                    if res_c is not None:
                        cx_found, cy_found, n_inliers, H_final, _, _, _, _ = res_c
                    else:
                        cx_found, cy_found = cx_cursor, cy_cursor

                current_cx, current_cy = cx_found, cy_found
                self._update_roi_visual(current_cx, current_cy, step_zoom, orig_w, orig_h)

                if step_zoom != zoom_steps[-1]:
                    template_gray, local_cx, local_cy = self._get_roi_crop(
                        scene_gray, current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)

            self.last_target_px = (current_cx, current_cy)
            self.last_inliers   = n_inliers
            ex_final = current_cx - cx_cursor
            ey_final = current_cy - cy_cursor
            self.last_error_px  = (ex_final, ey_final)

            yaw_final, pitch_final = self._calculate_and_send_cmd(
                ex_final, ey_final, target_zoom)
            self._update_roi_visual(cx_cursor, cy_cursor, target_zoom, orig_w, orig_h)

            self.status = (f"Correction done: yaw={yaw_final} deg  pitch={pitch_final} deg"
                           f" | inliers={n_inliers} | press 't' to track")
            print(f"\n{self.status}\n")

            if H_final is not None:
                self._save_debug(frame_z, template_gray, H_final,
                                 current_cx, current_cy,
                                 cx_cursor, cy_cursor,
                                 ex_final, ey_final)

            ts_str      = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir     = os.path.join(self.image_dir, f"benchmark_{ts_str}")
            os.makedirs(out_dir, exist_ok=True)
            self._save_benchmark_results(self.roi, self.current_yaw,
                                         self.current_pitch, out_dir, ts_str)

        except Exception as e:
            self.status = f"Error: {e}"
            print(e)
        finally:
            self.is_busy = False

    # ══════════════════════════════════════════════════════════════════════════
    # Foreground mask for tracking (Canny + flood-fill)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_foreground_mask(self, frame_bgr, cx, cy, epicenter, seed_w=None, seed_h=None):
        """Build a foreground mask using Canny edges + flood-fill from the object centre.
        Returns a binary uint8 mask (255=foreground) the same size as the epicenter crop.
        Falls back to None if the mask is implausible."""
        x1 = max(0, int(cx - epicenter / 2.0))
        y1 = max(0, int(cy - epicenter / 2.0))
        x2 = min(IMG_W, int(cx + epicenter / 2.0))
        y2 = min(IMG_H, int(cy + epicenter / 2.0))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        h, w = crop.shape[:2]

        # Seed point: object centre in crop coordinates
        seed_x = int(cx - x1)
        seed_y = int(cy - y1)
        seed_x = max(1, min(w - 2, seed_x))
        seed_y = max(1, min(h - 2, seed_y))

        # ── Canny edge detection ───────────────────────────────────────────────
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Mild blur to suppress fur texture noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Auto-threshold via Otsu on the gradient magnitude
        _, otsu_thresh = cv2.threshold(blurred, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_val = float(otsu_thresh.max())   # Otsu threshold value stored in dst
        # Recompute: use Otsu value as upper Canny threshold
        _, thresh_val = cv2.threshold(blurred, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Get the actual scalar threshold via OTSU flag
        otsu_scalar, _ = cv2.threshold(blurred, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred,
                          threshold1=otsu_scalar * 0.5,
                          threshold2=otsu_scalar)

        # ── Close edge gaps so the boundary is sealed for flood-fill ──────────
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_DILATE, close_k)

        # ── Flood-fill from object centre into the enclosed region ────────────
        # Flood-fill works on an inverted edge image: edges=0 (walls), interior=255
        fill_src = cv2.bitwise_not(edges_closed)
        # Add 1-pixel border required by floodFill
        fill_src_bordered = cv2.copyMakeBorder(fill_src, 1, 1, 1, 1,
                                               cv2.BORDER_CONSTANT, value=0)
        flood_mask = np.zeros((h + 4, w + 4), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(fill_src_bordered, flood_mask,
                      seedPoint=(seed_x + 1, seed_y + 1),
                      newVal=255, loDiff=10, upDiff=10, flags=flags)
        # Extract the filled region (strip the 2-pixel border added by floodFill)
        fg_mask = flood_mask[2:h + 2, 2:w + 2]

        # ── Morphological cleanup ─────────────────────────────────────────────
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_open)

        # ── Sanity check ──────────────────────────────────────────────────────
        coverage = fg_mask.mean() / 255.0
        if coverage < 0.03 or coverage > 0.95:
            print(f"  [Canny mask] Coverage {coverage:.2f} — discarding mask")
            return None
        print(f"  [Canny mask] Coverage {coverage:.2f} — OK")
        return fg_mask

    # ══════════════════════════════════════════════════════════════════════════
    # Tracking entry / exit
    # ══════════════════════════════════════════════════════════════════════════
    def _start_tracking(self):
        """
        Public entry point called from the main loop when 't' is pressed.
        Requires: template_saved=True, zoom already applied.
        """
        if not self.template_saved:
            print("⚠️  Save a template first (press 'c' after drawing ROI).")
            return
        if self.is_busy:
            print("⚠️  Zoom pipeline still running.")
            return
        if self.tracking_active:
            self._stop_tracking()
            return

        self.tracking_active = True
        # Compute search window from the ROI scaled to the current zoom level
        roi_w, roi_h = self._roi_orig_wh
        fx, fy = self._get_zoom_factors(self.current_target_zoom)
        scaled_w = roi_w * fx
        scaled_h = roi_h * fy
        self._track_epicenter_px = max(
            128, int(max(scaled_w, scaled_h) * TRACK_EPICENTER_ROI_RATIO))
        with self._track_lock:
            self._track_target_px = self.last_target_px
            self._track_error_px  = None
            self._track_lost      = False

        self._track_thread = threading.Thread(
            target=self._tracking_loop, daemon=True)
        self._track_thread.start()
        self.status = f"Tracking active at zoom x{self.current_target_zoom:.1f} | t/q to stop"
        print(f"\n{self.status}")

    def _stop_tracking(self):
        """Signal the tracking thread to exit and wait for it."""
        self.tracking_active = False
        if self._track_thread is not None:
            self._track_thread.join(timeout=2.0)
            self._track_thread = None
        self._track_mask = None
        self.status = "Tracking stopped. Manual mode ready."
        print(f"\n{self.status}")

    def _init_kalman(self, cx, cy):
        """Initialise a 4-state (cx, cy, vx, vy) constant-velocity Kalman filter."""
        kf = cv2.KalmanFilter(4, 2)
        dt = 1.0 / TRACK_HZ

        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        q = KALMAN_PROCESS_NOISE
        kf.processNoiseCov = np.diag([q, q, q * 2, q * 2]).astype(np.float32)

        r = KALMAN_MEASURE_NOISE
        kf.measurementNoiseCov = np.diag([r, r]).astype(np.float32)

        kf.statePre     = np.array([cx, cy, 0, 0], dtype=np.float32).reshape(4, 1)
        kf.statePost    = kf.statePre.copy()
        kf.errorCovPre  = np.eye(4, dtype=np.float32) * 100.0
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        return kf

    def _tracking_loop(self):
        """
        Background thread: continuous ORB-based closed-loop tracking.

        Design decisions
        ----------------
        Event-driven frame consumption (item 10): the loop wakes on
            _new_frame_event instead of sleeping a fixed dt, so it never
            processes a stale frame twice and has zero unnecessary latency.

        Keyframe refresh (items 4 & 7): the keyframe crop is stored already
            in grayscale to avoid repeated conversion.  It is refreshed either
            on the periodic timer OR immediately after a motor command is issued
            (after a short settle wait), whichever comes first.

        Adaptive deadband (item 6): the deadband is expressed in degrees
            (TRACK_DEADBAND_DEG) and converted to pixels at the current zoom
            level so it represents the same angular tolerance regardless of zoom.

        Proportional control: error is scaled by TRACK_KP before converting
            to degrees, trading correction speed for mechanical stability.

        Lost-target & re-acquisition (item 9): after TRACK_LOST_FRAMES
            consecutive failures the thread progressively expands the search
            window through TRACK_REACQUIRE_SCALES until the target is found or
            tracking is stopped.
        """
        zoom_level   = self.current_target_zoom
        cx_cursor, cy_cursor = CURSORS[self.cursor_sel]
        epicenter    = self._track_epicenter_px

        # Adaptive deadband: fixed angular threshold → pixels at current zoom
        fov_h, _ = self._get_fovs(zoom_level)
        deadband_px = TRACK_DEADBAND_DEG * IMG_W / fov_h

        # ── Build initial keyframe (already gray) ──────────────────────────────
        self._new_frame_event.wait(timeout=2.0)
        self._new_frame_event.clear()
        frame = self._get_frame()
        if frame is None:
            print("❌ [Track] No frame available at start."); return

        with self._track_lock:
            init_pos = self._track_target_px

        current_cx = float(init_pos[0]) if init_pos is not None else float(cx_cursor)
        current_cy = float(init_pos[1]) if init_pos is not None else float(cy_cursor)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kf_gray, kf_lcx, kf_lcy, kf_x1, kf_y1 = self._get_roi_crop_with_origin(
            gray, current_cx, current_cy, epicenter, epicenter)

        # Keep a copy of the initial keyframe as the re-acquisition template.
        # This is used when the search window expands beyond the epicenter — the
        # live kf_gray crop is the right size for normal tracking, but for a
        # full-frame search we need a template that ORB can match against a large scene.
        reacq_template      = kf_gray.copy()
        reacq_template_lcx  = kf_lcx
        reacq_template_lcy  = kf_lcy

        # ── Foreground mask — Canny+flood-fill, computed once at tracking start ──
        self.status = "Building foreground mask..."
        roi_w, roi_h = self._roi_orig_wh
        fx, fy = self._get_zoom_factors(zoom_level)
        seed_w = roi_w * fx
        seed_h = roi_h * fy
        kf_mask = self._build_foreground_mask(frame, current_cx, current_cy,
                                              epicenter, seed_w, seed_h)
        if kf_mask is not None:
            # Store mask and its top-left position for overlay rendering
            self._track_mask    = kf_mask
            self._track_mask_x1 = max(0, int(current_cx - epicenter / 2.0))
            self._track_mask_y1 = max(0, int(current_cy - epicenter / 2.0))
            print(f"  [Track] Canny mask active — background features suppressed")
        else:
            self._track_mask = None
            print(f"  [Track] Canny mask unavailable — using full keyframe")

        last_keyframe_t  = time.time()
        pending_refresh  = False
        refresh_after_t  = None
        fail_count       = 0
        reacquire_idx    = 0
        match_ms_cur     = 0.0   # match time this cycle (ms)
        match_hz_cur     = 0.0   # max Hz this cycle
        match_hz_min     = float('inf')  # session minimum Hz

        # ── Kalman filter init ─────────────────────────────────────────────────
        kf = self._init_kalman(current_cx, current_cy)

        print(f"  [Track] Loop started — zoom x{zoom_level:.1f}, "
              f"cursor ({cx_cursor},{cy_cursor}), "
              f"epicenter {epicenter}px, deadband {deadband_px:.1f}px")

        while self.tracking_active and not rospy.is_shutdown():

            # ── Event-driven: wait for next camera frame ───────────────────────
            self._new_frame_event.wait(timeout=1.0 / TRACK_HZ)
            self._new_frame_event.clear()

            now = time.time()

            # ── Post-command keyframe refresh (item 7) ─────────────────────────
            if pending_refresh and now >= refresh_after_t:
                frame_kf = self._get_frame()
                if frame_kf is not None:
                    gray_kf = cv2.cvtColor(frame_kf, cv2.COLOR_BGR2GRAY)
                    kf_gray, kf_lcx, kf_lcy, kf_x1, kf_y1 = \
                        self._get_roi_crop_with_origin(
                            gray_kf, current_cx, current_cy, epicenter, epicenter)
                    last_keyframe_t = now
                pending_refresh = False

            # ── Periodic keyframe refresh fallback ─────────────────────────────
            elif (now - last_keyframe_t) >= TRACK_KEYFRAME_S:
                frame_kf = self._get_frame()
                if frame_kf is not None:
                    gray_kf = cv2.cvtColor(frame_kf, cv2.COLOR_BGR2GRAY)
                    kf_gray, kf_lcx, kf_lcy, kf_x1, kf_y1 = \
                        self._get_roi_crop_with_origin(
                            gray_kf, current_cx, current_cy, epicenter, epicenter)
                    last_keyframe_t = now

            # ── Grab current frame (item 3: skip if same as keyframe refresh) ──
            frame_curr = self._get_frame()
            if frame_curr is None:
                continue

            gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

            # ── Kalman predict ─────────────────────────────────────────────────
            predicted = kf.predict()
            pred_cx = float(predicted[0])
            pred_cy = float(predicted[1])
            # Clamp prediction to image bounds
            pred_cx = max(0, min(IMG_W - 1, pred_cx))
            pred_cy = max(0, min(IMG_H - 1, pred_cy))

            # ── ORB match in search window (centred on Kalman prediction) ──────
            reacq_size     = TRACK_REACQUIRE_SIZES[reacquire_idx]
            is_reacquiring = reacq_size > 0

            if is_reacquiring:
                half = reacq_size
                cx_search = current_cx
                cy_search = current_cy
                tmpl      = reacq_template
                tmpl_lcx  = reacq_template_lcx
                tmpl_lcy  = reacq_template_lcy
                tmpl_mask = kf_mask
            else:
                half      = epicenter // 2
                # Centre search on Kalman prediction rather than last keyframe origin
                cx_search = pred_cx
                cy_search = pred_cy
                tmpl      = kf_gray
                tmpl_lcx  = kf_lcx
                tmpl_lcy  = kf_lcy
                tmpl_mask = kf_mask

            sx1 = max(0, int(cx_search - half))
            sy1 = max(0, int(cy_search - half))
            sx2 = min(IMG_W, int(cx_search + half))
            sy2 = min(IMG_H, int(cy_search + half))
            scene_crop = gray_curr[sy1:sy2, sx1:sx2]

            t_match = time.time()
            result = find_template_point_masked(tmpl, scene_crop,
                                                tmpl_lcx, tmpl_lcy, tmpl_mask)
            match_ms_cur = (time.time() - t_match) * 1000.0
            match_hz_cur = 1000.0 / match_ms_cur if match_ms_cur > 0 else 0.0
            if match_hz_cur > 0:
                match_hz_min = min(match_hz_min, match_hz_cur)

            if result is None:
                fail_count += 1
                lost_now = fail_count >= TRACK_LOST_FRAMES
                if lost_now:
                    reacquire_idx = min(reacquire_idx + 1,
                                        len(TRACK_REACQUIRE_SIZES) - 1)
                    sz = TRACK_REACQUIRE_SIZES[reacquire_idx]
                    sz_label = "full frame" if sz >= max(IMG_W, IMG_H) else f"{sz*2}px"
                    self.status = f"[Track] Target LOST - searching {sz_label} window..."
                    # Follow predicted trajectory with exponential velocity decay.
                    # Velocity decays by KALMAN_PREDICT_DECAY each lost frame until
                    # it falls below the deadband naturally — no hard cutoff.
                    if KALMAN_PREDICT_ON_LOSS:
                        kf.statePost[2] = float(kf.statePost[2]) * KALMAN_PREDICT_DECAY
                        kf.statePost[3] = float(kf.statePost[3]) * KALMAN_PREDICT_DECAY
                        ex_pred = pred_cx - cx_cursor
                        ey_pred = pred_cy - cy_cursor
                        err_pred = math.sqrt(ex_pred**2 + ey_pred**2)
                        if err_pred > deadband_px:
                            fov_h_now, fov_v_now = self._get_fovs(zoom_level)
                            ex_c = max(-(TRACK_MAX_STEP_DEG * IMG_W / fov_h_now),
                                       min( TRACK_MAX_STEP_DEG * IMG_W / fov_h_now,
                                            ex_pred * TRACK_KP))
                            ey_c = max(-(TRACK_MAX_STEP_DEG * IMG_H / fov_v_now),
                                       min( TRACK_MAX_STEP_DEG * IMG_H / fov_v_now,
                                            ey_pred * TRACK_KP))
                            self._calculate_and_send_cmd(ex_c, ey_c, zoom_level)
                with self._track_lock:
                    self._track_lost = lost_now
                # Advance current position estimate with Kalman prediction
                current_cx, current_cy = pred_cx, pred_cy
                continue

            # ── Successful match — Kalman update ──────────────────────────────
            fail_count    = 0
            reacquire_idx = 0
            cx_found, cy_found, n_inliers, _, _, _, _, _ = result

            cx_img = sx1 + cx_found
            cy_img = sy1 + cy_found

            if n_inliers < TRACK_MIN_INLIERS:
                with self._track_lock:
                    self._track_lost = False
                self.status = f"Tracking | low inliers ({n_inliers}) — skipping"
                # Reset velocity — measurement was unreliable
                kf.statePost[2] = 0.0
                kf.statePost[3] = 0.0
                current_cx, current_cy = pred_cx, pred_cy
                continue

            jump_px = math.sqrt((cx_img - current_cx)**2 + (cy_img - current_cy)**2)
            if not is_reacquiring and jump_px > TRACK_MAX_JUMP_PX:
                with self._track_lock:
                    self._track_lost = False
                self.status = f"Tracking | jump {jump_px:.0f}px rejected"
                # Reset velocity — match was likely a distractor
                kf.statePost[2] = 0.0
                kf.statePost[3] = 0.0
                current_cx, current_cy = pred_cx, pred_cy
                continue

            # Fuse ORB measurement into Kalman
            measurement = np.array([[cx_img], [cy_img]], dtype=np.float32)
            corrected   = kf.correct(measurement)
            # Clamp velocity to prevent runaway from a bad measurement
            vx_raw = float(kf.statePost[2])
            vy_raw = float(kf.statePost[3])
            kf.statePost[2] = np.clip(vx_raw, -KALMAN_MAX_VELOCITY_PX, KALMAN_MAX_VELOCITY_PX)
            kf.statePost[3] = np.clip(vy_raw, -KALMAN_MAX_VELOCITY_PX, KALMAN_MAX_VELOCITY_PX)
            current_cx  = float(kf.statePost[0])
            current_cy  = float(kf.statePost[1])

            ex = current_cx - cx_cursor
            ey = current_cy - cy_cursor
            err_px = math.sqrt(ex**2 + ey**2)

            with self._track_lock:
                self._track_target_px = (current_cx, current_cy)
                self._track_error_px  = (ex, ey)
                self._track_lost      = False

            # ── Proportional control with adaptive deadband ────────────────────
            if err_px > deadband_px:
                fov_h_now, fov_v_now = self._get_fovs(zoom_level)
                cap_px_x = TRACK_MAX_STEP_DEG * IMG_W / fov_h_now
                cap_px_y = TRACK_MAX_STEP_DEG * IMG_H / fov_v_now
                ex_capped = max(-cap_px_x, min(cap_px_x, ex * TRACK_KP))
                ey_capped = max(-cap_px_y, min(cap_px_y, ey * TRACK_KP))
                self._calculate_and_send_cmd(ex_capped, ey_capped, zoom_level)
                settle_s        = 0.3 + err_px / IMG_W
                pending_refresh = True
                refresh_after_t = time.time() + settle_s

            else:
                pass  # within deadband — Kalman already updated current_cx/cy above

            deg_x, deg_y = self._px_to_deg(ex, ey, zoom_level)
            err_deg = math.sqrt(deg_x**2 + deg_y**2)
            self.status = (f"Tracking | err={err_px:.1f}px ({err_deg:.3f} deg)"
                           f" | in={n_inliers}"
                           f" | {match_ms_cur:.0f}ms {match_hz_cur:.0f}Hz"
                           f" (min {match_hz_min:.0f}Hz)")

        print("  [Track] Loop exited.")

    # ══════════════════════════════════════════════════════════════════════════
    # Debug / benchmark save helpers (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def _save_debug(self, frame_z, template_gray, H,
                    cx_found, cy_found, cx_cursor, cy_cursor, ex, ey):
        debug  = frame_z.copy()
        th, tw = template_gray.shape
        corners = np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1,1,2)
        proj    = cv2.perspectiveTransform(corners, H)
        cv2.polylines(debug, [np.int32(proj)], True, (0,165,255), 2)
        cv2.drawMarker(debug, (int(cx_found), int(cy_found)),
                       (0,165,255), cv2.MARKER_CROSS, 24, 2)
        color = CURSOR_COLORS[self.cursor_sel]
        cv2.drawMarker(debug, (cx_cursor, cy_cursor),
                       color, cv2.MARKER_CROSS, 24, 2)
        cv2.arrowedLine(debug, (int(cx_found), int(cy_found)),
                        (cx_cursor, cy_cursor), (255,255,0), 2, tipLength=0.15)
        cv2.imwrite(os.path.join(self.image_dir, "orb_correction_debug.jpg"), debug)

    def _save_benchmark_results(self, roi, init_yaw, init_pitch, out_dir, ts):
        xr, yr, wr, hr = roi
        roi_cx, roi_cy  = xr + wr / 2.0, yr + hr / 2.0

        csv_path = os.path.join(out_dir, "results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# Benchmark", ts])
            w.writerow(["# ROI", f"x={xr} y={yr} w={wr} h={hr}",
                        f"cx={roi_cx:.1f}", f"cy={roi_cy:.1f}"])
            w.writerow(["# Initial pose",
                        f"yaw={init_yaw:.3f}", f"pitch={init_pitch:.3f}"])
            w.writerow(["# Target zoom", f"x{self.current_target_zoom:.1f}"])
            w.writerow([])
            w.writerow(["method","zoom","label","ex_px","ey_px","err_px","duration_s"])
            for mname, records in self.benchmark_results.items():
                if not records:
                    w.writerow([mname,"","FAILED","","","",""]); continue
                for r in records:
                    w.writerow([
                        mname, r["zoom"], r["label"],
                        f"{r['ex']:.2f}"        if r["ex"]       is not None else "NA",
                        f"{r['ey']:.2f}"        if r["ey"]       is not None else "NA",
                        f"{r['err_px']:.2f}"    if r["err_px"]   is not None else "NA",
                        f"{r['duration_s']:.4f}" if r.get("duration_s") is not None else "NA"
                    ])
        print(f"  CSV saved: {csv_path}")

        labels      = []; data_err = []; data_ex = []; data_ey = []
        mean_times  = []
        colors      = plt.cm.tab10(np.linspace(0, 1, len(METHOD_NAMES)))
        for mname in METHOD_NAMES:
            records = self.benchmark_results.get(mname, [])
            valid   = [r for r in records if r["err_px"] is not None and r["zoom"] > 1.0]
            labels.append(mname)
            data_err.append([r["err_px"] for r in valid])
            data_ex.append( [r["ex"]     for r in valid])
            data_ey.append( [r["ey"]     for r in valid])
            durations = [r["duration_s"] for r in records
                         if r.get("duration_s") is not None and r["zoom"] > 1.0]
            mean_times.append(np.mean(durations) if durations else 0.0)

        title = (f"Benchmark {ts}\n"
                 f"ROI {wr}x{hr}px @ ({roi_cx:.0f},{roi_cy:.0f})"
                 f"  zoom x{self.current_target_zoom:.1f}")

        fig, axes = plt.subplots(1, 3, figsize=(16, 7))
        fig.suptitle(title, fontsize=11, y=1.02)
        fig.subplots_adjust(top=0.88)
        for ax, data, ttl, ylabel in [
            (axes[0], data_err, "Euclidean error", "err (px)"),
            (axes[1], data_ex,  "X error (ex)",    "ex (px)"),
            (axes[2], data_ey,  "Y error (ey)",    "ey (px)"),
        ]:
            for i, (d, col) in enumerate(zip(data, colors)):
                if not d: continue
                bp = ax.boxplot([d], positions=[i+1], patch_artist=True,
                                medianprops=dict(color="black", linewidth=2),
                                widths=0.6)
                bp["boxes"][0].set_facecolor(col)
                bp["boxes"][0].set_alpha(0.6)
            ax.set_xticks(range(1, len(labels)+1))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title(ttl); ax.set_ylabel(ylabel); ax.set_xlabel("Method")
            ax.grid(True, axis="y", alpha=0.3)
            if ttl == "Euclidean error":
                ax.axhline(MICRO_CORRECT_PX, color="red", linestyle=":", alpha=0.6,
                           label=f"micro thr ({MICRO_CORRECT_PX}px)")
                ax.legend(fontsize=8)
            else:
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Plot saved: {plot_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Mouse / ROI
    # ══════════════════════════════════════════════════════════════════════════
    def _mouse_cb(self, event, x, y, flags, param):
        if self.is_busy or self.tracking_active: return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing   = True
            self.start_x,   self.start_y   = x, y
            self.current_x, self.current_y = x, y
            self.roi_selected = self.template_saved = False
            self.template_gray = None
            self.last_target_px = self.last_error_px = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x, self.current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x0 = min(self.start_x, x); y0 = min(self.start_y, y)
            w  = abs(x - self.start_x); h  = abs(y - self.start_y)
            if w > 10 and h > 10:
                self.roi          = (x0, y0, w, h)
                self._roi_orig_wh  = (w, h)
                self.roi_selected = True
                self.optimal_zoom = self._best_zoom(w, h)
                self.status = (f"ROI {w}x{h}px | zoom rec. x{self.optimal_zoom:.1f}."
                               f" Press 'c'")

    # ══════════════════════════════════════════════════════════════════════════
    # Overlay
    # ══════════════════════════════════════════════════════════════════════════
    def _draw_overlay(self, img):
        # ── Foreground mask overlay (drawn first, underneath everything else) ──
        if self.tracking_active and self._track_mask is not None:
            mask = self._track_mask
            x1, y1 = self._track_mask_x1, self._track_mask_y1
            h, w = mask.shape[:2]
            x2 = min(IMG_W, x1 + w)
            y2 = min(IMG_H, y1 + h)
            mh, mw = y2 - y1, x2 - x1
            if mh > 0 and mw > 0:
                crop_mask = mask[:mh, :mw]
                region    = img[y1:y2, x1:x2]
                # Semi-transparent green tint on foreground pixels only
                fg = crop_mask == 255
                tinted = region.copy()
                tinted[fg] = (region[fg].astype(np.int32) * 0.6
                              + np.array([0, 80, 0], dtype=np.int32)).clip(0, 255).astype(np.uint8)
                img[y1:y2, x1:x2] = tinted
                # Draw contour of the mask shape on the full image
                contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                shifted = [c + np.array([[[x1, y1]]]) for c in contours]
                cv2.drawContours(img, shifted, -1, (0, 255, 0), 2)

        if self.drawing:
            cv2.rectangle(img,
                          (self.start_x, self.start_y),
                          (self.current_x, self.current_y), (0,255,255), 1)

        if self.roi_selected and self.roi:
            x, y, w, h = [int(v) for v in self.roi]
            # During tracking, reposition box around the live target instead
            if self.tracking_active:
                with self._track_lock:
                    t_pos = self._track_target_px
                if t_pos is not None:
                    orig_w, orig_h = self._roi_orig_wh
                    tx, ty = int(t_pos[0]), int(t_pos[1])
                    x = tx - orig_w // 2
                    y = ty - orig_h // 2
                    w, h = orig_w, orig_h
            color = (0,255,255) if self.tracking_active else (0,255,0) if self.template_saved else (0,255,255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cx_roi, cy_roi = x + w//2, y + h//2
            cv2.drawMarker(img, (cx_roi, cy_roi),
                           (0,165,255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, f"Zoom x{self.current_target_zoom:.1f}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Tracking overlay
        with self._track_lock:
            t_px   = self._track_target_px
            t_err  = self._track_error_px
            t_lost = self._track_lost

        target_px = t_px if self.tracking_active else self.last_target_px
        error_px  = t_err if self.tracking_active else self.last_error_px

        if target_px is not None:
            tx, ty = int(target_px[0]), int(target_px[1])
            color_cross = (0,0,255) if (self.tracking_active and t_lost) else (0,165,255)
            cv2.drawMarker(img, (tx, ty), color_cross, cv2.MARKER_CROSS, 24, 3)

        if target_px is not None and error_px is not None:
            cx_c, cy_c = CURSORS[self.cursor_sel]
            tx, ty = int(target_px[0]), int(target_px[1])
            cv2.arrowedLine(img, (tx,ty), (cx_c, cy_c),
                            (255,255,0), 2, tipLength=0.15)

        for k, (cx, cy) in CURSORS.items():
            if self.cursor_sel in (0, k):
                cv2.drawMarker(img, (cx,cy), CURSOR_COLORS[k],
                               cv2.MARKER_CROSS, 20, 2)

        # Status bar
        cv2.putText(img, self.status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        if self.is_busy:
            cv2.putText(img, "AUTO MODE - Controls Locked",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

        if self.tracking_active:
            label = "TARGET LOST" if t_lost else "TRACKING"
            color = (0,0,255) if t_lost else (0,255,0)
            cv2.putText(img, label,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        if self._recording:
            cv2.circle(img, (IMG_W - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(img, "REC", (IMG_W - 60, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ══════════════════════════════════════════════════════════════════════════
    # Recording (ffmpeg pipe — avoids OpenCV VideoWriter segfault)
    # ══════════════════════════════════════════════════════════════════════════
    def _start_recording(self):
        with self._rec_lock:
            if self._recording:
                return
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(self.video_dir, f"rec_{ts}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{IMG_W}x{IMG_H}",
                "-pix_fmt", "bgr24",
                "-r", "30",
                "-i", "pipe:0",
                "-vcodec", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                out
            ]
            try:
                self._ffmpeg_proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._recording = True
                self.status = f"REC | {os.path.basename(out)}"
                print(f"\n⏺  Recording started: {out}")
            except FileNotFoundError:
                print("❌ ffmpeg not found. Install with: sudo apt install ffmpeg")

    def _stop_recording(self):
        with self._rec_lock:
            if not self._recording:
                return
            self._recording = False
            if self._ffmpeg_proc is not None:
                try:
                    self._ffmpeg_proc.stdin.close()
                    self._ffmpeg_proc.wait(timeout=5.0)
                except Exception:
                    self._ffmpeg_proc.kill()
                self._ffmpeg_proc = None
            self.status = "Recording saved."
            print("⏹  Recording stopped.")

    def _record_frame(self, frame):
        """Write one BGR frame to the ffmpeg pipe. Called from the main loop."""
        with self._rec_lock:
            if not self._recording or self._ffmpeg_proc is None:
                return
            try:
                self._ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                self._recording = False
                self._ffmpeg_proc = None
                print("⚠️  Recording pipe closed unexpectedly.")

    # ══════════════════════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════════════════════
    def _main_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            frame = self._get_frame()
            if frame is not None:
                vis = frame.copy()
                self._draw_overlay(vis)
                cv2.imshow("ORB Correct Once", vis)
                # Feed the overlay frame to the recorder
                self._record_frame(vis)

            key      = cv2.waitKeyEx(1)
            if key == -1:
                rate.sleep(); continue

            char_key = key & 0xFF

            if   char_key in (ord('w'), ord('W')): self._move_manual(0, -1)
            elif char_key in (ord('s'), ord('S')): self._move_manual(0,  1)
            elif char_key in (ord('a'), ord('A')): self._move_manual(1,  0)
            elif char_key in (ord('d'), ord('D')): self._move_manual(-1, 0)

            elif char_key == 27:   # ESC
                if self.tracking_active:
                    self._stop_tracking()
                self._stop_recording()
                break

            elif char_key in (ord('q'), ord('Q')):
                if self.tracking_active:
                    self._stop_tracking()

            elif char_key in (ord('g'), ord('G')):
                if self._recording:
                    self._stop_recording()
                else:
                    self._start_recording()

            elif char_key == ord('r') and not self.is_busy and not self.tracking_active:
                self.roi = None
                self.roi_selected = self.template_saved = False
                self.template_gray = None
                self.last_target_px = self.last_error_px = None
                self._set_zoom(1.0)
                self.current_zoom = 1.0
                self.status = "Reset. Ready."

            elif char_key == ord('k') and not self.is_busy and not self.tracking_active:
                threading.Thread(target=self.auto_calibrate_lens, daemon=True).start()

            elif char_key == ord('c') and self.roi_selected \
                    and not self.is_busy and not self.tracking_active:
                frame = self._get_frame()
                if frame is not None and self.roi is not None:
                    x, y, w, h = [int(v) for v in self.roi]
                    self.template_gray  = cv2.cvtColor(
                        frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    self.template_saved = True
                    self.status = "Template saved. Press 'z' to zoom+correct."

            elif char_key == ord('z') and self.template_saved \
                    and not self.is_busy and not self.tracking_active:
                tz = self.optimal_zoom
                self.is_busy             = True
                self.current_target_zoom = tz
                self.last_target_px      = None
                self.last_error_px       = None
                threading.Thread(
                    target=self._zoom_and_correct, daemon=True).start()

            elif char_key == ord('t') and not self.is_busy:
                self._start_tracking()

            elif char_key in (ord('0'), ord('1'), ord('2'), ord('3')):
                self.cursor_sel = int(chr(char_key))

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        OrbCorrectOnce()
    except rospy.ROSInterruptException:
        pass
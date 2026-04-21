#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optflow_correct_once_tracking.py
=================================
Live PTZ zoom corrector — OPTICAL FLOW (Lucas-Kanade), single correction
+ continuous closed-loop tracking mode after zoom.

Workflow:
  1. Draw ROI → 'c' lock target
  2. 'z' → zoom + correct (existing pipeline)
  3. 't' → start LK tracking at current zoom level
  4. 't' again or 'q' → stop tracking
  5. Everything else unchanged (WASD, r, k, 1/2/3, ESC)

Tracking design:
  - Runs in a background thread (does not block the display loop)
  - Lucas-Kanade sparse optical flow between consecutive frames
  - Affine warp compensation: same zoom_prev==zoom_curr, so scale=1, no warp needed
  - Proportional controller (TRACK_KP) with deadband (TRACK_DEADBAND_PX)
  - Periodic keyframe refresh every TRACK_KEYFRAME_S seconds
  - Lost-target detection after TRACK_LOST_FRAMES consecutive failures
  - _roi_orig_wh stored once at draw time — never mutated by tracking thread
"""

import rospy
import cv2
import numpy as np
import os
import threading
import time
import math
import csv
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
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
THRESHOLD_RECOMENDED_ZOOM = 0.40   # ← your modification

ZOOM_FOVS_REF = {
    1.0: (63.7, 35.84), 2.0: (56.9, 31.2), 3.0: (50.7, 27.3), 4.0: (45.9, 24.5),
    5.0: (40.5, 21.6), 6.0: (37.4, 19.6), 7.0: (32.2, 17.2), 8.0: (29.1, 15.2),
    9.0: (25.3, 13.0), 10.0: (21.7, 11.1), 11.0: (18.3, 9.3), 12.0: (15.2, 7.7),
    13.0: (10.0, 6.2), 14.0: (7.8, 4.8), 15.0: (6.2, 3.6), 16.0: (5.2, 2.9),
    17.0: (4.1, 2.3), 18.0: (3.5, 1.9), 19.0: (2.9, 1.7), 20.0: (2.3, 1.3)
}

MIN_INLIERS  = 6
STABILISE_S  = 1.5
EPICENTER_SIZE = 450

MICRO_CORRECT_PX = 15
METHOD_NAMES = ["OpticalFlow"]

# ── Tracking parameters (your tuned values) ────────────────────────────────────
TRACK_HZ           = 10      # control-loop frequency (Hz)
TRACK_DEADBAND_PX  = 15      # pixel deadband — no command below this error
TRACK_KP           = 0.8     # proportional gain
TRACK_SPEED_FAST   = 25      # motor speed when error > 5°
TRACK_SPEED_SLOW   = 12      # motor speed for fine corrections
TRACK_KEYFRAME_S   = 1       # seconds between forced keyframe refreshes
TRACK_LOST_FRAMES  = 8       # consecutive LK failures before "target lost"
TRACK_EPICENTER    = 250     # search window around last known position (px)


# ══════════════════════════════════════════════════════════════════════════════
class OpticalFlowCorrectOnce:

    def __init__(self):
        rospy.init_node('opticalflow_correct_once', anonymous=True)
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
        self._roi_orig_wh   = (0, 0)   # (w, h) at zoom x1, never scaled
        self.roi_selected   = False
        self.template_saved = False
        self.optimal_zoom   = 1.0
        self.current_zoom   = 1.0
        self.current_target_zoom = 1.0

        self.cursor_sel  = 1
        self.is_busy     = False

        self.last_target_px = None
        self.last_error_px  = None
        self.last_inliers   = 0

        self.benchmark_results = {m: [] for m in METHOD_NAMES}
        self.status = "Draw a ROI around the target, then press 'c'"

        # ── Tracking state ─────────────────────────────────────────────────────
        self.tracking_active  = False
        self._track_thread    = None
        self._track_target_px = None
        self._track_error_px  = None
        self._track_lost      = False
        self._track_lock      = threading.Lock()
        # ──────────────────────────────────────────────────────────────────────

        rospack  = rospkg.RosPack()
        pkg_path = rospack.get_path("pan_tilt_description")
        self.image_dir = os.path.join(pkg_path, "images")
        os.makedirs(self.image_dir, exist_ok=True)

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

        cv2.namedWindow("Optical Flow Correct Once")
        cv2.setMouseCallback("Optical Flow Correct Once", self._mouse_cb)

        print("\n🟢 optflow_correct_once_tracking started")
        print("   Draw ROI → 'c' lock target → 'z' zoom+correct (recommended) → 't' track")
        print("   't' again or 'q' → stop tracking")
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
        except Exception as e:
            rospy.logwarn(f"Image error: {e}")

    def _status_cb(self, msg):
        self.current_yaw   = msg.yaw_now
        self.current_pitch = msg.pitch_now

    def _get_frame(self):
        with self.image_lock:
            return self.image.copy() if self.image is not None else None

    # ══════════════════════════════════════════════════════════════════════════
    # PTZ commands
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

    def _calculate_and_send_cmd(self, ex, ey, zoom_level):
        """Direct incremental command — used by zoom pipeline (unchanged)."""
        fov_h, fov_v = self._get_fovs(zoom_level)
        px_per_deg_x = IMG_W / fov_h
        px_per_deg_y = IMG_H / fov_v
        ang_x = -ex / px_per_deg_x
        ang_y  =  ey / px_per_deg_y
        error_mag = math.sqrt(ang_x**2 + ang_y**2)
        speed_val = 25 if error_mag > 5.0 else 15
        nuevo_yaw   = max(self.PAN_MIN_DEG,  min(self.PAN_MAX_DEG,  self.current_yaw   + ang_x))
        nuevo_pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, self.current_pitch + ang_y))
        nuevo_yaw   = round(nuevo_yaw)
        nuevo_pitch = round(nuevo_pitch)
        self._send_cmd(nuevo_yaw, nuevo_pitch, speed=speed_val)
        return nuevo_yaw, nuevo_pitch

    def _calculate_and_send_proportional_cmd(self, ex, ey, zoom_level):
        """Proportional controller for tracking — scales error by TRACK_KP."""
        return self._calculate_and_send_cmd(ex * TRACK_KP, ey * TRACK_KP, zoom_level)

    # ══════════════════════════════════════════════════════════════════════════
    # ROI / geometry helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _update_roi_visual(self, cx, cy, zoom_level, orig_w, orig_h):
        fx, fy = self._get_zoom_factors(zoom_level)
        new_w, new_h = orig_w * fx, orig_h * fy
        new_x, new_y = cx - new_w / 2.0, cy - new_h / 2.0
        self.roi = (int(new_x), int(new_y), int(new_w), int(new_h))

    def _get_roi_bbox(self, cx, cy, crop_w, crop_h):
        x1 = max(0, int(cx - crop_w / 2.0))
        y1 = max(0, int(cy - crop_h / 2.0))
        x2 = min(IMG_W, int(cx + crop_w / 2.0))
        y2 = min(IMG_H, int(cy + crop_h / 2.0))
        return x1, y1, x2 - x1, y2 - y1

    def _best_zoom(self, w, h):
        for z_int in range(200, 9, -1):
            z = round(z_int / 10.0, 1)
            fx, fy = self._get_zoom_factors(z)
            if w * fx <= IMG_W * THRESHOLD_RECOMENDED_ZOOM and \
               h * fy <= IMG_H * THRESHOLD_RECOMENDED_ZOOM:
                return z
        return 1.0

    def _move_manual(self, dyaw, dpitch):
        if self.is_busy or self.tracking_active: return
        target_yaw   = max(self.PAN_MIN_DEG,  min(self.PAN_MAX_DEG,  round(self.current_yaw)   + dyaw))
        target_pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, round(self.current_pitch) + dpitch))
        self.status  = f"🕹️ Manual: yaw={target_yaw}° pitch={target_pitch}°"
        self._send_cmd(target_yaw, target_pitch, speed=15)

    # ══════════════════════════════════════════════════════════════════════════
    # Optical Flow tracker (unchanged from original)
    # ══════════════════════════════════════════════════════════════════════════
    def _find_point_optical_flow(self, prev_gray, curr_gray,
                                  target_x, target_y,
                                  roi_x1, roi_y1, roi_w, roi_h,
                                  zoom_prev, zoom_curr,
                                  guess_dx=0.0, guess_dy=0.0):
        """Lucas-Kanade tracker with affine zoom-warp and mechanical jump prediction."""
        fov_h1, _ = self._get_fovs(zoom_prev)
        fov_h2, _ = self._get_fovs(zoom_curr)
        scale = fov_h1 / fov_h2

        if abs(scale - 1.0) > 0.02:
            cx_cam, cy_cam = IMG_W / 2.0, IMG_H / 2.0
            M = cv2.getRotationMatrix2D((cx_cam, cy_cam), 0, scale)
            prev_gray = cv2.warpAffine(prev_gray, M, (IMG_W, IMG_H))
            pt = np.float32([[[target_x, target_y]]])
            target_x, target_y = cv2.transform(pt, M)[0][0]
            roi_c = np.float32([[[roi_x1 + roi_w / 2.0, roi_y1 + roi_h / 2.0]]])
            sim_roi_c = cv2.transform(roi_c, M)[0][0]
            roi_w, roi_h = int(roi_w * scale), int(roi_h * scale)
            roi_x1 = int(sim_roi_c[0] - roi_w / 2.0)
            roi_y1 = int(sim_roi_c[1] - roi_h / 2.0)

        mask = np.zeros_like(prev_gray)
        rx1, ry1 = max(0, roi_x1), max(0, roi_y1)
        rx2, ry2 = min(IMG_W, roi_x1 + roi_w), min(IMG_H, roi_y1 + roi_h)
        if rx2 <= rx1 or ry2 <= ry1: return None
        mask[ry1:ry2, rx1:rx2] = 255

        p0 = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=500, qualityLevel=0.02,
            minDistance=5, mask=mask)
        if p0 is None or len(p0) < MIN_INLIERS: return None

        lk_params = dict(winSize=(45, 45), maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                   30, 0.01))

        if abs(guess_dx) > 0.1 or abs(guess_dy) > 0.1:
            p1_guess = p0.copy()
            p1_guess[:, 0, 0] += guess_dx
            p1_guess[:, 0, 1] += guess_dy
            lk_params['flags'] = cv2.OPTFLOW_USE_INITIAL_FLOW
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, p1_guess, **lk_params)
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, None, **lk_params)

        if p1 is None: return None

        good_new, good_old = p1[st == 1], p0[st == 1]
        if len(good_new) < MIN_INLIERS: return None

        src_pts = good_old.reshape(-1, 1, 2)
        dst_pts = good_new.reshape(-1, 1, 2)
        H, mask_hom = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask_hom is None or int(mask_hom.sum()) < MIN_INLIERS:
            return None

        pt_final  = np.float32([[target_x, target_y]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pt_final, H)

        return (float(projected[0, 0, 0]), float(projected[0, 0, 1]),
                int(mask_hom.sum()), H, mask_hom, src_pts, dst_pts, None)

    # ══════════════════════════════════════════════════════════════════════════
    # Auto-calibration (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def auto_calibrate_lens(self):
        if self.is_busy: return
        self.is_busy = True
        try:
            self.status = "Starting chained auto-calibration..."
            print(f"\n{self.status}")
            fov_w, fov_t, z_max = 63.7, 2.3, 20.0
            zoom_steps = [1.0, 3.0, 5.0, 8.0, 11.0]
            valores_k = []

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
                self.status = f"Applying zoom x{z_target}..."
                print(self.status)
                self._set_zoom(z_target)
                time.sleep(2.5)
                frame_curr = self._get_frame()
                gray_curr  = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)
                roi_size = 400
                roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(
                    cx_prev, cy_prev, roi_size, roi_size)
                self.status = f"Computing deformation (x{z_prev} -> x{z_target})..."
                res = self._find_point_optical_flow(
                    gray_prev, gray_curr, cx_prev, cy_prev,
                    roi_x1, roi_y1, roi_w, roi_h, z_prev, z_target)
                if res is None:
                    print(f"⚠️  LK lost reference at x{z_target}."); break
                cx_found, cy_found, _, H, _, _, _, _ = res
                pts_orig = np.float32(
                    [[cx_prev - 50, cy_prev],
                     [cx_prev + 50, cy_prev]]).reshape(-1, 1, 2)
                pts_proy = cv2.perspectiveTransform(pts_orig, H)
                dist_nueva = np.linalg.norm(pts_proy[0, 0] - pts_proy[1, 0])
                escala_relativa = dist_nueva / 100.0
                escala_acumulada *= escala_relativa
                tan_w    = math.tan(math.radians(fov_w / 2.0))
                fov_real = 2.0 * math.degrees(math.atan(tan_w / escala_acumulada))
                t = (z_target - 1.0) / (z_max - 1.0)
                numerador = (fov_real - fov_t * t) / fov_w
                if numerador > 0 and (1.0 - t) > 0:
                    k_paso = math.log(numerador) / math.log(1.0 - t)
                    valores_k.append(k_paso)
                    print(f"  -> S_total={escala_acumulada:.2f}x "
                          f"| FOV={fov_real:.1f}° | k={k_paso:.3f}")
                gray_prev = gray_curr
                cx_prev   = cx_found
                cy_prev   = cy_found

            if valores_k:
                k_new = max(valores_k)
                self.status = f"✅ Calibration done! k = {k_new:.3f}"
                print(f"\n{self.status}")
            else:
                self.status = "❌ Failed. Point at a textured wall."
                print(self.status)
        except Exception as e:
            print(f"Calibration error: {e}")
        finally:
            self._set_zoom(1.0)
            self.is_busy = False
            self.status  = "Manual mode ready."

    # ══════════════════════════════════════════════════════════════════════════
    # Zoom + correct pipeline (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def _zoom_and_correct(self):
        try:
            self.benchmark_results = {m: [] for m in METHOD_NAMES}
            target_zoom = self.current_target_zoom
            cx_cursor, cy_cursor = CURSORS[self.cursor_sel]
            x, y, orig_w, orig_h = [int(v) for v in self.roi]

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
                self.status = "❌ No frame at x1.0"; return

            initial_cx, initial_cy = x + orig_w / 2.0, y + orig_h / 2.0
            crop_w_init = max(int(orig_w * 1.5), 250)
            crop_h_init = max(int(orig_h * 1.5), 250)
            frame_orig_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(
                initial_cx, initial_cy, crop_w_init, crop_h_init)

            ex_init, ey_init = initial_cx - cx_cursor, initial_cy - cy_cursor

            if abs(ex_init) > 40 or abs(ey_init) > 40:
                self.status = "Pre-centering target..."
                self._calculate_and_send_cmd(ex_init, ey_init, 1.0)
                time.sleep(2.0)
                frame1 = self._get_frame()
                scene_gray_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                res0 = self._find_point_optical_flow(
                    frame_orig_gray, scene_gray_1,
                    initial_cx, initial_cy,
                    roi_x1, roi_y1, roi_w, roi_h,
                    1.0, 1.0,
                    guess_dx=-ex_init, guess_dy=-ey_init)
                if res0 is None:
                    self.status = "❌ Failed to re-acquire after pre-centering"; return
                current_cx, current_cy = res0[0], res0[1]
                frame_base = frame1
            else:
                current_cx, current_cy = initial_cx, initial_cy
                frame_base = frame_orig

            self._update_roi_visual(current_cx, current_cy, 1.0, orig_w, orig_h)
            prev_gray = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)
            roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(
                current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)
            zoom_prev = 1.0

            MAX_SAFE_JUMP = 2.0
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

            H_final, n_inliers = None, 0

            for step_zoom in zoom_steps:
                self.status = f"Applying zoom x{step_zoom:.1f}..."
                self._set_zoom(step_zoom)
                time.sleep(STABILISE_S)

                frame_z = self._get_frame()
                if frame_z is None: return
                scene_gray_before = cv2.cvtColor(frame_z, cv2.COLOR_BGR2GRAY)

                self.status = f"Tracking Optical Flow at x{step_zoom:.1f}..."
                t_start = time.time()
                result = self._find_point_optical_flow(
                    prev_gray, scene_gray_before,
                    current_cx, current_cy,
                    roi_x1, roi_y1, roi_w, roi_h,
                    zoom_prev, step_zoom)
                match_dt = time.time() - t_start

                if result is None:
                    self.status = f"❌ Optical Flow failed at x{step_zoom:.1f}"; return

                cx_found, cy_found, n_inliers, H_final, mask, kp1, kp2, good = result
                ex_step, ey_step = cx_found - cx_cursor, cy_found - cy_cursor
                err_px = math.sqrt(ex_step**2 + ey_step**2)

                self.benchmark_results["OpticalFlow"].append({
                    "zoom": step_zoom, "label": f"x{step_zoom:.1f}",
                    "ex": ex_step, "ey": ey_step,
                    "err_px": err_px, "duration_s": match_dt
                })

                if abs(ex_step) > MICRO_CORRECT_PX or abs(ey_step) > MICRO_CORRECT_PX:
                    self.status = f"Micro-centering at x{step_zoom:.1f}..."
                    self._calculate_and_send_cmd(ex_step, ey_step, step_zoom)
                    time.sleep(2.0)
                    frame_z_after = self._get_frame()
                    scene_gray_after = cv2.cvtColor(frame_z_after, cv2.COLOR_BGR2GRAY)
                    roi_x1_mc, roi_y1_mc, roi_w_mc, roi_h_mc = self._get_roi_bbox(
                        cx_found, cy_found, EPICENTER_SIZE, EPICENTER_SIZE)
                    res_centered = self._find_point_optical_flow(
                        scene_gray_before, scene_gray_after,
                        cx_found, cy_found,
                        roi_x1_mc, roi_y1_mc, roi_w_mc, roi_h_mc,
                        step_zoom, step_zoom,
                        guess_dx=-ex_step, guess_dy=-ey_step)
                    if res_centered is not None:
                        cx_found, cy_found, n_inliers, H_final, _, _, _, _ = res_centered
                    else:
                        cx_found, cy_found = cx_cursor, cy_cursor
                    scene_gray_final = scene_gray_after
                else:
                    scene_gray_final = scene_gray_before

                current_cx, current_cy = cx_found, cy_found
                self._update_roi_visual(current_cx, current_cy, step_zoom, orig_w, orig_h)

                if step_zoom != zoom_steps[-1]:
                    prev_gray = scene_gray_final
                    zoom_prev = step_zoom
                    roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(
                        current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)

            self.last_target_px = (current_cx, current_cy)
            self.last_inliers   = n_inliers
            ex_final, ey_final  = current_cx - cx_cursor, current_cy - cy_cursor
            self.last_error_px  = (ex_final, ey_final)

            yaw_final, pitch_final = self._calculate_and_send_cmd(
                ex_final, ey_final, target_zoom)
            self._update_roi_visual(cx_cursor, cy_cursor, target_zoom, orig_w, orig_h)

            self.status = (f"✅ Done: yaw={yaw_final}° pitch={pitch_final}°"
                           f" in={n_inliers} — press 't' to track")
            print(f"\n{self.status}\n")

            if H_final is not None:
                frame_last = frame_z_after if 'frame_z_after' in locals() else frame_z
                self._save_debug(frame_last, H_final,
                                 current_cx, current_cy,
                                 cx_cursor, cy_cursor,
                                 roi_x1, roi_y1, roi_w, roi_h)

            ts_str  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.image_dir, f"benchmark_{ts_str}")
            os.makedirs(out_dir, exist_ok=True)
            self._save_benchmark_results(self.roi, self.current_yaw,
                                         self.current_pitch, out_dir, ts_str)
        except Exception as e:
            self.status = f"❌ Error: {e}"
            print(e)
        finally:
            self.is_busy = False

    # ══════════════════════════════════════════════════════════════════════════
    # ████  TRACKING MODE  ████
    # ══════════════════════════════════════════════════════════════════════════
    def _start_tracking(self):
        if not self.template_saved:
            print("⚠️  Lock target first (press 'c' after drawing ROI)."); return
        if self.is_busy:
            print("⚠️  Zoom pipeline still running."); return
        if self.tracking_active:
            self._stop_tracking(); return

        self.tracking_active = True
        with self._track_lock:
            self._track_target_px = self.last_target_px
            self._track_error_px  = None
            self._track_lost      = False

        self._track_thread = threading.Thread(
            target=self._tracking_loop, daemon=True)
        self._track_thread.start()
        self.status = (f"🎯 Tracking active at zoom x{self.current_target_zoom:.1f}"
                       f" — 't'/'q' to stop")
        print(f"\n{self.status}")

    def _stop_tracking(self):
        self.tracking_active = False
        if self._track_thread is not None:
            self._track_thread.join(timeout=2.0)
            self._track_thread = None
        self.status = "Tracking stopped. Manual mode ready."
        print(f"\n{self.status}")

    def _tracking_loop(self):
        """
        Background thread: continuous LK optical-flow closed-loop tracking.

        Design decisions
        ----------------
        Frame-to-frame LK:  Unlike the zoom pipeline (which tracks across zoom
            levels using the affine warp), the tracking loop always runs at a
            fixed zoom_prev == zoom_curr == zoom_level, so scale == 1.0 and the
            warp is a no-op.  This makes LK run fast and clean.

        ROI mask:  We restrict goodFeaturesToTrack to a window of
            TRACK_EPICENTER px around the last known target position,
            avoiding drift onto background features.

        Keyframe refresh:  Every TRACK_KEYFRAME_S seconds prev_gray is
            replaced by the current frame.  This prevents feature drift when the
            target moves significantly or illumination changes.

        Proportional control + deadband:  Error is scaled by TRACK_KP before
            conversion to degrees.  Commands below TRACK_DEADBAND_PX are
            suppressed to avoid motor chatter.

        Lost-target:  After TRACK_LOST_FRAMES consecutive LK failures the
            thread marks _track_lost=True and holds position.  Recovers
            automatically when LK succeeds again.
        """
        zoom_level  = self.current_target_zoom
        cx_cursor, cy_cursor = CURSORS[self.cursor_sel]

        # Snapshot ROI dimensions once — self.roi may be mutated by other threads
        if self.roi is not None:
            _, _, snap_w, snap_h = self.roi
            window_w = max(20, int(snap_w * 0.70))
            window_h = max(20, int(snap_h * 0.70))
        else:
            window_w = TRACK_EPICENTER
            window_h = TRACK_EPICENTER

        frame = self._get_frame()
        if frame is None:
            print("❌ [Track] No frame available at start."); return

        with self._track_lock:
            init_pos = self._track_target_px

        if init_pos is not None:
            current_cx, current_cy = float(init_pos[0]), float(init_pos[1])
        else:
            current_cx, current_cy = float(cx_cursor), float(cy_cursor)

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_keyframe_t = time.time()
        fail_count      = 0
        dt              = 1.0 / TRACK_HZ

        print(f"  [Track/LK] Loop started — zoom x{zoom_level:.1f}, "
              f"cursor ({cx_cursor},{cy_cursor})")

        while self.tracking_active and not rospy.is_shutdown():
            t0 = time.time()

            # ── Refresh keyframe periodically ──────────────────────────────
            if (time.time() - last_keyframe_t) >= TRACK_KEYFRAME_S:
                frame_kf = self._get_frame()
                if frame_kf is not None:
                    prev_gray = cv2.cvtColor(frame_kf, cv2.COLOR_BGR2GRAY)
                    last_keyframe_t = time.time()

            # ── Grab current frame ─────────────────────────────────────────
            frame_curr = self._get_frame()
            if frame_curr is None:
                time.sleep(dt); continue

            curr_gray = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

            # ── LK tracking: prev_gray → curr_gray ─────────────────────────
            # zoom_prev == zoom_curr == zoom_level, so no affine warp needed
            roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(
                current_cx, current_cy,
                window_w, window_h)

            result = self._find_point_optical_flow(
                prev_gray, curr_gray,
                current_cx, current_cy,
                roi_x1, roi_y1, roi_w, roi_h,
                zoom_level, zoom_level)   # same zoom → scale = 1, no warp

            if result is None:
                fail_count += 1
                lost_now = fail_count >= TRACK_LOST_FRAMES
                with self._track_lock:
                    self._track_lost = lost_now
                if lost_now:
                    self.status = "⚠️  [Track] Target LOST — searching..."
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
                continue

            # ── Successful match ───────────────────────────────────────────
            fail_count = 0
            cx_found, cy_found, n_inliers, _, _, _, _, _ = result

            ex = cx_found - cx_cursor
            ey = cy_found - cy_cursor
            err_px = math.sqrt(ex**2 + ey**2)

            with self._track_lock:
                self._track_target_px = (cx_found, cy_found)
                self._track_error_px  = (ex, ey)
                self._track_lost      = False

            # ── Proportional control with deadband ─────────────────────────
            if err_px > TRACK_DEADBAND_PX:
                self._calculate_and_send_proportional_cmd(ex, ey, zoom_level)
                # Optimistic position update: assume motor moves TRACK_KP fraction
                current_cx = cx_found - ex * TRACK_KP * 0.5
                current_cy = cy_found - ey * TRACK_KP * 0.5
            else:
                current_cx, current_cy = cx_found, cy_found

            # Update prev_gray for next LK iteration
            prev_gray = curr_gray

            deg_x, deg_y = self._px_to_deg(ex, ey, zoom_level)
            err_deg = math.sqrt(deg_x**2 + deg_y**2)
            self.status = (f"🎯 Tracking | err={err_px:.1f}px ({err_deg:.3f}°)"
                           f" | in={n_inliers}")

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

        print("  [Track/LK] Loop exited.")

    # ══════════════════════════════════════════════════════════════════════════
    # Debug / benchmark save helpers (unchanged)
    # ══════════════════════════════════════════════════════════════════════════
    def _save_debug(self, frame_z, H,
                    cx_found, cy_found, cx_cursor, cy_cursor,
                    roi_x1, roi_y1, roi_w, roi_h):
        debug = frame_z.copy()
        corners = np.float32([
            [roi_x1, roi_y1], [roi_x1+roi_w, roi_y1],
            [roi_x1+roi_w, roi_y1+roi_h], [roi_x1, roi_y1+roi_h]
        ]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)
        cv2.polylines(debug, [np.int32(proj)], True, (0,165,255), 2)
        cv2.drawMarker(debug, (int(cx_found), int(cy_found)),
                       (0,165,255), cv2.MARKER_CROSS, 24, 2)
        color = CURSOR_COLORS[self.cursor_sel]
        cv2.drawMarker(debug, (cx_cursor, cy_cursor),
                       color, cv2.MARKER_CROSS, 24, 2)
        cv2.arrowedLine(debug, (int(cx_found), int(cy_found)),
                        (cx_cursor, cy_cursor), (255,255,0), 2, tipLength=0.15)
        cv2.imwrite(os.path.join(self.image_dir,
                                 "opticalflow_correction_debug.jpg"), debug)

    def _save_benchmark_results(self, roi, init_yaw, init_pitch, out_dir, ts):
        xr, yr, wr, hr = [int(v) for v in roi]
        roi_cx, roi_cy = xr + wr / 2.0, yr + hr / 2.0

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
                        f"{r['ex']:.2f}"         if r["ex"]       is not None else "NA",
                        f"{r['ey']:.2f}"         if r["ey"]       is not None else "NA",
                        f"{r['err_px']:.2f}"     if r["err_px"]   is not None else "NA",
                        f"{r['duration_s']:.4f}" if r.get("duration_s") is not None else "NA"
                    ])
        print(f"  CSV saved: {csv_path}")

        labels = []; data_err = []; data_ex = []; data_ey = []; mean_times = []
        colors = plt.cm.tab10(np.linspace(0, 1, len(METHOD_NAMES)))
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
        plt.savefig(os.path.join(out_dir, "comparison.png"),
                    dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Plot saved: comparison.png")

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.suptitle(f"Execution time — {title}", fontsize=11, y=1.02)
        fig2.subplots_adjust(top=0.85)
        bars = ax2.bar(range(len(METHOD_NAMES)), mean_times, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(METHOD_NAMES)))
        ax2.set_xticklabels(METHOD_NAMES, rotation=30, ha="right")
        ax2.set_title("Mean match time per step")
        ax2.set_ylabel("Time (s)"); ax2.set_xlabel("Method")
        ax2.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, mean_times):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                         f"{val:.3f}s", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "execution_time.png"),
                    dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Time plot saved: execution_time.png")
        print(f"  Output dir: {out_dir}")

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
            self.last_target_px = self.last_error_px = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x, self.current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x0 = min(self.start_x, x); y0 = min(self.start_y, y)
            w  = abs(x - self.start_x); h  = abs(y - self.start_y)
            if w > 10 and h > 10:
                self.roi          = (x0, y0, w, h)
                self._roi_orig_wh = (w, h)
                self.roi_selected = True
                self.optimal_zoom = self._best_zoom(w, h)
                self.status = (f"ROI {w}×{h}px → zoom rec. x{self.optimal_zoom:.1f}."
                               f" Press 'c'")

    # ══════════════════════════════════════════════════════════════════════════
    # Overlay
    # ══════════════════════════════════════════════════════════════════════════
    def _draw_overlay(self, img):
        if self.drawing:
            cv2.rectangle(img,
                          (self.start_x, self.start_y),
                          (self.current_x, self.current_y), (0,255,255), 1)

        if self.roi_selected and self.roi:
            x, y, w, h = [int(v) for v in self.roi]
            # During tracking, reposition box around live target
            if self.tracking_active:
                with self._track_lock:
                    t_pos = self._track_target_px
                if t_pos is not None:
                    orig_w, orig_h = self._roi_orig_wh
                    tx, ty = int(t_pos[0]), int(t_pos[1])
                    x = tx - orig_w // 2
                    y = ty - orig_h // 2
                    w, h = orig_w, orig_h
            color = ((0,255,255) if self.tracking_active
                     else (0,255,0) if self.template_saved
                     else (0,255,255))
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

        target_px = t_px  if self.tracking_active else self.last_target_px
        error_px  = t_err if self.tracking_active else self.last_error_px

        if target_px is not None:
            tx, ty = int(target_px[0]), int(target_px[1])
            color_cross = (0,0,255) if (self.tracking_active and t_lost) else (0,165,255)
            cv2.drawMarker(img, (tx, ty), color_cross, cv2.MARKER_CROSS, 24, 3)

        if target_px is not None and error_px is not None:
            cx_c, cy_c = CURSORS[self.cursor_sel]
            tx, ty = int(target_px[0]), int(target_px[1])
            cv2.arrowedLine(img, (tx, ty), (cx_c, cy_c),
                            (255,255,0), 2, tipLength=0.15)

        for k, (cx, cy) in CURSORS.items():
            if self.cursor_sel in (0, k):
                cv2.drawMarker(img, (cx, cy), CURSOR_COLORS[k],
                               cv2.MARKER_CROSS, 20, 2)

        cv2.putText(img, self.status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        if self.is_busy:
            cv2.putText(img, "AUTO MODE — Controls Locked",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

        if self.tracking_active:
            label = "TARGET LOST" if t_lost else "TRACKING"
            color = (0,0,255) if t_lost else (0,255,0)
            cv2.putText(img, f"🎯 {label}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

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
                cv2.imshow("Optical Flow Correct Once", vis)

            key      = cv2.waitKeyEx(1)
            if key == -1:
                rate.sleep(); continue

            char_key = key & 0xFF

            if   char_key in (ord('w'), ord('W')): self._move_manual(0, -1)
            elif char_key in (ord('s'), ord('S')): self._move_manual(0,  1)
            elif char_key in (ord('a'), ord('A')): self._move_manual(1,  0)
            elif char_key in (ord('d'), ord('D')): self._move_manual(-1, 0)

            elif char_key == 27:   # ESC
                if self.tracking_active: self._stop_tracking()
                break

            elif char_key in (ord('q'), ord('Q')):
                if self.tracking_active: self._stop_tracking()

            elif char_key == ord('r') and not self.is_busy and not self.tracking_active:
                self.roi = None
                self.roi_selected = self.template_saved = False
                self.last_target_px = self.last_error_px = None
                self._set_zoom(1.0)
                self.current_zoom = 1.0
                self.status = "Reset. Ready."

            elif char_key == ord('k') and not self.is_busy and not self.tracking_active:
                threading.Thread(target=self.auto_calibrate_lens, daemon=True).start()

            elif char_key == ord('c') and self.roi_selected \
                    and not self.is_busy and not self.tracking_active:
                self.template_saved = True
                self.status = "Target locked. Press 'z' to zoom+correct."

            elif char_key == ord('z'):
                print(f"[z] template_saved={self.template_saved} is_busy={self.is_busy} tracking_active={self.tracking_active}", flush=True)
                if self.template_saved and not self.is_busy and not self.tracking_active:
                    self.is_busy             = True
                    self.current_target_zoom = self.optimal_zoom
                    self.last_target_px      = None
                    self.last_error_px       = None
                    print(f"[z] Launching zoom x{self.optimal_zoom:.1f}", flush=True)
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
        OpticalFlowCorrectOnce()
    except rospy.ROSInterruptException:
        pass
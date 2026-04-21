#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optical_flow_benchmark.py
=========================
PTZ zoom correction benchmark — Optical Flow (Lucas-Kanade) only.
Sequential testing with camera pose save/restore between repeats,
error boxplots in degrees and pixels, execution time, target accuracy
scatter plots, and continuous video recording with overlay.
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
import tf2_ros
import tf2_geometry_msgs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point
from cv_bridge import CvBridge
from pan_tilt_msgs.msg import PanTiltCmdDeg, PanTiltStatus
from video_stream.srv import SetZoomLevel
import rospkg

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720

CURSORS = {
    1: (640, 360),
    2: (643, 415),
    3: (643, 387),
}
CURSOR_LABELS = {1: "Image center", 2: "Camera center", 3: "Extra point"}
CURSOR_COLORS = {1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 0, 255)}

ZOOM_FOVS_REF = {
    1.0: (63.7, 35.84), 2.0: (56.9, 31.2), 3.0: (50.7, 27.3), 4.0: (45.9, 24.5),
    5.0: (40.5, 21.6), 6.0: (37.4, 19.6), 7.0: (32.2, 17.2), 8.0: (29.1, 15.2),
    9.0: (25.3, 13.0), 10.0: (21.7, 11.1), 11.0: (18.3, 9.3), 12.0: (15.2, 7.7),
    13.0: (10.0, 6.2), 14.0: (7.8, 4.8), 15.0: (6.2, 3.6), 16.0: (5.2, 2.9),
    17.0: (4.1, 2.3), 18.0: (3.5, 1.9), 19.0: (2.9, 1.7), 20.0: (2.3, 1.3)
}

LOWE_RATIO = 0.75
MIN_INLIERS = 6
STABILISE_S = 1.5
EPICENTER_SIZE = 450

MICRO_CORRECT_PX = 15
INTER_METHOD_WAIT = 6.0
PRE_CENTRE_PX = 40
N_REPEATS = 20

METHOD_NAMES = ["OpticalFlow"]

# ── ORB Feature Matching (kept only for auto-calibration utility) ────────────

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

# ── Main Class ────────────────────────────────────────────────────────────────

class OpticalFlowBenchmark:

    def __init__(self):
        rospy.init_node('optical_flow_benchmark', anonymous=True)
        self.bridge = CvBridge()
        self.image = None
        self.image_lock = threading.Lock()

        self.TF_BASE = "base_link"
        self.TF_CAMERA = "camera_visor"
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.PAN_MIN_DEG, self.PAN_MAX_DEG = -60, 60
        self.TILT_MIN_DEG, self.TILT_MAX_DEG = -60, 60

        self.current_yaw = 0.0
        self.current_pitch = 0.0

        rospy.sleep(1.0)

        self.drawing = False
        self.start_x = self.start_y = -1
        self.current_x = self.current_y = -1
        self.roi = None
        self.roi_selected = False
        self.template_gray = None
        self.template_saved = False
        self.optimal_zoom = 1.0
        self.current_zoom = 1.0
        self.current_target_zoom = 1.0

        self.cursor_sel = 1
        self.is_busy = False

        self.last_target_px = None
        self.last_error_px = None
        self.last_inliers = 0
        self.current_method = ""

        self.benchmark_results = {m: [] for m in METHOD_NAMES}

        # Video recording
        self.video_writer = None
        self.video_lock = threading.Lock()

        self.status = "Draw a ROI around the target, then press 'c'"

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("pan_tilt_description")
        self.image_dir = os.path.join(pkg_path, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.pub_cmd = rospy.Publisher('/pan_tilt_cmd_deg', PanTiltCmdDeg, queue_size=10)
        rospy.Subscriber('/datavideo/video', Image, self._img_cb)
        rospy.Subscriber('/pan_tilt_status', PanTiltStatus, self._status_cb)

        rospy.loginfo("Connecting to /set_zoom service...")
        try:
            rospy.wait_for_service('/set_zoom', timeout=3.0)
            self.zoom_srv = rospy.ServiceProxy('/set_zoom', SetZoomLevel)
        except rospy.ROSException:
            rospy.logwarn("/set_zoom service not found.")
            self.zoom_srv = None

        cv2.namedWindow("OpticalFlow Benchmark")
        cv2.setMouseCallback("OpticalFlow Benchmark", self._mouse_cb)

        print("\n🟢 optical_flow_benchmark started (Lucas-Kanade only)")
        print("   Draw ROI → 'c' save template → 'z' benchmark")
        print("   W/A/S/D → Move camera manually (1°)")
        print("   k → Auto-calibrate lens")
        print("   1/2/3 cursor   0 all   r reset a x1   ESC quit\n")

        self._main_loop()

    # ==========================================
    # UNIVERSAL MATHEMATICAL MODELS
    # ==========================================
    def _get_fovs(self, target_z):
        """Power-weighted interpolation equation to compute real FOV in degrees"""
        z = max(1.0, min(20.0, float(target_z)))
        t = (z - 1.0) / 19.0
        k = 1.78

        fov_w_h, fov_t_h = 63.7, 2.3
        fov_w_v, fov_t_v = 35.84, 1.3

        fov_h = fov_w_h * ((1.0 - t)**k) + fov_t_h * t
        fov_v = fov_w_v * ((1.0 - t)**k) + fov_t_v * t

        return fov_h, fov_v

    def _get_zoom_factors(self, target_z):
        """Analytically computes the growth rate (scale)"""
        fov_h_1, fov_v_1 = self._get_fovs(1.0)
        fov_h_z, fov_v_z = self._get_fovs(target_z)

        k_scale = 1

        fx = (math.tan(math.radians(fov_h_1 / 2.0)) / math.tan(math.radians(fov_h_z / 2.0))) ** k_scale
        fy = (math.tan(math.radians(fov_v_1 / 2.0)) / math.tan(math.radians(fov_v_z / 2.0))) ** k_scale

        return fx, fy

    def _px_to_deg(self, ex, ey, zoom_level):
        """Convert pixel error to degrees using FOV at the given zoom level."""
        fov_h, fov_v = self._get_fovs(zoom_level)
        deg_x = ex * fov_h / IMG_W
        deg_y = ey * fov_v / IMG_H
        return deg_x, deg_y
    # ==========================================

    def auto_calibrate_lens(self):
        """Chained autonomous routine to deduce the constant 'k' of curvature.
        Uses ORB purely as a calibration aid; not part of the tracking benchmark."""
        if self.is_busy: return
        self.is_busy = True

        try:
            self.status = "Starting chained auto-calibration..."
            print(f"\n{self.status}")

            fov_w = 63.7
            fov_t = 2.3
            z_max = 20.0

            zoom_steps = [1.0, 3.0, 5.0, 8.0, 11.0]
            k_values = []

            self._set_zoom(zoom_steps[0])
            time.sleep(3.0)

            frame_prev = self._get_frame()
            if frame_prev is None:
                print("❌ Error: No video frame available.")
                return

            gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
            cx_prev, cy_prev = IMG_W / 2.0, IMG_H / 2.0

            cumulative_scale = 1.0

            for i in range(1, len(zoom_steps)):
                z_target = zoom_steps[i]
                z_prev = zoom_steps[i-1]

                crop_size = 400
                template, lcx, lcy = self._get_roi_crop(gray_prev, cx_prev, cy_prev, crop_size, crop_size)

                self.status = f"Applying zoom x{z_target}..."
                print(self.status)
                self._set_zoom(z_target)
                time.sleep(2.5)

                frame_curr = self._get_frame()
                gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

                self.status = f"Computing deformation (x{z_prev} -> x{z_target})..."
                res = find_template_point(template, gray_curr, lcx, lcy)

                if res is None:
                    print(f"⚠️ ORB lost reference at x{z_target}. Breaking chain here.")
                    break

                cx_found, cy_found, _, H, _, _, _, _ = res

                pts_orig = np.float32([ [lcx - 50, lcy], [lcx + 50, lcy] ]).reshape(-1, 1, 2)
                pts_proy = cv2.perspectiveTransform(pts_orig, H)

                dist_orig = 100.0
                dist_new = np.linalg.norm(pts_proy[0, 0] - pts_proy[1, 0])
                relative_scale = dist_new / dist_orig

                cumulative_scale *= relative_scale

                tan_w = math.tan(math.radians(fov_w / 2.0))
                fov_real = 2.0 * math.degrees(math.atan(tan_w / cumulative_scale))

                t = (z_target - 1.0) / (z_max - 1.0)
                numerator = (fov_real - fov_t * t) / fov_w

                if numerator > 0 and (1.0 - t) > 0:
                    k_step = math.log(numerator) / math.log(1.0 - t)
                    k_values.append(k_step)
                    print(f"  -> Segment completed: S_total={cumulative_scale:.2f}x | FOV={fov_real:.1f}° | k={k_step:.3f}")

                gray_prev = gray_curr
                cx_prev = cx_found
                cy_prev = cy_found

            if len(k_values) > 0:
                k_new = max(k_values)
                self.status = f"✅ Calibration done! k_damping = {k_new:.3f}"
                print(f"\n{self.status}")
                print(f"Based on {len(k_values)} chained measurements.")
            else:
                self.status = "❌ Failed. Point at a wall with textures/posters."
                print(self.status)

        except Exception as e:
            print(f"Calibration error: {e}")
        finally:
            self.status = "Returning to x1.0..."
            self._set_zoom(1.0)
            self.is_busy = False
            self.status = "Manual mode ready."

    # ── Optical Flow tracker ──────────────────────────────────────────────────

    def _find_point_optical_flow(self, prev_gray, curr_gray, target_x, target_y, roi_x1, roi_y1, roi_w, roi_h, zoom_prev, zoom_curr, guess_dx=0.0, guess_dy=0.0):
        """Lucas-Kanade tracker with affine warp and mechanical jump prediction."""
        fov_h1, _ = self._get_fovs(zoom_prev)
        fov_h2, _ = self._get_fovs(zoom_curr)
        scale = fov_h1 / fov_h2

        if abs(scale - 1.0) > 0.02:
            cx_cam, cy_cam = IMG_W / 2.0, IMG_H / 2.0
            M = cv2.getRotationMatrix2D((cx_cam, cy_cam), 0, scale)

            prev_gray = cv2.warpAffine(prev_gray, M, (IMG_W, IMG_H))

            pt = np.float32([[[target_x, target_y]]])
            target_x, target_y = cv2.transform(pt, M)[0][0]

            roi_c = np.float32([[[roi_x1 + roi_w/2.0, roi_y1 + roi_h/2.0]]])
            sim_roi_c = cv2.transform(roi_c, M)[0][0]

            roi_w, roi_h = int(roi_w * scale), int(roi_h * scale)
            roi_x1 = int(sim_roi_c[0] - roi_w/2.0)
            roi_y1 = int(sim_roi_c[1] - roi_h/2.0)

        mask = np.zeros_like(prev_gray)
        rx1, ry1 = max(0, roi_x1), max(0, roi_y1)
        rx2, ry2 = min(IMG_W, roi_x1 + roi_w), min(IMG_H, roi_y1 + roi_h)

        if rx2 <= rx1 or ry2 <= ry1: return None
        mask[ry1:ry2, rx1:rx2] = 255

        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.02, minDistance=5, mask=mask)
        if p0 is None or len(p0) < MIN_INLIERS: return None

        lk_params = dict(winSize=(45, 45), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        if abs(guess_dx) > 0.1 or abs(guess_dy) > 0.1:
            p1_guess = p0.copy()
            p1_guess[:, 0, 0] += guess_dx
            p1_guess[:, 0, 1] += guess_dy
            lk_params['flags'] = cv2.OPTFLOW_USE_INITIAL_FLOW
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, p1_guess, **lk_params)
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

        if p1 is None: return None

        good_new, good_old = p1[st == 1], p0[st == 1]
        if len(good_new) < MIN_INLIERS: return None

        src_pts, dst_pts = good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2)
        H, mask_hom = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None or mask_hom is None or int(mask_hom.sum()) < MIN_INLIERS: return None

        pt_final = np.float32([[target_x, target_y]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pt_final, H)

        return float(projected[0,0,0]), float(projected[0,0,1]), int(mask_hom.sum()), H, mask_hom, src_pts, dst_pts, None

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.image = frame
        except Exception as e:
            rospy.logwarn(f"Image error: {e}")

    def _status_cb(self, msg):
        self.current_yaw = msg.yaw_now
        self.current_pitch = msg.pitch_now

    def _get_frame(self):
        with self.image_lock:
            return self.image.copy() if self.image is not None else None

    # ── Video recording ───────────────────────────────────────────────────────

    def _start_video(self, path):
        with self.video_lock:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(path, fourcc, 15.0, (IMG_W, IMG_H))
            print(f"  🎥 Recording started: {path}")

    def _write_video_frame(self, frame):
        with self.video_lock:
            if self.video_writer is not None:
                self.video_writer.write(frame)

    def _stop_video(self):
        with self.video_lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                print("  🎥 Recording stopped")

    # ── Mouse / ROI ───────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        if self.is_busy: return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_x, self.current_y = x, y
            self.roi_selected = False
            self.template_saved = False
            self.template_gray = None
            self.last_target_px = self.last_error_px = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x, self.current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x0, y0 = min(self.start_x, x), min(self.start_y, y)
            w, h = abs(x - self.start_x), abs(y - self.start_y)
            if w > 10 and h > 10:
                self.roi = (x0, y0, w, h)
                self.roi_selected = True
                self.optimal_zoom = self._best_zoom(w, h)
                self.status = f"ROI {w}×{h}px → zoom recom. x{self.optimal_zoom:.1f}. Press 'c'"

    def _best_zoom(self, w, h):
        for z_int in range(200, 9, -1):
            z = round(z_int / 10.0, 1)
            fx, fy = self._get_zoom_factors(z)
            if w * fx <= IMG_W * 0.95 and h * fy <= IMG_H * 0.95:
                return z
        return 1.0

    # ── Zoom / PTZ commands ───────────────────────────────────────────────────

    def _set_zoom(self, level):
        if self.zoom_srv:
            try: self.zoom_srv(float(level))
            except rospy.ServiceException as e: print(f"❌ Error sending zoom: {e}")
        else: print(f"⚠️ Zoom simulation: x{float(level):.1f}")

    def _send_cmd(self, yaw, pitch, speed=20):
        cmd = PanTiltCmdDeg()
        cmd.yaw, cmd.pitch, cmd.speed = yaw, pitch, speed
        self.pub_cmd.publish(cmd)

    def _calculate_and_send_direct_cmd(self, ex, ey, zoom_level):
        """Direct incremental command (used by OpticalFlow)."""
        fov_h, fov_v = self._get_fovs(zoom_level)

        px_per_deg_x = IMG_W / fov_h
        px_per_deg_y = IMG_H / fov_v

        ang_x = -ex / px_per_deg_x
        ang_y = ey / px_per_deg_y

        error_magnitude = math.sqrt(ang_x**2 + ang_y**2)
        speed_val = 25 if error_magnitude > 5.0 else 15

        new_yaw = self.current_yaw + ang_x
        new_pitch = self.current_pitch + ang_y

        new_yaw = max(self.PAN_MIN_DEG, min(self.PAN_MAX_DEG, new_yaw))
        new_pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, new_pitch))

        new_yaw = round(new_yaw)
        new_pitch = round(new_pitch)

        self._send_cmd(new_yaw, new_pitch, speed=speed_val)
        return new_yaw, new_pitch

    # ── ROI helpers ───────────────────────────────────────────────────────────

    def _update_roi_visual(self, cx, cy, zoom_level, orig_w, orig_h):
        fx, fy = self._get_zoom_factors(zoom_level)
        new_w, new_h = orig_w * fx, orig_h * fy
        new_x, new_y = cx - new_w / 2.0, cy - new_h / 2.0
        self.roi = (int(new_x), int(new_y), int(new_w), int(new_h))

    def _get_roi_crop(self, image_gray, cx, cy, crop_w, crop_h):
        x1 = max(0, int(cx - crop_w / 2.0))
        y1 = max(0, int(cy - crop_h / 2.0))
        x2 = min(IMG_W, int(cx + crop_w / 2.0))
        y2 = min(IMG_H, int(cy + crop_h / 2.0))
        crop_gray = image_gray[y1:y2, x1:x2]
        local_cx = cx - x1
        local_cy = cy - y1
        return crop_gray, local_cx, local_cy

    def _get_roi_bbox(self, cx, cy, crop_w, crop_h):
        x1 = max(0, int(cx - crop_w / 2.0))
        y1 = max(0, int(cy - crop_h / 2.0))
        x2 = min(IMG_W, int(cx + crop_w / 2.0))
        y2 = min(IMG_H, int(cy + crop_h / 2.0))
        return x1, y1, x2 - x1, y2 - y1

    def _move_manual(self, dyaw, dpitch):
        if self.is_busy: return
        target_yaw = round(self.current_yaw) + dyaw
        target_pitch = round(self.current_pitch) + dpitch
        target_yaw = max(self.PAN_MIN_DEG, min(self.PAN_MAX_DEG, target_yaw))
        target_pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, target_pitch))
        self.status = f"🕹️ Manual Movement: yaw={target_yaw}° pitch={target_pitch}°"
        print(self.status)
        self._send_cmd(target_yaw, target_pitch, speed=15)

    # ── Pose save/restore ─────────────────────────────────────────────────────

    def _save_pose(self):
        return (self.current_yaw, self.current_pitch)

    def _restore_pose(self, pose):
        yaw, pitch = pose
        self._send_cmd(round(yaw), round(pitch), speed=15)
        self._set_zoom(1.0)
        time.sleep(INTER_METHOD_WAIT)

    # ── Optical Flow correction pipeline ──────────────────────────────────────

    def _run_optflow(self, target_zoom, cx_cursor, cy_cursor, x, y, orig_w, orig_h):
        """Run Optical Flow correction pipeline. Returns list of result dicts."""
        results = []

        self._set_zoom(1.0)
        time.sleep(0.5)
        frame_orig = self._get_frame()
        if frame_orig is None: return results

        initial_cx, initial_cy = x + orig_w / 2.0, y + orig_h / 2.0
        crop_w_init, crop_h_init = max(int(orig_w * 1.5), 250), max(int(orig_h * 1.5), 250)
        frame_orig_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(initial_cx, initial_cy, crop_w_init, crop_h_init)

        ex_init, ey_init = initial_cx - cx_cursor, initial_cy - cy_cursor

        if abs(ex_init) > PRE_CENTRE_PX or abs(ey_init) > PRE_CENTRE_PX:
            self.status = "[OptFlow] Pre-centering target..."
            self._calculate_and_send_direct_cmd(ex_init, ey_init, 1.0)
            time.sleep(2.0)

            frame1 = self._get_frame()
            scene_gray_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            res0 = self._find_point_optical_flow(
                frame_orig_gray, scene_gray_1, initial_cx, initial_cy,
                roi_x1, roi_y1, roi_w, roi_h, 1.0, 1.0,
                guess_dx=-ex_init, guess_dy=-ey_init
            )
            if res0 is None:
                self.status = "❌ [OptFlow] Failed to re-acquire after pre-centering"
                return results
            current_cx, current_cy = res0[0], res0[1]
            frame_base = frame1
        else:
            current_cx, current_cy = initial_cx, initial_cy
            frame_base = frame_orig

        self._update_roi_visual(current_cx, current_cy, 1.0, orig_w, orig_h)

        prev_gray = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)
        roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)
        zoom_prev = 1.0

        MAX_SAFE_JUMP = 2.0
        total_distance = target_zoom - 1.0
        zoom_steps = []
        if total_distance > 0:
            num_steps = math.ceil(total_distance / MAX_SAFE_JUMP)
            dynamic_step = total_distance / num_steps
            curr_z = 1.0
            for _ in range(num_steps):
                curr_z += dynamic_step
                zoom_steps.append(round(curr_z, 1))
            if zoom_steps: zoom_steps[-1] = float(round(target_zoom, 1))

        H_final, n_inliers = None, 0
        scene_gray_final = prev_gray

        for step_zoom in zoom_steps:
            self.status = f"[OptFlow] Applying zoom x{step_zoom:.1f}..."
            self._set_zoom(step_zoom)
            time.sleep(STABILISE_S)

            frame_z = self._get_frame()
            if frame_z is None: return results
            scene_gray_before = cv2.cvtColor(frame_z, cv2.COLOR_BGR2GRAY)

            self.status = f"[OptFlow] Tracking at x{step_zoom:.1f}..."

            t_start = time.time()
            result = self._find_point_optical_flow(
                prev_gray, scene_gray_before, current_cx, current_cy,
                roi_x1, roi_y1, roi_w, roi_h, zoom_prev, step_zoom
            )
            match_duration = time.time() - t_start

            if result is None:
                self.status = f"❌ [OptFlow] Failed at x{step_zoom:.1f}"
                return results

            cx_found, cy_found, n_inliers, H_final, mask_h, kp1, kp2, good = result
            ex_step, ey_step = cx_found - cx_cursor, cy_found - cy_cursor
            err_px = math.sqrt(ex_step**2 + ey_step**2)
            deg_x, deg_y = self._px_to_deg(ex_step, ey_step, step_zoom)
            err_deg = math.sqrt(deg_x**2 + deg_y**2)

            results.append({
                "zoom": step_zoom, "label": f"x{step_zoom:.1f}",
                "ex": ex_step, "ey": ey_step, "err_px": err_px,
                "ex_deg": deg_x, "ey_deg": deg_y, "err_deg": err_deg,
                "duration_s": match_duration
            })

            if abs(ex_step) > MICRO_CORRECT_PX or abs(ey_step) > MICRO_CORRECT_PX:
                self.status = f"[OptFlow] Micro-centering at x{step_zoom:.1f}..."
                self._calculate_and_send_direct_cmd(ex_step, ey_step, step_zoom)
                time.sleep(2.0)

                frame_z_after = self._get_frame()
                scene_gray_after = cv2.cvtColor(frame_z_after, cv2.COLOR_BGR2GRAY)

                roi_x1_mc, roi_y1_mc, roi_w_mc, roi_h_mc = self._get_roi_bbox(cx_found, cy_found, EPICENTER_SIZE, EPICENTER_SIZE)

                res_centered = self._find_point_optical_flow(
                    scene_gray_before, scene_gray_after,
                    cx_found, cy_found,
                    roi_x1_mc, roi_y1_mc, roi_w_mc, roi_h_mc,
                    step_zoom, step_zoom,
                    guess_dx=-ex_step, guess_dy=-ey_step
                )

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
                roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)

        self.last_target_px = (current_cx, current_cy)
        self.last_inliers = n_inliers

        ex_final, ey_final = current_cx - cx_cursor, current_cy - cy_cursor
        self.last_error_px = (ex_final, ey_final)

        yaw_final, pitch_final = self._calculate_and_send_direct_cmd(ex_final, ey_final, target_zoom)

        # ── Re-measure after final correction ──
        time.sleep(2.0)
        frame_post = self._get_frame()
        if frame_post is not None:
            scene_post = cv2.cvtColor(frame_post, cv2.COLOR_BGR2GRAY)
            roi_x1_p, roi_y1_p, roi_w_p, roi_h_p = self._get_roi_bbox(current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)
            res_post = self._find_point_optical_flow(
                scene_gray_final, scene_post, current_cx, current_cy,
                roi_x1_p, roi_y1_p, roi_w_p, roi_h_p,
                target_zoom, target_zoom,
                guess_dx=-ex_final, guess_dy=-ey_final
            )
            if res_post is not None:
                cx_post, cy_post, n_post = res_post[0], res_post[1], res_post[2]
                ex_post = cx_post - cx_cursor
                ey_post = cy_post - cy_cursor
                err_post = math.sqrt(ex_post**2 + ey_post**2)
                deg_x_post, deg_y_post = self._px_to_deg(ex_post, ey_post, target_zoom)
                err_deg_post = math.sqrt(deg_x_post**2 + deg_y_post**2)

                results.append({
                    "zoom": target_zoom, "label": f"x{target_zoom:.1f}_post",
                    "ex": ex_post, "ey": ey_post, "err_px": err_post,
                    "ex_deg": deg_x_post, "ey_deg": deg_y_post, "err_deg": err_deg_post,
                    "duration_s": 0.0, "is_post_correction": True
                })
                self.last_target_px = (cx_post, cy_post)
                self.last_error_px = (ex_post, ey_post)
                current_cx, current_cy = cx_post, cy_post

        self._update_roi_visual(cx_cursor, cy_cursor, target_zoom, orig_w, orig_h)

        self.status = f"✅ [OptFlow] yaw={yaw_final}° pitch={pitch_final}° in={n_inliers}"
        print(f"  {self.status}")

        return results

    # ── Benchmark orchestration ───────────────────────────────────────────────

    def _zoom_and_correct(self):
        try:
            target_zoom = self.current_target_zoom
            cx_cursor, cy_cursor = CURSORS[self.cursor_sel]
            x, y, orig_w, orig_h = self.roi

            # Safety limit
            max_safe_zoom = 1.0
            for z_int in range(10, 201):
                z = round(z_int / 10.0, 1)
                fx, fy = self._get_zoom_factors(z)
                if orig_w * fx > IMG_W * 0.95 or orig_h * fy > IMG_H * 0.95:
                    break
                max_safe_zoom = z
            if target_zoom > max_safe_zoom:
                print(f"⚠️ Optical limit: x{max_safe_zoom:.1f}")
                target_zoom = max_safe_zoom

            # Save initial pose
            initial_pose = self._save_pose()
            orig_roi = (x, y, orig_w, orig_h)

            # Setup output directory
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.image_dir, f"optflow_benchmark_{ts_str}")
            os.makedirs(out_dir, exist_ok=True)

            # Start video recording
            video_path = os.path.join(out_dir, "benchmark.avi")
            self._start_video(video_path)

            # Collect final-zoom errors across all repeats
            final_errors = {m: [] for m in METHOD_NAMES}
            all_step_results = {m: [] for m in METHOD_NAMES}

            # ── Run OpticalFlow N_REPEATS times ──
            for method in METHOD_NAMES:
                self.current_method = method

                print(f"\n{'='*60}")
                print(f"  {method}: {N_REPEATS} repeats at zoom x{target_zoom:.1f}")
                print(f"{'='*60}")

                for rep in range(1, N_REPEATS + 1):
                    self.last_target_px = None
                    self.last_error_px = None

                    # Restore pose (zoom out to x1.0 + return to initial yaw/pitch)
                    self._restore_pose(initial_pose)

                    # After the camera is back at zoom 1.0 and original pose,
                    # snap the overlay ROI back to the exact original rectangle
                    # (position AND dimensions) so every repeat starts identically.
                    self.roi = orig_roi
                    self.status = f"[{method}] Repeat {rep}/{N_REPEATS} — restoring pose..."
                    time.sleep(1.0)

                    print(f"\n  [{method}] Repeat {rep}/{N_REPEATS}")

                    results = self._run_optflow(target_zoom, cx_cursor, cy_cursor, x, y, orig_w, orig_h)

                    # Tag each result with the repeat number
                    for r in results:
                        r["repeat"] = rep

                    all_step_results[method].extend(results)

                    # Extract the post-correction error (last entry with is_post_correction)
                    # or fall back to the last entry
                    if results:
                        post = [r for r in results if r.get("is_post_correction")]
                        final = post[-1] if post else results[-1]
                        final_errors[method].append({
                            "repeat": rep,
                            "zoom": final["zoom"],
                            "ex_px": final["ex"], "ey_px": final["ey"], "err_px": final["err_px"],
                            "ex_deg": final["ex_deg"], "ey_deg": final["ey_deg"], "err_deg": final["err_deg"],
                            "duration_s": final["duration_s"],
                            "cx_cursor": cx_cursor, "cy_cursor": cy_cursor,
                            "cx_actual": cx_cursor + final["ex"],
                            "cy_actual": cy_cursor + final["ey"],
                        })
                        print(f"    Final error: {final['err_deg']:.4f}° ({final['err_px']:.1f}px)")
                    else:
                        final_errors[method].append({
                            "repeat": rep, "zoom": target_zoom,
                            "ex_px": None, "ey_px": None, "err_px": None,
                            "ex_deg": None, "ey_deg": None, "err_deg": None,
                            "duration_s": None,
                            "cx_cursor": cx_cursor, "cy_cursor": cy_cursor,
                            "cx_actual": None, "cy_actual": None,
                        })
                        print(f"    ⚠️ FAILED")

            # Stop video
            self._stop_video()

            self.current_method = ""
            self.status = f"✅ Benchmark complete — {N_REPEATS}x OpticalFlow"
            print(f"\n{self.status}\n")

            # Store for plotting
            self.benchmark_results = all_step_results
            self.final_errors = final_errors

            # Save results
            self._save_benchmark_results(orig_roi, initial_pose[0], initial_pose[1], out_dir, ts_str)

        except Exception as e:
            self.status = f"❌ Error: {e}"
            print(e)
            import traceback; traceback.print_exc()
        finally:
            self._stop_video()
            self.is_busy = False

    # ── Save results ──────────────────────────────────────────────────────────

    def _save_benchmark_results(self, roi, init_yaw, init_pitch, out_dir, ts):
        xr, yr, wr, hr = roi
        roi_cx, roi_cy = xr + wr/2.0, yr + hr/2.0

        # ── Full per-step CSV (all repeats, all zoom steps) ──
        csv_path = os.path.join(out_dir, "results_all_steps.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# Benchmark", ts, f"N_REPEATS={N_REPEATS}"])
            w.writerow(["# ROI", f"x={xr} y={yr} w={wr} h={hr}", f"cx={roi_cx:.1f}", f"cy={roi_cy:.1f}"])
            w.writerow(["# Initial pose", f"yaw={init_yaw:.3f}", f"pitch={init_pitch:.3f}"])
            w.writerow(["# Target zoom", f"x{self.current_target_zoom:.1f}"])
            w.writerow([])
            w.writerow(["method","repeat","zoom","label","ex_px","ey_px","err_px","ex_deg","ey_deg","err_deg","duration_s"])
            for mname, records in self.benchmark_results.items():
                if not records:
                    w.writerow([mname,"","","FAILED","","","","","","",""]); continue
                for r in records:
                    w.writerow([mname, r.get("repeat",""), r["zoom"], r["label"],
                                f"{r['ex']:.2f}", f"{r['ey']:.2f}", f"{r['err_px']:.2f}",
                                f"{r['ex_deg']:.4f}", f"{r['ey_deg']:.4f}", f"{r['err_deg']:.4f}",
                                f"{r['duration_s']:.4f}"])
        print(f"  CSV saved: {csv_path}")

        # ── Per-method final-zoom-error CSVs ──
        for mname in METHOD_NAMES:
            fname = f"{mname.lower()}_final_errors.csv"
            fpath = os.path.join(out_dir, fname)
            records = self.final_errors.get(mname, [])
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["repeat","zoom","cx_cursor","cy_cursor","cx_actual","cy_actual",
                            "ex_px","ey_px","err_px","ex_deg","ey_deg","err_deg","duration_s"])
                for r in records:
                    if r["err_px"] is not None:
                        w.writerow([r["repeat"], r["zoom"],
                                    f"{r['cx_cursor']:.1f}", f"{r['cy_cursor']:.1f}",
                                    f"{r['cx_actual']:.1f}", f"{r['cy_actual']:.1f}",
                                    f"{r['ex_px']:.2f}", f"{r['ey_px']:.2f}", f"{r['err_px']:.2f}",
                                    f"{r['ex_deg']:.4f}", f"{r['ey_deg']:.4f}", f"{r['err_deg']:.4f}",
                                    f"{r['duration_s']:.4f}"])
                    else:
                        w.writerow([r["repeat"], r["zoom"],
                                    f"{r['cx_cursor']:.1f}", f"{r['cy_cursor']:.1f}",
                                    "FAILED","","","","","","","",""])
            print(f"  CSV saved: {fpath}")

        # ── Prepare data from final errors (N_REPEATS points per method) ──
        labels = []
        data_err_deg = []; data_ex_deg = []; data_ey_deg = []
        data_err_px  = []; data_ex_px  = []; data_ey_px  = []
        mean_times   = []
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(METHOD_NAMES), 2)))[:len(METHOD_NAMES)]

        for mname in METHOD_NAMES:
            records = self.final_errors.get(mname, [])
            valid = [r for r in records if r["err_px"] is not None]
            labels.append(mname)
            data_err_deg.append([r["err_deg"] for r in valid])
            data_ex_deg.append([r["ex_deg"]   for r in valid])
            data_ey_deg.append([r["ey_deg"]   for r in valid])
            data_err_px.append([r["err_px"]   for r in valid])
            data_ex_px.append([r["ex_px"]     for r in valid])
            data_ey_px.append([r["ey_px"]     for r in valid])
            durations = [r["duration_s"] for r in valid if r["duration_s"] is not None]
            mean_times.append(np.mean(durations) if durations else 0.0)

        title = (f"Benchmark {ts}  ({N_REPEATS} repeats)\n"
                 f"ROI {wr}x{hr}px @ ({roi_cx:.0f},{roi_cy:.0f})  zoom x{self.current_target_zoom:.1f}")

        # ── Boxplots in DEGREES ──
        fig, axes = plt.subplots(1, 3, figsize=(16, 7))
        fig.suptitle(title + "\nFinal-zoom error [degrees]", fontsize=11, y=1.02)
        fig.subplots_adjust(top=0.85)
        for ax, data, ttl, ylabel in [
            (axes[0], data_err_deg, "Euclidean error", "err (°)"),
            (axes[1], data_ex_deg,  "X error (ex)",    "ex (°)"),
            (axes[2], data_ey_deg,  "Y error (ey)",    "ey (°)"),
        ]:
            for i, (d, col) in enumerate(zip(data, colors)):
                if not d: continue
                bp = ax.boxplot([d], positions=[i+1], patch_artist=True, notch=False,
                                medianprops=dict(color="black", linewidth=2), widths=0.6)
                bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.6)
            ax.set_xticks(range(1, len(labels)+1))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title(ttl); ax.set_ylabel(ylabel); ax.set_xlabel("Method")
            ax.grid(True, axis="y", alpha=0.3)
            if ttl != "Euclidean error":
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "comparison_degrees.png"), dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Plot saved: comparison_degrees.png")

        # ── Boxplots in PIXELS ──
        fig, axes = plt.subplots(1, 3, figsize=(16, 7))
        fig.suptitle(title + "\nFinal-zoom error [pixels]", fontsize=11, y=1.02)
        fig.subplots_adjust(top=0.85)
        for ax, data, ttl, ylabel in [
            (axes[0], data_err_px, "Euclidean error", "err (px)"),
            (axes[1], data_ex_px,  "X error (ex)",    "ex (px)"),
            (axes[2], data_ey_px,  "Y error (ey)",    "ey (px)"),
        ]:
            for i, (d, col) in enumerate(zip(data, colors)):
                if not d: continue
                bp = ax.boxplot([d], positions=[i+1], patch_artist=True, notch=False,
                                medianprops=dict(color="black", linewidth=2), widths=0.6)
                bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.6)
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
        plt.savefig(os.path.join(out_dir, "comparison_pixels.png"), dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Plot saved: comparison_pixels.png")

        # ── Execution time ──
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.suptitle(f"Execution time — {title}", fontsize=11, y=1.02)
        fig2.subplots_adjust(top=0.82)
        bars = ax2.bar(range(len(METHOD_NAMES)), mean_times, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(METHOD_NAMES)))
        ax2.set_xticklabels(METHOD_NAMES, rotation=30, ha="right")
        ax2.set_title("Mean match time (final zoom step)")
        ax2.set_ylabel("Time (s)"); ax2.set_xlabel("Method")
        ax2.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, mean_times):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                         f"{val:.3f}s", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "execution_time.png"), dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Time plot saved: execution_time.png")

        # ── Bullseye / target plot (px error scatter) ──
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.set_aspect('equal')
        ax3.set_title(f"Target accuracy plot ({N_REPEATS} repeats)\n"
                       f"zoom x{self.current_target_zoom:.1f}", fontsize=11)

        # Draw concentric rings
        max_radius = 0
        for mname in METHOD_NAMES:
            for r in self.final_errors.get(mname, []):
                if r["err_px"] is not None:
                    max_radius = max(max_radius, abs(r["ex_px"]), abs(r["ey_px"]))
        max_radius = max(max_radius * 1.3, 20)  # padding

        ring_count = 5
        for i in range(1, ring_count + 1):
            radius = max_radius * i / ring_count
            circle = plt.Circle((0, 0), radius, fill=False, color='gray',
                                linewidth=1.5 if i == ring_count else 0.8,
                                linestyle='-', alpha=0.5)
            ax3.add_patch(circle)
            ax3.text(radius + 1, 1, f"{radius:.0f}px", fontsize=7, color='gray', alpha=0.7)

        # Crosshairs
        ax3.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax3.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

        # Required position (center)
        ax3.plot(0, 0, 'o', color='red', markersize=12, zorder=10, label='Required position')

        # Plot actual positions per method
        method_colors = {'OpticalFlow': '#4CAF50'}
        method_markers = {'OpticalFlow': 'X'}

        for mname in METHOD_NAMES:
            records = self.final_errors.get(mname, [])
            valid = [r for r in records if r["err_px"] is not None]
            if not valid:
                continue
            xs = [r["ex_px"] for r in valid]
            ys = [r["ey_px"] for r in valid]
            color = method_colors.get(mname, 'green')
            marker = method_markers.get(mname, '+')
            ax3.scatter(xs, ys, c=color, marker=marker, s=120, linewidths=2,
                        zorder=5, label=f'{mname} actual', alpha=0.8)

        ax3.set_xlim(-max_radius, max_radius)
        ax3.set_ylim(-max_radius, max_radius)
        ax3.set_xlabel("X error (px)")
        ax3.set_ylabel("Y error (px)")
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "target_accuracy.png"), dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Plot saved: target_accuracy.png")

        # ── Bullseye / target plot (degree error scatter) ──
        fig4, ax4 = plt.subplots(figsize=(8, 8))
        ax4.set_aspect('equal')
        ax4.set_title(f"Target accuracy plot ({N_REPEATS} repeats)\n"
                       f"zoom x{self.current_target_zoom:.1f} [degrees]", fontsize=11)

        max_radius_deg = 0
        for mname in METHOD_NAMES:
            for r in self.final_errors.get(mname, []):
                if r["err_deg"] is not None:
                    max_radius_deg = max(max_radius_deg, abs(r["ex_deg"]), abs(r["ey_deg"]))
        max_radius_deg = max(max_radius_deg * 1.3, 0.5)

        for i in range(1, ring_count + 1):
            radius = max_radius_deg * i / ring_count
            circle = plt.Circle((0, 0), radius, fill=False, color='gray',
                                linewidth=1.5 if i == ring_count else 0.8,
                                linestyle='-', alpha=0.5)
            ax4.add_patch(circle)
            ax4.text(radius + 0.01, 0.01, f"{radius:.2f}°", fontsize=7, color='gray', alpha=0.7)

        ax4.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax4.axvline(0, color='gray', linewidth=0.5, alpha=0.5)

        ax4.plot(0, 0, 'o', color='red', markersize=12, zorder=10, label='Required position')

        for mname in METHOD_NAMES:
            records = self.final_errors.get(mname, [])
            valid = [r for r in records if r["err_deg"] is not None]
            if not valid:
                continue
            xs = [r["ex_deg"] for r in valid]
            ys = [r["ey_deg"] for r in valid]
            color = method_colors.get(mname, 'green')
            marker = method_markers.get(mname, '+')
            ax4.scatter(xs, ys, c=color, marker=marker, s=120, linewidths=2,
                        zorder=5, label=f'{mname} actual', alpha=0.8)

        ax4.set_xlim(-max_radius_deg, max_radius_deg)
        ax4.set_ylim(-max_radius_deg, max_radius_deg)
        ax4.set_xlabel("X error (°)")
        ax4.set_ylabel("Y error (°)")
        ax4.legend(loc='upper right', fontsize=9)
        ax4.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "target_accuracy_degrees.png"), dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Plot saved: target_accuracy_degrees.png")

        print(f"  Output dir: {out_dir}")

    # ── Overlay and main loop ─────────────────────────────────────────────────

    def _draw_overlay(self, img):
        if self.drawing:
            cv2.rectangle(img, (self.start_x, self.start_y), (self.current_x, self.current_y), (0, 255, 255), 1)

        if self.roi_selected and self.roi:
            x, y, w, h = self.roi
            color = (0, 255, 0) if self.template_saved else (0, 255, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            cx_roi, cy_roi = x + w // 2, y + h // 2
            cv2.drawMarker(img, (cx_roi, cy_roi), (0, 165, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, f"Target Zoom x{self.current_target_zoom:.1f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.last_target_px is not None:
            tx, ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.drawMarker(img, (tx, ty), (0,165,255), cv2.MARKER_CROSS, 24, 3)

        if self.last_target_px is not None and self.last_error_px is not None:
            cx_c, cy_c = CURSORS[self.cursor_sel]
            tx, ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.arrowedLine(img, (tx, ty), (cx_c, cy_c), (255,255,0), 2, tipLength=0.15)

        for k, (cx, cy) in CURSORS.items():
            if self.cursor_sel in (0, k):
                cv2.drawMarker(img, (cx, cy), CURSOR_COLORS[k], cv2.MARKER_CROSS, 20, 2)

        # Status line
        cv2.putText(img, self.status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        if self.is_busy:
            label = f"BENCHMARK [{self.current_method}]" if self.current_method else "AUTO MODE"
            cv2.putText(img, f"{label} - Controls Locked", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

    def _main_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            frame = self._get_frame()
            if frame is not None:
                vis = frame.copy()
                self._draw_overlay(vis)
                cv2.imshow("OpticalFlow Benchmark", vis)

                # Write to video if recording
                self._write_video_frame(vis)

            key = cv2.waitKeyEx(1)
            if key == -1:
                rate.sleep()
                continue

            char_key = key & 0xFF

            if char_key in (ord('w'), ord('W')): self._move_manual(0, -1)
            elif char_key in (ord('s'), ord('S')): self._move_manual(0, 1)
            elif char_key in (ord('a'), ord('A')): self._move_manual(1, 0)
            elif char_key in (ord('d'), ord('D')): self._move_manual(-1, 0)
            elif char_key == 27: break
            elif char_key == ord('r') and not self.is_busy:
                self.roi = None
                self.roi_selected = self.template_saved = False
                self.template_gray = None
                self.last_target_px = self.last_error_px = None
                self._set_zoom(1.0)
                self.current_zoom = 1.0
                self.status = "Reset. Ready."
            elif char_key == ord('k') and not self.is_busy:
                threading.Thread(target=self.auto_calibrate_lens, daemon=True).start()
            elif char_key == ord('c') and self.roi_selected and not self.is_busy:
                frame = self._get_frame()
                if frame is not None and self.roi is not None:
                    x, y, w, h = self.roi
                    self.template_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    self.template_saved = True
            elif char_key == ord('z') and self.template_saved and not self.is_busy:
                try:
                    val = input("✏️ Enter zoom (1.0-20.0) or press Enter for recommended: ")
                    tz = float(val) if val.strip() != "" else self.optimal_zoom
                    if 1.0 <= tz <= 20.0:
                        self.is_busy = True
                        self.current_target_zoom = tz
                        self.last_target_px = self.last_error_px = None
                        threading.Thread(target=self._zoom_and_correct, daemon=True).start()
                    else:
                        print("❌ Invalid zoom. Must be between 1.0 and 20.0.")
                except ValueError: pass
            elif char_key in (ord('0'), ord('1'), ord('2'), ord('3')):
                self.cursor_sel = int(chr(char_key))

            rate.sleep()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try: OpticalFlowBenchmark()
    except rospy.ROSInterruptException: pass
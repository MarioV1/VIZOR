#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optical_flow_benchmark.py
=========================
PTZ zoom correction ablation study — Equation vs Experimental Table.
Runs sequential testing (40 iterations total: 20 Eq + 20 Table),
generating side-by-side boxplots and comparative scatter plots.
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

# Tablas Experimentales (Ablation Study)
ZOOM_FOVS_TABLE = {
    1.0: (63.7, 35.84), 2.0: (56.9, 31.2), 3.0: (50.7, 27.3), 4.0: (45.9, 24.5),
    5.0: (40.5, 21.6), 6.0: (37.4, 19.6), 7.0: (32.2, 17.2), 8.0: (29.1, 15.2),
    9.0: (25.3, 13.0), 10.0: (21.7, 11.1), 11.0: (18.3, 9.3), 12.0: (15.2, 7.7),
    13.0: (10.0, 6.2), 14.0: (7.8, 4.8), 15.0: (6.2, 3.6), 16.0: (5.2, 2.9),
    17.0: (4.1, 2.3), 18.0: (3.5, 1.9), 19.0: (2.9, 1.7), 20.0: (2.3, 1.3)
}

ZOOM_FACTORES_TABLE = {
    1.0: (1.00, 1.00), 2.0: (1.12, 1.10), 3.0: (1.28, 1.25), 4.0: (1.39, 1.33),
    5.0: (1.61, 1.53), 6.0: (1.74, 1.67), 7.0: (1.98, 1.89), 8.0: (2.19, 2.10),
    9.0: (2.59, 2.43), 10.0: (3.09, 2.84), 11.0: (3.64, 3.25), 12.0: (4.37, 3.79),
    13.0: (6.29, 5.15), 14.0: (7.95, 6.48), 15.0: (9.89, 7.95), 16.0: (12.16, 9.35),
    17.0: (14.77, 10.86), 18.0: (16.82, 12.10), 19.0: (18.23, 12.85), 20.0: (18.56, 13.00),
}

LOWE_RATIO = 0.75
MIN_INLIERS = 6
STABILISE_S = 1.5
EPICENTER_SIZE = 450

MICRO_CORRECT_PX = 15
INTER_METHOD_WAIT = 6.0
PRE_CENTRE_PX = 40
N_REPEATS = 20  # 20 iteraciones por cada método (Total: 40)

# ABLATION STUDY METHODS
METHOD_NAMES = ["OptFlow_Ecuacion", "OptFlow_Tabla"]

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

        self.status = "Dibuja un ROI, y presiona 'c'"

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("pan_tilt_description")
        self.image_dir = os.path.join(pkg_path, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        self.pub_cmd = rospy.Publisher('/pan_tilt_cmd_deg', PanTiltCmdDeg, queue_size=10)
        rospy.Subscriber('/datavideo/video', Image, self._img_cb)
        rospy.Subscriber('/pan_tilt_status', PanTiltStatus, self._status_cb)

        rospy.loginfo("Conectando al servicio /set_zoom...")
        try:
            rospy.wait_for_service('/set_zoom', timeout=3.0)
            self.zoom_srv = rospy.ServiceProxy('/set_zoom', SetZoomLevel)
        except rospy.ROSException:
            rospy.logwarn("Servicio /set_zoom no encontrado.")
            self.zoom_srv = None

        cv2.namedWindow("Ablation Study")
        cv2.setMouseCallback("Ablation Study", self._mouse_cb)

        print("\n🟢 Ablation Study Iniciado (Ecuación Analítica vs Tabla Interpolada)")
        print(f"   Se realizarán 20 repeticiones por cada modelo (Total: 40 zooms).")
        print("   Dibuja ROI → 'c' guardar template → 'z' Iniciar Benchmark\n")

        self._main_loop()

    # ==========================================
    # SWITCHER MATEMÁTICO (Ecuación vs Tabla)
    # ==========================================
    def _get_interpolated_dict(self, target_z, data_dict):
        target_z = max(1.0, min(20.0, float(target_z)))
        z_lower = float(math.floor(target_z))
        z_upper = float(math.ceil(target_z))
        if z_lower == z_upper: return data_dict[z_lower]
        val_lower, val_upper = data_dict[z_lower], data_dict[z_upper]
        ratio = target_z - z_lower
        v1 = val_lower[0] + (val_upper[0] - val_lower[0]) * ratio
        v2 = val_lower[1] + (val_upper[1] - val_lower[1]) * ratio
        return (v1, v2)

    def _get_fovs(self, target_z):
        # ABLATION SWITCH
        if self.current_method == "OptFlow_Tabla":
            return self._get_interpolated_dict(target_z, ZOOM_FOVS_TABLE)
        else:
            # OptFlow_Ecuacion (Por defecto)
            z = max(1.0, min(20.0, float(target_z)))
            t = (z - 1.0) / 19.0
            k = 1.78
            fov_h = 63.7 * ((1.0 - t)**k) + 2.3 * t
            fov_v = 35.84 * ((1.0 - t)**k) + 1.3 * t
            return fov_h, fov_v

    def _get_zoom_factors(self, target_z):
        # ABLATION SWITCH
        if self.current_method == "OptFlow_Tabla":
            return self._get_interpolated_dict(target_z, ZOOM_FACTORES_TABLE)
        else:
            # OptFlow_Ecuacion (Ecuación Tangente exacta)
            fh1, fv1 = self._get_fovs(1.0)
            fhz, fvz = self._get_fovs(target_z)
            fx = math.tan(math.radians(fh1 / 2.0)) / math.tan(math.radians(fhz / 2.0))
            fy = math.tan(math.radians(fv1 / 2.0)) / math.tan(math.radians(fvz / 2.0))
            return fx, fy

    def _px_to_deg(self, ex, ey, zoom_level):
        fov_h, fov_v = self._get_fovs(zoom_level)
        deg_x = ex * fov_h / IMG_W
        deg_y = ey * fov_v / IMG_H
        return deg_x, deg_y
    # ==========================================

    # ── Optical Flow tracker ──────────────────────────────────────────────────

    def _find_point_optical_flow(self, prev_gray, curr_gray, target_x, target_y, roi_x1, roi_y1, roi_w, roi_h, zoom_prev, zoom_curr, guess_dx=0.0, guess_dy=0.0):
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
            pass

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

    def _write_video_frame(self, frame):
        with self.video_lock:
            if self.video_writer is not None:
                self.video_writer.write(frame)

    def _stop_video(self):
        with self.video_lock:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

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
                # Usamos la ecuación como base por defecto para el sugerido visual
                self.current_method = "OptFlow_Ecuacion"
                self.optimal_zoom = self._best_zoom(w, h)
                self.current_method = ""
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
            except rospy.ServiceException as e: pass

    def _send_cmd(self, yaw, pitch, speed=20):
        cmd = PanTiltCmdDeg()
        cmd.yaw, cmd.pitch, cmd.speed = yaw, pitch, speed
        self.pub_cmd.publish(cmd)

    def _calculate_and_send_direct_cmd(self, ex, ey, zoom_level):
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

    def _move_manual(self, dyaw, dpitch):
        if self.is_busy: return
        target_yaw = round(self.current_yaw) + dyaw
        target_pitch = round(self.current_pitch) + dpitch
        target_yaw = max(self.PAN_MIN_DEG, min(self.PAN_MAX_DEG, target_yaw))
        target_pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, target_pitch))
        self._send_cmd(target_yaw, target_pitch, speed=15)

    def _save_pose(self):
        return (self.current_yaw, self.current_pitch)

    def _restore_pose(self, pose):
        yaw, pitch = pose
        self._send_cmd(round(yaw), round(pitch), speed=15)
        self._set_zoom(1.0)
        time.sleep(INTER_METHOD_WAIT)

    # ── Optical Flow correction pipeline ──────────────────────────────────────

    def _run_optflow(self, target_zoom, cx_cursor, cy_cursor, x, y, orig_w, orig_h):
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
            self.status = f"[{self.current_method}] Pre-centering..."
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
                self.status = f"❌ [{self.current_method}] Pre-center FAIL"
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
            self.status = f"[{self.current_method}] Zoom x{step_zoom:.1f}..."
            self._set_zoom(step_zoom)
            time.sleep(STABILISE_S)

            frame_z = self._get_frame()
            if frame_z is None: return results
            scene_gray_before = cv2.cvtColor(frame_z, cv2.COLOR_BGR2GRAY)

            t_start = time.time()
            result = self._find_point_optical_flow(
                prev_gray, scene_gray_before, current_cx, current_cy,
                roi_x1, roi_y1, roi_w, roi_h, zoom_prev, step_zoom
            )
            match_duration = time.time() - t_start

            if result is None:
                self.status = f"❌ [{self.current_method}] Failed at x{step_zoom:.1f}"
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
                cx_post, cy_post = res_post[0], res_post[1]
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
        return results

    # ── Benchmark orchestration ───────────────────────────────────────────────

    def _zoom_and_correct(self):
        try:
            target_zoom = self.current_target_zoom
            cx_cursor, cy_cursor = CURSORS[self.cursor_sel]
            x, y, orig_w, orig_h = self.roi

            # Evaluamos límite con la ecuación
            self.current_method = "OptFlow_Ecuacion"
            max_safe_zoom = 1.0
            for z_int in range(10, 201):
                z = round(z_int / 10.0, 1)
                fx, fy = self._get_zoom_factors(z)
                if orig_w * fx > IMG_W * 0.95 or orig_h * fy > IMG_H * 0.95:
                    break
                max_safe_zoom = z
            if target_zoom > max_safe_zoom:
                target_zoom = max_safe_zoom

            initial_pose = self._save_pose()
            orig_roi = (x, y, orig_w, orig_h)

            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.image_dir, f"ablation_study_{ts_str}")
            os.makedirs(out_dir, exist_ok=True)

            video_path = os.path.join(out_dir, "ablation_benchmark.avi")
            self._start_video(video_path)

            final_errors = {m: [] for m in METHOD_NAMES}
            all_step_results = {m: [] for m in METHOD_NAMES}

            # ── Ejecutar ABLATION STUDY: 20 Ecuacion, y luego 20 Tabla ──
            for method in METHOD_NAMES:
                self.current_method = method # ¡AQUÍ CAMBIA EL MODELO MATEMÁTICO!

                print(f"\n{'='*60}")
                print(f"  {method}: {N_REPEATS} repeats at zoom x{target_zoom:.1f}")
                print(f"{'='*60}")

                for rep in range(1, N_REPEATS + 1):
                    self.last_target_px = None
                    self.last_error_px = None

                    self._restore_pose(initial_pose)
                    self.roi = orig_roi
                    
                    self.status = f"[{method}] {rep}/{N_REPEATS} — Restoring pose..."
                    time.sleep(1.0)
                    print(f"\n  [{method}] Repeat {rep}/{N_REPEATS}")

                    results = self._run_optflow(target_zoom, cx_cursor, cy_cursor, x, y, orig_w, orig_h)

                    for r in results: r["repeat"] = rep
                    all_step_results[method].extend(results)

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
                        print(f"    Final err: {final['err_deg']:.4f}° ({final['err_px']:.1f}px)")
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

            self._stop_video()
            self.current_method = ""
            self.status = f"✅ Ablation Study completado — {N_REPEATS*2}x iteraciones."
            print(f"\n{self.status}\n")

            self.benchmark_results = all_step_results
            self.final_errors = final_errors
            self._save_benchmark_results(orig_roi, initial_pose[0], initial_pose[1], out_dir, ts_str)

        except Exception as e:
            self.status = f"❌ Error: {e}"
            print(e)
        finally:
            self._stop_video()
            self.is_busy = False

    # ── Save results and comparative plots ────────────────────────────────────

    def _save_benchmark_results(self, roi, init_yaw, init_pitch, out_dir, ts):
        xr, yr, wr, hr = roi
        roi_cx, roi_cy = xr + wr/2.0, yr + hr/2.0

        csv_path = os.path.join(out_dir, "results_all_steps.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# Ablation Study", ts, f"N_REPEATS={N_REPEATS} por método"])
            w.writerow(["method","repeat","zoom","label","ex_px","ey_px","err_px","ex_deg","ey_deg","err_deg","duration_s"])
            for mname, records in self.benchmark_results.items():
                if not records: continue
                for r in records:
                    w.writerow([mname, r.get("repeat",""), r["zoom"], r["label"],
                                f"{r['ex']:.2f}", f"{r['ey']:.2f}", f"{r['err_px']:.2f}",
                                f"{r['ex_deg']:.4f}", f"{r['ey_deg']:.4f}", f"{r['err_deg']:.4f}",
                                f"{r['duration_s']:.4f}"])

        # Preparar data
        labels = []
        data_err_deg = []; data_ex_deg = []; data_ey_deg = []
        data_err_px  = []; data_ex_px  = []; data_ey_px  = []
        
        # Color azul para ecuación, Naranja para tabla
        colors = ['#1f77b4', '#ff7f0e'] 

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

        title = f"Ablation Study (Ecuación vs Tabla)\n{N_REPEATS} repeats/method | zoom x{self.current_target_zoom:.1f}"

        # ── Boxplots COMPARATIVOS PIXELES ──
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
                bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.7)
            ax.set_xticks(range(1, len(labels)+1))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_title(ttl); ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "comparativa_ablacion_pixeles.png"), dpi=150, bbox_inches="tight"); plt.close()

        # ── Scatter COMPARATIVO DIANA ──
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        ax3.set_aspect('equal')
        ax3.set_title(title, fontsize=11)

        max_radius = max([max([abs(r["ex_px"]), abs(r["ey_px"])]) for m in METHOD_NAMES for r in self.final_errors.get(m, []) if r["err_px"] is not None] + [20]) * 1.3
        
        for i in range(1, 6):
            radius = max_radius * i / 5
            ax3.add_patch(plt.Circle((0, 0), radius, fill=False, color='gray', linestyle='-', alpha=0.5))

        ax3.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax3.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax3.plot(0, 0, 'o', color='red', markersize=12, zorder=10, label='Target (Centro)')

        method_colors = {'OptFlow_Ecuacion': '#1f77b4', 'OptFlow_Tabla': '#ff7f0e'}
        method_markers = {'OptFlow_Ecuacion': 'o', 'OptFlow_Tabla': 'X'}

        for mname in METHOD_NAMES:
            records = self.final_errors.get(mname, [])
            valid = [r for r in records if r["err_px"] is not None]
            if not valid: continue
            xs = [r["ex_px"] for r in valid]
            ys = [r["ey_px"] for r in valid]
            c = method_colors.get(mname)
            m = method_markers.get(mname)
            ax3.scatter(xs, ys, c=c, marker=m, s=90, linewidths=1.5, zorder=5, label=mname, alpha=0.8)

        ax3.set_xlim(-max_radius, max_radius)
        ax3.set_ylim(-max_radius, max_radius)
        ax3.set_xlabel("X error (px)")
        ax3.set_ylabel("Y error (px)")
        ax3.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "diana_ablacion_pixeles.png"), dpi=150, bbox_inches="tight"); plt.close()

    def _draw_overlay(self, img):
        if self.drawing:
            cv2.rectangle(img, (self.start_x, self.start_y), (self.current_x, self.current_y), (0, 255, 255), 1)

        if self.roi_selected and self.roi:
            x, y, w, h = self.roi
            color = (0, 255, 0) if self.template_saved else (0, 255, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.drawMarker(img, (x+w//2, y+h//2), (0, 165, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, f"Target x{self.current_target_zoom:.1f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.last_target_px is not None:
            tx, ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.drawMarker(img, (tx, ty), (0,165,255), cv2.MARKER_CROSS, 24, 3)

        for k, (cx, cy) in CURSORS.items():
            if self.cursor_sel in (0, k):
                cv2.drawMarker(img, (cx, cy), CURSOR_COLORS[k], cv2.MARKER_CROSS, 20, 2)

        cv2.putText(img, self.status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        if self.is_busy:
            label = f"ABLATION [{self.current_method}]" if self.current_method else "AUTO MODE"
            cv2.putText(img, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

    def _main_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            frame = self._get_frame()
            if frame is not None:
                vis = frame.copy()
                self._draw_overlay(vis)
                cv2.imshow("Ablation Study", vis)
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
                self.last_target_px = self.last_error_px = None
                self._set_zoom(1.0)
                self.current_method = ""
                self.status = "Reset. Ready."
            elif char_key == ord('c') and self.roi_selected and not self.is_busy:
                self.template_saved = True
            elif char_key == ord('z') and self.template_saved and not self.is_busy:
                try:
                    val = input("✏️ Enter zoom (1.0-20.0): ")
                    tz = float(val) if val.strip() != "" else self.optimal_zoom
                    if 1.0 <= tz <= 20.0:
                        self.is_busy = True
                        self.current_target_zoom = tz
                        self.last_target_px = self.last_error_px = None
                        threading.Thread(target=self._zoom_and_correct, daemon=True).start()
                except ValueError: pass
            elif char_key in (ord('0'), ord('1'), ord('2'), ord('3')):
                self.cursor_sel = int(chr(char_key))

            rate.sleep()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try: OpticalFlowBenchmark()
    except rospy.ROSInterruptException: pass
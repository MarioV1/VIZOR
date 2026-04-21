#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optflow_correct_once.py
=======================
Single-shot PTZ zoom correction using Lucas-Kanade optical flow.

Workflow:
  1. Draw a ROI around the target.
  2. Press 'c' to save the template (also computes recommended zoom).
  3. Press 'z' to immediately apply the recommended zoom and run ONE
     optical-flow correction pipeline. No prompt, no repeats.

Outputs (per run, under <pkg>/images/optflow_once_<ts>/):
  - results_steps.csv         : per-step errors of the single correction
  - summary_boxplot.png       : single-run summary in boxplot style
                                (one box per axis: err, ex, ey in deg & px)

Notes
-----
- No TF2 (segfaults on this system). Uses direct incremental yaw/pitch.
- No video recording.
- No benchmark loop / no repeats.
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
from cv_bridge import CvBridge
from pan_tilt_msgs.msg import PanTiltCmdDeg, PanTiltStatus
from video_stream.srv import SetZoomLevel
import rospkg

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 1280, 720

# Calibrated optical centre
CURSOR_CX, CURSOR_CY = 643, 388

LOWE_RATIO = 0.75
MIN_INLIERS = 6
STABILISE_S = 1.5
EPICENTER_SIZE = 450

MICRO_CORRECT_PX = 15
PRE_CENTRE_PX = 40
MAX_SAFE_JUMP = 2.0  # zoom step size for OptFlow path

# ============================================================
# FLOWCHART PHASE IDs — set by the script during execution
# ============================================================
PHASE_IDLE          = "idle"
PHASE_WAIT_CAMERA   = "wait_camera"
PHASE_ROI_CHECK     = "roi_check"
PHASE_PRESS_C       = "press_c"
PHASE_SAVE_TEMPLATE = "save_template"
PHASE_PRESS_Z       = "press_z"
PHASE_CHECK_40PX    = "check_40px"
PHASE_PRE_CENTRE    = "pre_centre"
PHASE_SET_ZOOM      = "set_zoom_steps"
PHASE_ZOOM_IN       = "zoom_in"
PHASE_SAVE_NEW_TPL  = "save_new_template"
PHASE_AFFINE_WARP   = "affine_warp"
PHASE_SHI_TOMASI    = "shi_tomasi"
PHASE_LK_FLOW       = "lk_flow"
PHASE_HOMOGRAPHY    = "homography"
PHASE_PROJECT       = "project"
PHASE_CHECK_15PX    = "check_15px"
PHASE_MICRO_CORRECT = "micro_correct"
PHASE_ACCUMULATE    = "accumulate"
PHASE_LAST_STEP     = "last_step"
PHASE_FINAL_CMD     = "final_cmd"
PHASE_SAVE_RESULTS  = "save_results"
PHASE_DONE          = "done"

# ============================================================
# FLOWCHART RENDERER — draws with OpenCV into a second window
# ============================================================
# Colours (BGR)
_BG        = (30, 30, 30)
_DEFAULT   = (200, 200, 200)
_ACTIVE    = (80, 200, 255)
_DONE      = (100, 200, 100)
_TXT       = (240, 240, 240)
_TXT_D     = (30, 30, 30)
_ARROW     = (160, 160, 160)
_DEC       = (200, 200, 200)
_DEC_ACT   = (100, 180, 255)
_DASH      = (150, 150, 150)
_DASH_ACT  = (130, 210, 255)
_BOLD      = (220, 220, 220)
_BOLD_ACT  = (60, 180, 255)

FC_W, FC_H = 720, 720


class FlowchartRenderer:
    """Renders the full OptFlow-correction flowchart with real-time highlights."""

    def __init__(self):
        self.history = set()
        self.current = PHASE_IDLE

        # (id, type, x, y, w, h, label, [label2])
        # Left column centred at cx=230, right column centred at cx=510
        self.nodes = [
            # --- Left column: main user flow ---
            ("idle",             "pill",    180, 10,  100, 32, "Start"),
            ("wait_camera",      "rect",    140, 62,  180, 34, "Wait for camera"),
            ("roi_check",        "diamond", 150, 114, 160, 52, "ROI selected?"),
            ("press_c",          "diamond", 160, 194, 140, 48, "Press 'c'"),
            ("save_template",    "dashed",  140, 268, 180, 34, "Save template"),
            ("press_z",          "diamond", 160, 326, 140, 48, "Press 'z'"),
            ("check_40px",       "diamond", 150, 402, 160, 52, "ex|ey > 40px?"),
            ("pre_centre",       "rect",      4, 412,  92, 34, "Pre-centre"),
            ("set_zoom_steps",   "dashed",  140, 478, 180, 34, "Set zoom steps"),
            ("zoom_in",          "bold",    125, 534, 210, 34, "Zoom in to next step"),
            ("save_new_template","dashed",  140, 590, 180, 34, "Save new template"),

            # --- Right column: OptFlow pipeline (bottom-up) cx=510 ---
            ("affine_warp",      "rect",    410, 590, 200, 34, "Affine pre-warp"),
            ("shi_tomasi",       "bold",    405, 534, 210, 40, "Shi-Tomasi corners", "in ROI mask"),
            ("lk_flow",          "bold",    405, 478, 210, 40, "Pyramidal LK", "winSize=45, lvl=5"),
            ("homography",       "bold",    410, 424, 200, 40, "Compute homography", "RANSAC"),
            ("project",          "bold",    410, 376, 200, 34, "Project target via H"),

            # --- Right column: decision loop (above pipeline, cx=510) ---
            ("check_15px",       "diamond", 435, 304, 150, 48, "ex|ey > 15px?"),
            ("micro_correct",    "rect",    630, 314,  60, 34, "Micro", "correct"),
            ("accumulate",       "dashed",  450, 242, 120, 34, "Accumulate err"),
            ("last_step",        "diamond", 445, 172, 130, 44, "Last step?"),

            # --- Top-right: final sequence (End at top) ---
            ("done",             "pill",    460, 10,  100, 32, "End"),
            ("save_results",     "dashed",  410, 56,  200, 34, "Save results + plots"),
            ("final_cmd",        "bold",    410, 110, 200, 34, "Send final correction"),
        ]

        # Arrows: list of (waypoints, label, label_pos)
        # Left column cx=230, right column cx=510
        # Diamond edges: top=(cx,y) bottom=(cx,y+h) left=(x,cy) right=(x+w,cy)
        self.arrows = [
            # --- Left column ---
            ([(230, 42), (230, 62)], None, None),
            ([(230, 96), (230, 114)], None, None),
            # ROI No -> loop back right
            ([(310, 140), (338, 140), (338, 79), (320, 79)], "No", (314, 136)),
            # ROI Yes -> Press c
            ([(230, 166), (230, 194)], "Yes", (236, 186)),
            # Press c Yes -> Save template
            ([(230, 242), (230, 268)], "Yes", (236, 260)),
            # Save template -> Press z
            ([(230, 302), (230, 326)], None, None),
            # Press z Yes -> check 40px
            ([(230, 374), (230, 402)], "Yes", (236, 396)),
            # check 40px Yes -> Pre-centre (left tip x=150)
            ([(150, 428), (96, 428)], "Yes", (104, 422)),
            # Pre-centre -> Set zoom steps
            ([(50, 446), (50, 495), (140, 495)], None, None),
            # check 40px No -> Set zoom (bottom cx=230)
            ([(230, 454), (230, 478)], "No", (236, 470)),
            # Set zoom -> Zoom in
            ([(230, 512), (230, 534)], None, None),
            # Zoom in -> Save new template
            ([(230, 568), (230, 590)], None, None),
            # Save new template -> Affine warp (horizontal entry into pipeline)
            ([(320, 607), (410, 607)], None, None),

            # --- Right column pipeline (upward) ---
            # Affine warp -> Shi-Tomasi
            ([(510, 590), (510, 574)], None, None),
            # Shi-Tomasi -> LK flow
            ([(510, 534), (510, 518)], None, None),
            # LK flow -> Homography
            ([(510, 478), (510, 464)], None, None),
            # Homography -> Project
            ([(510, 424), (510, 410)], None, None),
            # Project -> check 15px
            ([(510, 376), (510, 352)], None, None),

            # --- Right column decision loop ---
            # check 15px Yes -> Micro correct (right tip x+w=585, cy=328)
            ([(585, 328), (630, 328)], "Yes", (592, 322)),
            # Micro correct -> up to Last step (via right side, right tip at x+w=575, cy=194)
            ([(660, 314), (660, 194), (575, 194)], None, None),
            # check 15px No -> Accumulate (top cx=510, y=304)
            ([(510, 304), (510, 276)], "No", (516, 298)),
            # Accumulate -> Last step (top -> bottom of last_step cx=510, y+h=216)
            ([(510, 242), (510, 216)], None, None),
            # Last step No -> loops back to Zoom in (left tip x=445, cy=194)
            ([(445, 194), (380, 194), (380, 551), (335, 551)], "No", (386, 188)),
            # Last step Yes -> straight up to Send final correction (top of diamond cx=510, y=172)
            ([(510, 172), (510, 144)], "Yes", (516, 162)),

            # --- Top-right final sequence (upward) ---
            # Send final -> Save results
            ([(510, 110), (510, 90)], None, None),
            # Save results -> End
            ([(510, 56), (510, 42)], None, None),
        ]

    # --- Public API ---
    def set_phase(self, phase_id):
        if self.current != PHASE_IDLE and self.current != phase_id:
            self.history.add(self.current)
        self.current = phase_id

    def reset(self):
        self.history.clear()
        self.current = PHASE_IDLE

    def render(self):
        img = np.full((FC_H, FC_W, 3), _BG, dtype=np.uint8)

        # Arrows
        for wps, label, lpos in self.arrows:
            for i in range(len(wps) - 1):
                if i == len(wps) - 2:
                    cv2.arrowedLine(img, wps[i], wps[i + 1], _ARROW, 1, tipLength=0.06)
                else:
                    cv2.line(img, wps[i], wps[i + 1], _ARROW, 1)
            if label and lpos:
                cv2.putText(img, label, lpos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, _ARROW, 1, cv2.LINE_AA)

        # Nodes
        for nd in self.nodes:
            nid, ntype = nd[0], nd[1]
            nx, ny, nw, nh = nd[2], nd[3], nd[4], nd[5]
            l1 = nd[6]
            l2 = nd[7] if len(nd) > 7 else None
            st = "active" if nid == self.current else ("done" if nid in self.history else "default")
            self._draw_node(img, ntype, nx, ny, nw, nh, l1, l2, st)

        # Legend (top-left, vertical stack)
        lx, ly0 = 8, 8
        for i, (col, lbl) in enumerate([(_ACTIVE, "Active"), (_DONE, "Done"), (_DEFAULT, "Pending")]):
            ly = ly0 + i * 20
            cv2.rectangle(img, (lx, ly), (lx + 12, ly + 12), col, -1)
            cv2.putText(img, lbl, (lx + 18, ly + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _TXT, 1, cv2.LINE_AA)
        return img

    # --- Node drawing ---
    def _draw_node(self, img, ntype, x, y, w, h, label, label2, state):
        cx, cy = x + w // 2, y + h // 2

        if ntype == "pill":
            col = _ACTIVE if state == "active" else (_DONE if state == "done" else _DEFAULT)
            r = h // 2
            cv2.rectangle(img, (x + r, y), (x + w - r, y + h), col, -1)
            cv2.ellipse(img, (x + r, cy), (r, r), 0, 90, 270, col, -1)
            cv2.ellipse(img, (x + w - r, cy), (r, r), 0, -90, 90, col, -1)
            cv2.putText(img, label, (cx - len(label) * 4, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TXT_D, 1, cv2.LINE_AA)
            if state == "active":
                cv2.rectangle(img, (x + r, y), (x + w - r, y + h), (0, 140, 255), 2)
                cv2.ellipse(img, (x + r, cy), (r, r), 0, 90, 270, (0, 140, 255), 2)
                cv2.ellipse(img, (x + w - r, cy), (r, r), 0, -90, 90, (0, 140, 255), 2)

        elif ntype == "diamond":
            if state == "active":
                col, brd = _DEC_ACT, (0, 140, 255)
            elif state == "done":
                col, brd = _DONE, (60, 160, 60)
            else:
                col, brd = _DEC, (140, 140, 140)
            pts = np.array([[cx, y], [x + w, cy], [cx, y + h], [x, cy]], np.int32)
            cv2.fillPoly(img, [pts], col)
            cv2.polylines(img, [pts], True, brd, 2 if state == "active" else 1)
            fs = 0.35 if len(label) > 14 else 0.40
            tw = int(len(label) * (5 if fs < 0.4 else 6))
            cv2.putText(img, label, (cx - tw // 2, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, _TXT_D, 1, cv2.LINE_AA)

        elif ntype in ("rect", "dashed", "bold"):
            if state == "active":
                col = {"dashed": _DASH_ACT, "bold": _BOLD_ACT}.get(ntype, _ACTIVE)
                brd = (0, 140, 255) if ntype != "bold" else (0, 100, 255)
            elif state == "done":
                col, brd = _DONE, (60, 160, 60)
            else:
                col = (50, 50, 50) if ntype == "dashed" else (45, 45, 45)
                brd = {"dashed": _DASH, "bold": _BOLD}.get(ntype, _DEFAULT)

            cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
            thk = 2 if state == "active" else 1
            if ntype == "dashed" and state == "default":
                self._dashed_rect(img, x, y, w, h, brd)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), brd, thk)

            tc = _TXT_D if state in ("active", "done") else _TXT
            if label2:
                cv2.putText(img, label, (cx - len(label) * 4, cy - 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.38, tc, 1, cv2.LINE_AA)
                tc2 = (60, 60, 60) if state in ("active", "done") else (160, 160, 160)
                cv2.putText(img, label2, (cx - len(label2) * 3, cy + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, tc2, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, label, (cx - len(label) * 4, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, tc, 1, cv2.LINE_AA)

    def _dashed_rect(self, img, x, y, w, h, color, dash=8, gap=5):
        for (p1, p2) in [((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)),
                         ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))]:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            l = math.sqrt(dx * dx + dy * dy)
            if l == 0:
                continue
            ux, uy = dx / l, dy / l
            d = 0
            while d < l:
                s = (int(p1[0] + ux * d), int(p1[1] + uy * d))
                e = (int(p1[0] + ux * min(d + dash, l)), int(p1[1] + uy * min(d + dash, l)))
                cv2.line(img, s, e, color, 1)
                d += dash + gap


class OptFlowCorrectOnce:

    def __init__(self):
        rospy.init_node('optflow_correct_once', anonymous=True)
        self.bridge = CvBridge()
        self.image = None
        self.image_lock = threading.Lock()

        self.PAN_MIN_DEG, self.PAN_MAX_DEG = -60, 60
        self.TILT_MIN_DEG, self.TILT_MAX_DEG = -60, 60

        self.current_yaw = 0.0
        self.current_pitch = 0.0

        rospy.sleep(1.0)

        # ROI / template state
        self.drawing = False
        self.start_x = self.start_y = -1
        self.current_x = self.current_y = -1
        self.roi = None
        self.roi_selected = False
        self.template_gray = None
        self.template_saved = False
        self.optimal_zoom = 1.0
        self.current_target_zoom = 1.0

        self.is_busy = False

        self.last_target_px = None
        self.last_error_px = None
        self.last_inliers = 0

        # Flowchart
        self.flowchart = FlowchartRenderer()
        self.flowchart.set_phase(PHASE_WAIT_CAMERA)

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

        cv2.namedWindow("OptFlow Correct Once")
        cv2.setMouseCallback("OptFlow Correct Once", self._mouse_cb)
        cv2.namedWindow("Flowchart", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Flowchart", FC_W, FC_H)

        print("\n🟢 optflow_correct_once started")
        print("   Draw ROI → 'c' save template → 'z' apply recommended zoom + correct")
        print("   W/A/S/D → Move camera manually (1°)")
        print("   r reset   ESC quit\n")

        self._main_loop()

    # ── FOV / pixel-to-degree model ───────────────────────────────────────────
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
        fx = math.tan(math.radians(fov_h_1 / 2.0)) / math.tan(math.radians(fov_h_z / 2.0))
        fy = math.tan(math.radians(fov_v_1 / 2.0)) / math.tan(math.radians(fov_v_z / 2.0))
        return fx, fy

    def _px_to_deg(self, ex, ey, zoom_level):
        fov_h, fov_v = self._get_fovs(zoom_level)
        return ex * fov_h / IMG_W, ey * fov_v / IMG_H

    # ── Optical Flow tracker ──────────────────────────────────────────────────
    def _find_point_optical_flow(self, prev_gray, curr_gray, target_x, target_y,
                                  roi_x1, roi_y1, roi_w, roi_h,
                                  zoom_prev, zoom_curr, guess_dx=0.0, guess_dy=0.0):
        """Lucas-Kanade tracker with affine pre-warp for FOV change."""
        self.flowchart.set_phase(PHASE_AFFINE_WARP)
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

        self.flowchart.set_phase(PHASE_SHI_TOMASI)
        mask = np.zeros_like(prev_gray)
        rx1, ry1 = max(0, roi_x1), max(0, roi_y1)
        rx2, ry2 = min(IMG_W, roi_x1 + roi_w), min(IMG_H, roi_y1 + roi_h)
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        mask[ry1:ry2, rx1:rx2] = 255

        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.02,
                                     minDistance=5, mask=mask)
        if p0 is None or len(p0) < MIN_INLIERS:
            return None

        self.flowchart.set_phase(PHASE_LK_FLOW)
        lk_params = dict(winSize=(45, 45), maxLevel=5,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        if abs(guess_dx) > 0.1 or abs(guess_dy) > 0.1:
            p1_guess = p0.copy()
            p1_guess[:, 0, 0] += guess_dx
            p1_guess[:, 0, 1] += guess_dy
            lk_params['flags'] = cv2.OPTFLOW_USE_INITIAL_FLOW
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, p1_guess, **lk_params)
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)

        if p1 is None:
            return None

        good_new, good_old = p1[st == 1], p0[st == 1]
        if len(good_new) < MIN_INLIERS:
            return None

        self.flowchart.set_phase(PHASE_HOMOGRAPHY)
        src_pts, dst_pts = good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2)
        H, mask_hom = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask_hom is None or int(mask_hom.sum()) < MIN_INLIERS:
            return None

        self.flowchart.set_phase(PHASE_PROJECT)
        pt_final = np.float32([[target_x, target_y]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pt_final, H)
        return float(projected[0, 0, 0]), float(projected[0, 0, 1]), int(mask_hom.sum())

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

    # ── Mouse / ROI ───────────────────────────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if self.is_busy:
            return
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
                self.status = f"ROI {w}x{h}px → recommended zoom x{self.optimal_zoom:.1f}. Press 'c'"

    def _best_zoom(self, w, h):
        for z_int in range(200, 9, -1):
            z = round(z_int / 10.0, 1)
            fx, fy = self._get_zoom_factors(z)
            if w * fx <= IMG_W * 0.95 and h * fy <= IMG_H * 0.95:
                return z
        return 1.0

    # ── PTZ commands ──────────────────────────────────────────────────────────
    def _set_zoom(self, level):
        if self.zoom_srv:
            try:
                self.zoom_srv(int(round(level)))
            except rospy.ServiceException as e:
                print(f"❌ Error sending zoom: {e}")
        else:
            print(f"⚠️ Zoom simulation: x{float(level):.1f}")

    def _send_cmd(self, yaw, pitch, speed=20):
        cmd = PanTiltCmdDeg()
        cmd.yaw, cmd.pitch, cmd.speed = yaw, pitch, speed
        self.pub_cmd.publish(cmd)

    def _calculate_and_send_direct_cmd(self, ex, ey, zoom_level):
        """Direct incremental yaw/pitch command (no TF2)."""
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

    def _move_manual(self, dyaw, dpitch):
        if self.is_busy:
            return
        target_yaw = round(self.current_yaw) + dyaw
        target_pitch = round(self.current_pitch) + dpitch
        target_yaw = max(self.PAN_MIN_DEG, min(self.PAN_MAX_DEG, target_yaw))
        target_pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, target_pitch))
        self.status = f"🕹️ Manual: yaw={target_yaw}° pitch={target_pitch}°"
        print(self.status)
        self._send_cmd(target_yaw, target_pitch, speed=15)

    # ── ROI helpers ───────────────────────────────────────────────────────────
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

    # ── Optical Flow correction pipeline (single-shot) ────────────────────────
    def _run_optflow(self, target_zoom, cx_cursor, cy_cursor, x, y, orig_w, orig_h):
        results = []

        self._set_zoom(1.0)
        time.sleep(0.5)
        frame_orig = self._get_frame()
        if frame_orig is None:
            return results

        initial_cx, initial_cy = x + orig_w / 2.0, y + orig_h / 2.0
        crop_w_init = max(int(orig_w * 1.5), 250)
        crop_h_init = max(int(orig_h * 1.5), 250)
        frame_orig_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(initial_cx, initial_cy, crop_w_init, crop_h_init)

        ex_init = initial_cx - cx_cursor
        ey_init = initial_cy - cy_cursor

        self.flowchart.set_phase(PHASE_CHECK_40PX)
        if abs(ex_init) > PRE_CENTRE_PX or abs(ey_init) > PRE_CENTRE_PX:
            self.flowchart.set_phase(PHASE_PRE_CENTRE)
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

        # Build zoom step ladder
        self.flowchart.set_phase(PHASE_SET_ZOOM)
        total_distance = target_zoom - 1.0
        zoom_steps = []
        if total_distance > 0:
            num_steps = math.ceil(total_distance / MAX_SAFE_JUMP)
            dynamic_step = total_distance / num_steps
            curr_z = 1.0
            for _ in range(num_steps):
                curr_z += dynamic_step
                zoom_steps.append(round(curr_z, 1))
            if zoom_steps:
                zoom_steps[-1] = float(round(target_zoom, 1))

        n_inliers = 0
        scene_gray_final = prev_gray

        for step_zoom in zoom_steps:
            self.flowchart.set_phase(PHASE_ZOOM_IN)
            self.status = f"[OptFlow] Applying zoom x{step_zoom:.1f}..."
            self._set_zoom(step_zoom)
            time.sleep(STABILISE_S)

            frame_z = self._get_frame()
            if frame_z is None:
                return results
            scene_gray_before = cv2.cvtColor(frame_z, cv2.COLOR_BGR2GRAY)

            self.flowchart.set_phase(PHASE_SAVE_NEW_TPL)
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

            cx_found, cy_found, n_inliers = result
            ex_step = cx_found - cx_cursor
            ey_step = cy_found - cy_cursor
            err_px = math.sqrt(ex_step**2 + ey_step**2)
            deg_x, deg_y = self._px_to_deg(ex_step, ey_step, step_zoom)
            err_deg = math.sqrt(deg_x**2 + deg_y**2)

            results.append({
                "zoom": step_zoom, "label": f"x{step_zoom:.1f}",
                "ex": ex_step, "ey": ey_step, "err_px": err_px,
                "ex_deg": deg_x, "ey_deg": deg_y, "err_deg": err_deg,
                "duration_s": match_duration,
            })

            self.flowchart.set_phase(PHASE_CHECK_15PX)
            if abs(ex_step) > MICRO_CORRECT_PX or abs(ey_step) > MICRO_CORRECT_PX:
                self.flowchart.set_phase(PHASE_MICRO_CORRECT)
                self.status = f"[OptFlow] Micro-centering at x{step_zoom:.1f}..."
                self._calculate_and_send_direct_cmd(ex_step, ey_step, step_zoom)
                time.sleep(2.0)

                frame_z_after = self._get_frame()
                scene_gray_after = cv2.cvtColor(frame_z_after, cv2.COLOR_BGR2GRAY)

                rx1_mc, ry1_mc, rw_mc, rh_mc = self._get_roi_bbox(
                    cx_found, cy_found, EPICENTER_SIZE, EPICENTER_SIZE)

                res_centered = self._find_point_optical_flow(
                    scene_gray_before, scene_gray_after,
                    cx_found, cy_found,
                    rx1_mc, ry1_mc, rw_mc, rh_mc,
                    step_zoom, step_zoom,
                    guess_dx=-ex_step, guess_dy=-ey_step
                )
                if res_centered is not None:
                    cx_found, cy_found, n_inliers = res_centered
                else:
                    cx_found, cy_found = cx_cursor, cy_cursor

                scene_gray_final = scene_gray_after
            else:
                scene_gray_final = scene_gray_before

            current_cx, current_cy = cx_found, cy_found
            self._update_roi_visual(current_cx, current_cy, step_zoom, orig_w, orig_h)

            self.flowchart.set_phase(PHASE_ACCUMULATE)
            self.flowchart.set_phase(PHASE_LAST_STEP)

            if step_zoom != zoom_steps[-1]:
                prev_gray = scene_gray_final
                zoom_prev = step_zoom
                roi_x1, roi_y1, roi_w, roi_h = self._get_roi_bbox(
                    current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)

        self.last_target_px = (current_cx, current_cy)
        self.last_inliers = n_inliers

        ex_final = current_cx - cx_cursor
        ey_final = current_cy - cy_cursor
        self.last_error_px = (ex_final, ey_final)

        self.flowchart.set_phase(PHASE_FINAL_CMD)
        yaw_final, pitch_final = self._calculate_and_send_direct_cmd(ex_final, ey_final, target_zoom)

        # ── Re-measure after final correction ──
        time.sleep(2.0)
        frame_post = self._get_frame()
        if frame_post is not None:
            scene_post = cv2.cvtColor(frame_post, cv2.COLOR_BGR2GRAY)
            rx1_p, ry1_p, rw_p, rh_p = self._get_roi_bbox(
                current_cx, current_cy, EPICENTER_SIZE, EPICENTER_SIZE)
            res_post = self._find_point_optical_flow(
                scene_gray_final, scene_post, current_cx, current_cy,
                rx1_p, ry1_p, rw_p, rh_p,
                target_zoom, target_zoom,
                guess_dx=-ex_final, guess_dy=-ey_final
            )
            if res_post is not None:
                cx_post, cy_post, _ = res_post
                ex_post = cx_post - cx_cursor
                ey_post = cy_post - cy_cursor
                err_post = math.sqrt(ex_post**2 + ey_post**2)
                deg_x_post, deg_y_post = self._px_to_deg(ex_post, ey_post, target_zoom)
                err_deg_post = math.sqrt(deg_x_post**2 + deg_y_post**2)

                results.append({
                    "zoom": target_zoom, "label": f"x{target_zoom:.1f}_post",
                    "ex": ex_post, "ey": ey_post, "err_px": err_post,
                    "ex_deg": deg_x_post, "ey_deg": deg_y_post, "err_deg": err_deg_post,
                    "duration_s": 0.0, "is_post_correction": True,
                })
                self.last_target_px = (cx_post, cy_post)
                self.last_error_px = (ex_post, ey_post)
                current_cx, current_cy = cx_post, cy_post

        self._update_roi_visual(cx_cursor, cy_cursor, target_zoom, orig_w, orig_h)

        self.status = f"✅ [OptFlow] yaw={yaw_final}° pitch={pitch_final}° in={n_inliers}"
        print(f"  {self.status}")
        return results

    # ── Run + save ────────────────────────────────────────────────────────────
    def _zoom_and_correct(self):
        try:
            target_zoom = self.current_target_zoom
            cx_cursor, cy_cursor = CURSOR_CX, CURSOR_CY
            x, y, orig_w, orig_h = self.roi
            orig_roi = (x, y, orig_w, orig_h)

            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.image_dir, f"optflow_once_{ts_str}")
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  OptFlow single-shot @ zoom x{target_zoom:.1f}")
            print(f"{'='*60}")

            results = self._run_optflow(target_zoom, cx_cursor, cy_cursor, x, y, orig_w, orig_h)

            if results:
                post = [r for r in results if r.get("is_post_correction")]
                final = post[-1] if post else results[-1]
                print(f"\n  Final error: {final['err_deg']:.4f}° ({final['err_px']:.1f}px)\n")

            self.flowchart.set_phase(PHASE_SAVE_RESULTS)
            self._save_results(results, orig_roi, out_dir, ts_str, target_zoom)

            self.flowchart.set_phase(PHASE_DONE)

        except Exception as e:
            self.status = f"❌ Error: {e}"
            import traceback
            traceback.print_exc()
        finally:
            self.is_busy = False

    def _save_results(self, results, roi, out_dir, ts, target_zoom):
        xr, yr, wr, hr = roi
        roi_cx, roi_cy = xr + wr / 2.0, yr + hr / 2.0

        # ── Per-step CSV ──
        csv_path = os.path.join(out_dir, "results_steps.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# OptFlow single-shot", ts])
            w.writerow(["# ROI", f"x={xr} y={yr} w={wr} h={hr}",
                        f"cx={roi_cx:.1f}", f"cy={roi_cy:.1f}"])
            w.writerow(["# Target zoom", f"x{target_zoom:.1f}"])
            w.writerow([])
            w.writerow(["zoom", "label", "ex_px", "ey_px", "err_px",
                        "ex_deg", "ey_deg", "err_deg", "duration_s"])
            for r in results:
                w.writerow([r["zoom"], r["label"],
                            f"{r['ex']:.2f}", f"{r['ey']:.2f}", f"{r['err_px']:.2f}",
                            f"{r['ex_deg']:.4f}", f"{r['ey_deg']:.4f}", f"{r['err_deg']:.4f}",
                            f"{r['duration_s']:.4f}"])
        print(f"  CSV saved: {csv_path}")

        if not results:
            print("  ⚠️ No results to plot.")
            return

        # ── Single-run summary in boxplot style (one box per metric) ──
        # Use all per-step samples to form the box; mark final post-correction in red.
        err_deg_vals = [r["err_deg"] for r in results]
        ex_deg_vals  = [r["ex_deg"]  for r in results]
        ey_deg_vals  = [r["ey_deg"]  for r in results]
        err_px_vals  = [r["err_px"]  for r in results]
        ex_px_vals   = [r["ex_px"] if "ex_px" in r else r["ex"] for r in results]
        ey_px_vals   = [r["ey_px"] if "ey_px" in r else r["ey"] for r in results]

        post = [r for r in results if r.get("is_post_correction")]
        final = post[-1] if post else results[-1]

        title = (f"OptFlow single-shot {ts}\n"
                 f"ROI {wr}x{hr}px @ ({roi_cx:.0f},{roi_cy:.0f})  zoom x{target_zoom:.1f}")

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(title, fontsize=11, y=1.00)

        plot_specs = [
            (axes[0, 0], err_deg_vals, final["err_deg"], "Euclidean error", "err (°)"),
            (axes[0, 1], ex_deg_vals,  final["ex_deg"],  "X error",         "ex (°)"),
            (axes[0, 2], ey_deg_vals,  final["ey_deg"],  "Y error",         "ey (°)"),
            (axes[1, 0], err_px_vals,  final["err_px"],  "Euclidean error", "err (px)"),
            (axes[1, 1], ex_px_vals,   final["ex"],      "X error",         "ex (px)"),
            (axes[1, 2], ey_px_vals,   final["ey"],      "Y error",         "ey (px)"),
        ]

        for ax, data, final_val, ttl, ylabel in plot_specs:
            bp = ax.boxplot([data], positions=[1], patch_artist=True, notch=False,
                            medianprops=dict(color="black", linewidth=2), widths=0.5)
            bp["boxes"][0].set_facecolor("#4CAF50")
            bp["boxes"][0].set_alpha(0.6)
            # Overlay raw step samples
            ax.scatter([1] * len(data), data, color="#2E7D32", alpha=0.8,
                       zorder=5, s=40, label="step samples")
            # Mark final post-correction
            ax.scatter([1], [final_val], color="red", marker="*", s=200,
                       zorder=10, label="final (post)")
            ax.set_xticks([1])
            ax.set_xticklabels(["OptFlow"])
            ax.set_title(ttl)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)
            if "px" in ylabel and ttl == "Euclidean error":
                ax.axhline(MICRO_CORRECT_PX, color="red", linestyle=":", alpha=0.6,
                           label=f"micro thr ({MICRO_CORRECT_PX}px)")
            elif ttl != "Euclidean error":
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4)
            ax.legend(fontsize=7, loc="best")

        plt.tight_layout()
        out_png = os.path.join(out_dir, "summary_boxplot.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {out_png}")
        print(f"  Output dir: {out_dir}")

    # ── Overlay and main loop ─────────────────────────────────────────────────
    def _draw_overlay(self, img):
        if self.drawing:
            cv2.rectangle(img, (self.start_x, self.start_y),
                          (self.current_x, self.current_y), (0, 255, 255), 1)

        if self.roi_selected and self.roi:
            x, y, w, h = self.roi
            color = (0, 255, 0) if self.template_saved else (0, 255, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cx_roi, cy_roi = x + w // 2, y + h // 2
            cv2.drawMarker(img, (cx_roi, cy_roi), (0, 165, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, f"Zoom x{self.optimal_zoom:.1f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.last_target_px is not None:
            tx, ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.drawMarker(img, (tx, ty), (0, 165, 255), cv2.MARKER_CROSS, 24, 3)

        if self.last_target_px is not None and self.last_error_px is not None:
            tx, ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.arrowedLine(img, (tx, ty), (CURSOR_CX, CURSOR_CY),
                            (255, 255, 0), 2, tipLength=0.15)

        cv2.drawMarker(img, (CURSOR_CX, CURSOR_CY), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

        cv2.putText(img, self.status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        if self.is_busy:
            cv2.putText(img, "RUNNING - Controls Locked", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    def _main_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            frame = self._get_frame()
            if frame is not None:
                vis = frame.copy()
                self._draw_overlay(vis)
                cv2.imshow("OptFlow Correct Once", vis)

                # Idle-state phase tracking (only when not running pipeline)
                if not self.is_busy:
                    if self.template_saved:
                        self.flowchart.set_phase(PHASE_PRESS_Z)
                    elif self.roi_selected:
                        self.flowchart.set_phase(PHASE_PRESS_C)
                    else:
                        self.flowchart.set_phase(PHASE_ROI_CHECK)
            else:
                if not self.is_busy:
                    self.flowchart.set_phase(PHASE_WAIT_CAMERA)

            cv2.imshow("Flowchart", self.flowchart.render())

            key = cv2.waitKeyEx(1)
            if key == -1:
                rate.sleep()
                continue

            char_key = key & 0xFF

            if char_key in (ord('w'), ord('W')):
                self._move_manual(0, -1)
            elif char_key in (ord('s'), ord('S')):
                self._move_manual(0, 1)
            elif char_key in (ord('a'), ord('A')):
                self._move_manual(1, 0)
            elif char_key in (ord('d'), ord('D')):
                self._move_manual(-1, 0)
            elif char_key == 27:
                break
            elif char_key == ord('r') and not self.is_busy:
                self.roi = None
                self.roi_selected = self.template_saved = False
                self.template_gray = None
                self.last_target_px = self.last_error_px = None
                self._set_zoom(1.0)
                self.status = "Reset. Ready."
                self.flowchart.reset()
                self.flowchart.set_phase(PHASE_WAIT_CAMERA)
            elif char_key == ord('c') and self.roi_selected and not self.is_busy:
                frame = self._get_frame()
                if frame is not None and self.roi is not None:
                    x, y, w, h = self.roi
                    self.template_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                    self.template_saved = True
                    self.status = f"Template saved. Press 'z' for zoom x{self.optimal_zoom:.1f}"
                    self.flowchart.set_phase(PHASE_SAVE_TEMPLATE)
            elif char_key == ord('z') and self.template_saved and not self.is_busy:
                # No prompt — apply recommended zoom immediately
                self.is_busy = True
                self.current_target_zoom = self.optimal_zoom
                self.last_target_px = self.last_error_px = None
                threading.Thread(target=self._zoom_and_correct, daemon=True).start()

            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        OptFlowCorrectOnce()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

IMG_W, IMG_H = 1280, 720

CURSORS = {
    1: (640, 360),
    2: (643, 415),
    3: (643, 387),
}
CURSOR_LABELS = {1: "Image centre", 2: "Camera centre", 3: "Extra point"}
CURSOR_COLORS = {1: (0, 255, 0), 2: (255, 0, 0), 3: (255, 0, 255)}

LOWE_RATIO  = 0.75
MIN_INLIERS = 6
STABILISE_S = 1.5
EPICENTER_SIZE = 450

MICRO_CORRECT_PX = 15
METHOD_NAMES = ["ORB"]

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
PHASE_KEYPOINTS     = "keypoints"
PHASE_MATCH         = "match"
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

FC_W, FC_H = 680, 700


class FlowchartRenderer:
    """Renders the full ORB-correction flowchart with real-time highlights."""

    def __init__(self):
        self.history = set()
        self.current = PHASE_IDLE

        # (id, type, x, y, w, h, label, [label2])
        self.nodes = [
            # --- Left column: main user flow (centred at x=230) ---
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
            ("keypoints",        "bold",    115, 646, 230, 40, "Establish keypoints", "features=2000"),

            # --- Right column: ORB pipeline (centred at x=490) ---
            ("match",            "bold",    390, 646, 200, 40, "Match keypoints", "between frames"),
            ("homography",       "bold",    390, 576, 200, 40, "Compute homography", "RANSAC"),
            ("project",          "bold",    390, 510, 200, 34, "Project target via H"),

            # --- Right column: decision loop (above pipeline, cx=490) ---
            ("check_15px",       "diamond", 415, 438, 150, 48, "ex|ey > 15px?"),
            ("micro_correct",    "rect",    610, 448,  60, 34, "Micro", "correct"),
            ("accumulate",       "dashed",  430, 376, 120, 34, "Accumulate err"),
            ("last_step",        "diamond", 425, 306, 130, 44, "Last step?"),

            # --- Top-right: final (inverted: End at top, Send final nearest to loop) ---
            ("done",             "pill",    440, 130, 100, 32, "End"),
            ("save_results",     "dashed",  390, 174, 200, 34, "Save results + plots"),
            ("final_cmd",        "bold",    390, 222, 200, 34, "Send final correction"),
        ]

        # Arrows: list of (waypoints, label, label_pos)
        # Left column cx=230, right column cx=490
        # Diamond edges: top=(cx,y) bottom=(cx,y+h) left=(x,cy) right=(x+w,cy)
        self.arrows = [
            # --- Left column ---
            ([(230,42),(230,62)], None, None),
            ([(230,96),(230,114)], None, None),
            # ROI No -> loop back right
            ([(310,140),(338,140),(338,79),(320,79)], "No", (314,136)),
            # ROI Yes -> Press c
            ([(230,166),(230,194)], "Yes", (236,186)),
            # Press c Yes -> Save template
            ([(230,242),(230,268)], "Yes", (236,260)),
            # Save template -> Press z
            ([(230,302),(230,326)], None, None),
            # Press z Yes -> check 40px
            ([(230,374),(230,402)], "Yes", (236,396)),
            # check 40px Yes -> Pre-centre (left tip x=150)
            ([(150,428),(96,428)], "Yes", (104,422)),
            # Pre-centre -> Set zoom steps
            ([(50,446),(50,495),(140,495)], None, None),
            # check 40px No -> Set zoom (bottom cx=230)
            ([(230,454),(230,478)], "No", (236,470)),
            # Set zoom -> Zoom in
            ([(230,512),(230,534)], None, None),
            # Zoom in -> Save new template
            ([(230,568),(230,590)], None, None),
            # Save new template -> Keypoints
            ([(230,624),(230,646)], None, None),
            # Keypoints -> Match (horizontal)
            ([(345,666),(390,666)], None, None),

            # --- Right column pipeline (upward) ---
            # Match -> Homography
            ([(490,646),(490,616)], None, None),
            # Homography -> Project
            ([(490,576),(490,544)], None, None),
            # Project -> check 15px
            ([(490,510),(490,486)], None, None),

            # --- Right column decision loop ---
            # check 15px Yes -> Micro correct (right tip x+w=565, cy=462)
            ([(565,462),(610,462)], "Yes", (572,456)),
            # Micro correct -> up to Last step (via right side, right tip at x+w=555, cy=328)
            ([(640,448),(640,328),(555,328)], None, None),
            # check 15px No -> Accumulate (top cx=490, y=438)
            ([(490,438),(490,410)], "No", (496,432)),
            # Accumulate -> Last step (top -> bottom of last_step cx=490, y+h=350)
            ([(490,376),(490,350)], None, None),
            # Last step No -> loops back to Zoom in (left tip x=425, cy=328)
            ([(425,328),(370,328),(370,551),(335,551)], "No", (376,322)),
            # Last step Yes -> straight up to Send final correction (top of diamond cx=490, y=306)
            ([(490,306),(490,256)], "Yes", (496,296)),

            # --- Top-right final sequence (upward) ---
            # Send final -> Save results
            ([(490,222),(490,208)], None, None),
            # Save results -> End
            ([(490,174),(490,162)], None, None),
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
            for i in range(len(wps)-1):
                if i == len(wps)-2:
                    cv2.arrowedLine(img, wps[i], wps[i+1], _ARROW, 1, tipLength=0.06)
                else:
                    cv2.line(img, wps[i], wps[i+1], _ARROW, 1)
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
        for i, (col, lbl) in enumerate([(_ACTIVE,"Active"), (_DONE,"Done"), (_DEFAULT,"Pending")]):
            ly = ly0 + i * 20
            cv2.rectangle(img, (lx, ly), (lx+12, ly+12), col, -1)
            cv2.putText(img, lbl, (lx+18, ly+11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _TXT, 1, cv2.LINE_AA)
        return img

    # --- Node drawing ---
    def _draw_node(self, img, ntype, x, y, w, h, label, label2, state):
        cx, cy = x + w//2, y + h//2

        if ntype == "pill":
            col = _ACTIVE if state=="active" else (_DONE if state=="done" else _DEFAULT)
            r = h // 2
            cv2.rectangle(img, (x+r,y), (x+w-r,y+h), col, -1)
            cv2.ellipse(img, (x+r,cy), (r,r), 0, 90, 270, col, -1)
            cv2.ellipse(img, (x+w-r,cy), (r,r), 0, -90, 90, col, -1)
            cv2.putText(img, label, (cx-len(label)*4, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TXT_D, 1, cv2.LINE_AA)
            if state == "active":
                cv2.rectangle(img, (x+r,y), (x+w-r,y+h), (0,140,255), 2)
                cv2.ellipse(img, (x+r,cy), (r,r), 0, 90, 270, (0,140,255), 2)
                cv2.ellipse(img, (x+w-r,cy), (r,r), 0, -90, 90, (0,140,255), 2)

        elif ntype == "diamond":
            if state=="active":    col, brd = _DEC_ACT, (0,140,255)
            elif state=="done":    col, brd = _DONE, (60,160,60)
            else:                  col, brd = _DEC, (140,140,140)
            pts = np.array([[cx,y],[x+w,cy],[cx,y+h],[x,cy]], np.int32)
            cv2.fillPoly(img, [pts], col)
            cv2.polylines(img, [pts], True, brd, 2 if state=="active" else 1)
            fs = 0.35 if len(label)>14 else 0.40
            tw = int(len(label)*(5 if fs<0.4 else 6))
            cv2.putText(img, label, (cx-tw//2, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, _TXT_D, 1, cv2.LINE_AA)

        elif ntype in ("rect","dashed","bold"):
            if state=="active":
                col = {
                    "dashed": _DASH_ACT, "bold": _BOLD_ACT
                }.get(ntype, _ACTIVE)
                brd = (0,140,255) if ntype!="bold" else (0,100,255)
            elif state=="done":
                col, brd = _DONE, (60,160,60)
            else:
                col = (50,50,50) if ntype=="dashed" else (45,45,45)
                brd = {"dashed":_DASH,"bold":_BOLD}.get(ntype, _DEFAULT)

            cv2.rectangle(img, (x,y), (x+w,y+h), col, -1)
            thk = 2 if state=="active" else 1
            if ntype=="dashed" and state=="default":
                self._dashed_rect(img, x, y, w, h, brd)
            else:
                cv2.rectangle(img, (x,y), (x+w,y+h), brd, thk)

            tc = _TXT_D if state in ("active","done") else _TXT
            if label2:
                cv2.putText(img, label, (cx-len(label)*4, cy-2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.38, tc, 1, cv2.LINE_AA)
                tc2 = (60,60,60) if state in ("active","done") else (160,160,160)
                cv2.putText(img, label2, (cx-len(label2)*3, cy+14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, tc2, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, label, (cx-len(label)*4, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, tc, 1, cv2.LINE_AA)

    def _dashed_rect(self, img, x, y, w, h, color, dash=8, gap=5):
        for (p1,p2) in [((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),((x+w,y+h),(x,y+h)),((x,y+h),(x,y))]:
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            l = math.sqrt(dx*dx+dy*dy)
            if l==0: continue
            ux, uy = dx/l, dy/l
            d = 0
            while d < l:
                s = (int(p1[0]+ux*d), int(p1[1]+uy*d))
                e = (int(p1[0]+ux*min(d+dash,l)), int(p1[1]+uy*min(d+dash,l)))
                cv2.line(img, s, e, color, 1)
                d += dash+gap


# ============================================================
# ORB Feature Matching
# ============================================================
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
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None: return None
    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < MIN_INLIERS: return None
    pt = np.float32([[target_x, target_y]]).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(pt, H)
    return float(projected[0,0,0]), float(projected[0,0,1]), n_inliers, H, mask, kp1, kp2, good


# ============================================================
# Main class
# ============================================================
class OrbCorrectOnce:

    def __init__(self):
        rospy.init_node('orb_correct_once', anonymous=True)
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
        self.benchmark_results = {m: [] for m in METHOD_NAMES}
        self.status = "Draw a ROI around the target, then press 'c'"

        # Flowchart
        self.flowchart = FlowchartRenderer()
        self.flowchart.set_phase(PHASE_WAIT_CAMERA)

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
            rospy.logwarn("/set_zoom service not found. Make sure the node is running.")
            self.zoom_srv = None

        cv2.namedWindow("ORB Correct Once")
        cv2.setMouseCallback("ORB Correct Once", self._mouse_cb)
        cv2.namedWindow("Flowchart", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Flowchart", FC_W, FC_H)

        print("\n  orb_correct_once started (With Flowchart)")
        print("   Draw ROI -> 'c' save template -> 'z' zoom + correct")
        print("   W/A/S/D -> Move camera manually (1 deg)")
        print("   1/2/3 cursor   0 all   r reset to x1   ESC quit\n")

        self._main_loop()

    # --- Math models ---
    def _get_fovs(self, target_z):
        z = max(1.0, min(20.0, float(target_z)))
        t = (z - 1.0) / 19.0
        k = 1.78
        fov_h = 63.7 * ((1.0-t)**k) + 2.3 * t
        fov_v = 35.84 * ((1.0-t)**k) + 1.3 * t
        return fov_h, fov_v

    def _get_zoom_factors(self, target_z):
        fh1, fv1 = self._get_fovs(1.0)
        fhz, fvz = self._get_fovs(target_z)
        fx = math.tan(math.radians(fh1/2.0)) / math.tan(math.radians(fhz/2.0))
        fy = math.tan(math.radians(fv1/2.0)) / math.tan(math.radians(fvz/2.0))
        return fx, fy

    # --- Callbacks ---
    def _img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock: self.image = frame
        except Exception as e:
            rospy.logwarn(f"Image error: {e}")

    def _status_cb(self, msg):
        self.current_yaw = msg.yaw_now
        self.current_pitch = msg.pitch_now

    def _get_frame(self):
        with self.image_lock:
            return self.image.copy() if self.image is not None else None

    def _mouse_cb(self, event, x, y, flags, param):
        if self.is_busy: return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_x, self.current_y = x, y
            self.roi_selected = self.template_saved = False
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
                self.status = f"ROI {w}x{h}px -> zoom x{self.optimal_zoom:.1f}. Press 'c'"

    def _best_zoom(self, w, h):
        for z_int in range(200, 9, -1):
            z = round(z_int / 10.0, 1)
            fx, fy = self._get_zoom_factors(z)
            if w*fx <= IMG_W*0.95 and h*fy <= IMG_H*0.95: return z
        return 1.0

    def _set_zoom(self, level):
        if self.zoom_srv:
            try: self.zoom_srv(float(level))
            except rospy.ServiceException as e: print(f"Error sending zoom: {e}")
        else: print(f"Zoom Simulation: x{float(level):.1f}")

    def _send_cmd(self, yaw, pitch, speed=20):
        cmd = PanTiltCmdDeg()
        cmd.yaw, cmd.pitch, cmd.speed = yaw, pitch, speed
        self.pub_cmd.publish(cmd)

    def _calculate_and_send_tf_cmd(self, ex, ey, zoom_level):
        fov_h, fov_v = self._get_fovs(zoom_level)
        ang_x = -ex / (IMG_W / fov_h)
        ang_y = -ey / (IMG_H / fov_v)
        speed = 25 if math.sqrt(ang_x**2 + ang_y**2) > 5.0 else 15
        target_cam = PointStamped()
        target_cam.header.frame_id = self.TF_CAMERA
        target_cam.header.stamp = rospy.Time(0)
        target_cam.point.x = 20.0
        target_cam.point.y = 20.0 * math.tan(math.radians(ang_x))
        target_cam.point.z = 20.0 * math.tan(math.radians(ang_y))
        try:
            tf = self.tf_buffer.lookup_transform(self.TF_BASE, self.TF_CAMERA, rospy.Time(0), rospy.Duration(1.0))
            tb = tf2_geometry_msgs.do_transform_point(target_cam, tf)
            cx = tf.transform.translation.x
            cy = tf.transform.translation.y
            cz = tf.transform.translation.z
            dx, dy, dz = tb.point.x-cx, tb.point.y-cy, tb.point.z-cz
            pan = math.degrees(math.atan2(dy, dx))
            tilt = math.degrees(math.atan2(dz, math.sqrt(dx**2+dy**2)))
            ny = round(max(self.PAN_MIN_DEG, min(self.PAN_MAX_DEG, pan)))
            np_ = round(max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, -tilt)))
            self._send_cmd(ny, np_, speed=speed)
            return ny, np_
        except: return 0.0, 0.0

    def _update_roi_visual(self, cx, cy, zl, ow, oh):
        fx, fy = self._get_zoom_factors(zl)
        nw, nh = ow*fx, oh*fy
        self.roi = (int(cx-nw/2), int(cy-nh/2), int(nw), int(nh))

    def _get_roi_crop(self, img_g, cx, cy, cw, ch):
        x1 = max(0, int(cx-cw/2)); y1 = max(0, int(cy-ch/2))
        x2 = min(IMG_W, int(cx+cw/2)); y2 = min(IMG_H, int(cy+ch/2))
        return img_g[y1:y2, x1:x2], cx-x1, cy-y1

    def _move_manual(self, dyaw, dpitch):
        if self.is_busy: return
        ty = max(self.PAN_MIN_DEG, min(self.PAN_MAX_DEG, round(self.current_yaw)+dyaw))
        tp = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, round(self.current_pitch)+dpitch))
        self.status = f"Manual: yaw={ty} pitch={tp}"
        self._send_cmd(ty, tp, speed=15)

    # ==========================================
    # ZOOM AND CORRECT — with flowchart phases
    # ==========================================
    def _zoom_and_correct(self):
        try:
            self.benchmark_results = {m: [] for m in METHOD_NAMES}
            target_zoom = self.current_target_zoom
            cx_cursor, cy_cursor = CURSORS[self.cursor_sel]
            x, y, orig_w, orig_h = self.roi

            max_safe_zoom = 1.0
            for z_int in range(10, 201):
                z = round(z_int/10.0, 1)
                fx, fy = self._get_zoom_factors(z)
                if orig_w*fx > IMG_W*0.95 or orig_h*fy > IMG_H*0.95: break
                max_safe_zoom = z
            if target_zoom > max_safe_zoom:
                print(f"Optical Limit: adjusting to x{max_safe_zoom:.1f}")
                target_zoom = max_safe_zoom

            self._set_zoom(1.0); time.sleep(0.5)
            frame_orig = self._get_frame()
            if frame_orig is None: self.status = "No frame"; return

            initial_cx = x + orig_w/2.0
            initial_cy = y + orig_h/2.0
            cw_i = max(int(orig_w*1.5), 250); ch_i = max(int(orig_h*1.5), 250)
            fog = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            tpl_o, lcx_o, lcy_o = self._get_roi_crop(fog, initial_cx, initial_cy, cw_i, ch_i)

            ex_i = initial_cx - cx_cursor; ey_i = initial_cy - cy_cursor

            # ---- check 40px ----
            self.flowchart.set_phase(PHASE_CHECK_40PX)
            time.sleep(0.15)

            if abs(ex_i) > 40 or abs(ey_i) > 40:
                self.flowchart.set_phase(PHASE_PRE_CENTRE)
                self.status = "Pre-centring target..."
                self._calculate_and_send_tf_cmd(ex_i, ey_i, 1.0)
                time.sleep(2.0)
                f1 = self._get_frame(); sg1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                r0 = find_template_point(tpl_o, sg1, lcx_o, lcy_o)
                if r0 is None: self.status = "Failed pre-centre"; return
                ccx, ccy = r0[0], r0[1]; frame_base = f1
            else:
                ccx, ccy = initial_cx, initial_cy; frame_base = frame_orig

            self._update_roi_visual(ccx, ccy, 1.0, orig_w, orig_h)

            # ---- set zoom steps ----
            self.flowchart.set_phase(PHASE_SET_ZOOM)
            sgb = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)
            tpl_g, lcx, lcy = self._get_roi_crop(sgb, ccx, ccy, EPICENTER_SIZE, EPICENTER_SIZE)

            td = target_zoom - 1.0; zoom_steps = []
            if td > 0:
                ns = math.ceil(td / 4.0); ds = td / ns
                cz = 1.0
                for _ in range(ns): cz += ds; zoom_steps.append(round(cz,1))
                if zoom_steps: zoom_steps[-1] = float(round(target_zoom,1))

            H_final = None; n_inliers = 0

            for step_zoom in zoom_steps:
                # ---- zoom in ----
                self.flowchart.set_phase(PHASE_ZOOM_IN)
                self.status = f"Applying zoom x{step_zoom:.1f}..."
                self._set_zoom(step_zoom); time.sleep(STABILISE_S)
                fz = self._get_frame()
                if fz is None: return
                sg = cv2.cvtColor(fz, cv2.COLOR_BGR2GRAY)

                # ---- keypoints ----
                self.flowchart.set_phase(PHASE_KEYPOINTS)
                self.status = f"Keypoints at x{step_zoom:.1f}..."
                time.sleep(0.1)

                # ---- match ----
                self.flowchart.set_phase(PHASE_MATCH)
                self.status = f"Matching at x{step_zoom:.1f}..."
                ts = time.time()
                result = find_template_point(tpl_g, sg, lcx, lcy)
                md = time.time() - ts

                if result is None:
                    self.status = f"ORB failed at x{step_zoom:.1f}"; return
                cxf, cyf, n_inliers, H_final, mask, kp1, kp2, good = result

                # ---- homography ----
                self.flowchart.set_phase(PHASE_HOMOGRAPHY)
                time.sleep(0.05)

                # ---- project ----
                self.flowchart.set_phase(PHASE_PROJECT)
                exs = cxf - cx_cursor; eys = cyf - cy_cursor
                err = math.sqrt(exs**2 + eys**2)
                self.benchmark_results["ORB"].append({
                    "zoom": step_zoom, "label": f"x{step_zoom:.1f}",
                    "ex": exs, "ey": eys, "err_px": err, "duration_s": md})

                # ---- check 15px ----
                self.flowchart.set_phase(PHASE_CHECK_15PX)
                time.sleep(0.1)

                if abs(exs) > MICRO_CORRECT_PX or abs(eys) > MICRO_CORRECT_PX:
                    self.flowchart.set_phase(PHASE_MICRO_CORRECT)
                    self.status = f"Micro-centring at x{step_zoom:.1f}..."
                    self._calculate_and_send_tf_cmd(exs, eys, step_zoom)
                    time.sleep(2.0)
                    fz = self._get_frame(); sg = cv2.cvtColor(fz, cv2.COLOR_BGR2GRAY)
                    rc = find_template_point(tpl_g, sg, lcx, lcy)
                    if rc is not None: cxf, cyf, n_inliers, H_final = rc[0], rc[1], rc[2], rc[3]
                    else: cxf, cyf = cx_cursor, cy_cursor
                else:
                    self.flowchart.set_phase(PHASE_ACCUMULATE)
                    time.sleep(0.1)

                ccx, ccy = cxf, cyf
                self._update_roi_visual(ccx, ccy, step_zoom, orig_w, orig_h)

                # ---- last step? ----
                self.flowchart.set_phase(PHASE_LAST_STEP)
                time.sleep(0.1)

                if step_zoom != zoom_steps[-1]:
                    self.flowchart.set_phase(PHASE_SAVE_NEW_TPL)
                    tpl_g, lcx, lcy = self._get_roi_crop(sg, ccx, ccy, EPICENTER_SIZE, EPICENTER_SIZE)

            self.last_target_px = (ccx, ccy); self.last_inliers = n_inliers
            exf = ccx - cx_cursor; eyf = ccy - cy_cursor
            self.last_error_px = (exf, eyf)

            # ---- final cmd ----
            self.flowchart.set_phase(PHASE_FINAL_CMD)
            yf, pf = self._calculate_and_send_tf_cmd(exf, eyf, target_zoom)
            self._update_roi_visual(cx_cursor, cy_cursor, target_zoom, orig_w, orig_h)
            self.status = f"Done: yaw={yf} pitch={pf} inliers={n_inliers}"
            print(f"\n{self.status}\n")

            if H_final is not None:
                self._save_debug(fz, tpl_g, H_final, ccx, ccy, cx_cursor, cy_cursor, exf, eyf)

            # ---- save results ----
            self.flowchart.set_phase(PHASE_SAVE_RESULTS)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            od = os.path.join(self.image_dir, f"benchmark_{ts_str}"); os.makedirs(od, exist_ok=True)
            self._save_benchmark_results(self.roi, self.current_yaw, self.current_pitch, od, ts_str)

            # ---- done ----
            self.flowchart.set_phase(PHASE_DONE)

        except Exception as e:
            self.status = f"Error: {e}"; print(e)
        finally:
            self.is_busy = False

    def _save_debug(self, fz, tpl_g, H, cxf, cyf, cxc, cyc, ex, ey):
        d = fz.copy()
        th, tw = tpl_g.shape
        corners = np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1,1,2)
        proj = cv2.perspectiveTransform(corners, H)
        cv2.polylines(d, [np.int32(proj)], True, (0,165,255), 2)
        cv2.drawMarker(d, (int(cxf),int(cyf)), (0,165,255), cv2.MARKER_CROSS, 24, 2)
        cv2.drawMarker(d, (cxc,cyc), CURSOR_COLORS[self.cursor_sel], cv2.MARKER_CROSS, 24, 2)
        cv2.arrowedLine(d, (int(cxf),int(cyf)), (cxc,cyc), (255,255,0), 2, tipLength=0.15)
        cv2.imwrite(os.path.join(self.image_dir, "orb_correction_debug.jpg"), d)

    def _draw_overlay(self, img):
        if self.drawing:
            cv2.rectangle(img, (self.start_x,self.start_y), (self.current_x,self.current_y), (0,255,255), 1)
        if self.roi_selected and self.roi:
            x,y,w,h = self.roi
            c = (0,255,0) if self.template_saved else (0,255,255)
            cv2.rectangle(img, (x,y), (x+w,y+h), c, 2)
            cx_r, cy_r = x+w//2, y+h//2
            cv2.drawMarker(img, (cx_r,cy_r), (0,165,255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(img, "ROI Centre", (cx_r+8,cy_r-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
            cv2.putText(img, f"Zoom x{self.current_target_zoom:.1f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if self.last_target_px is not None:
            tx,ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.drawMarker(img, (tx,ty), (0,165,255), cv2.MARKER_CROSS, 24, 3)
        if self.last_target_px is not None and self.last_error_px is not None:
            cxc, cyc = CURSORS[self.cursor_sel]
            tx,ty = int(self.last_target_px[0]), int(self.last_target_px[1])
            cv2.arrowedLine(img, (tx,ty), (cxc,cyc), (255,255,0), 2, tipLength=0.15)
        for k,(cx,cy) in CURSORS.items():
            if self.cursor_sel in (0,k):
                cv2.drawMarker(img, (cx,cy), CURSOR_COLORS[k], cv2.MARKER_CROSS, 20, 2)
        cv2.putText(img, self.status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        if self.is_busy:
            cv2.putText(img, "AUTOMATIC MODE - Controls Locked", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

    def _save_benchmark_results(self, roi, iy, ip, od, ts):
        xr,yr,wr,hr = roi; rcx,rcy = xr+wr/2.0, yr+hr/2.0
        cp = os.path.join(od, "results.csv")
        with open(cp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# Benchmark", ts])
            w.writerow(["# ROI", f"x={xr} y={yr} w={wr} h={hr}", f"cx={rcx:.1f}", f"cy={rcy:.1f}"])
            w.writerow(["# Initial pose", f"yaw={iy:.3f}", f"pitch={ip:.3f}"])
            w.writerow(["# Target zoom", f"x{self.current_target_zoom:.1f}"]); w.writerow([])
            w.writerow(["method","zoom","label","ex_px","ey_px","err_px","duration_s"])
            for mn, recs in self.benchmark_results.items():
                if not recs: w.writerow([mn,"","FAILED","","","",""]); continue
                for r in recs:
                    w.writerow([mn, r["zoom"], r["label"],
                        f"{r['ex']:.2f}" if r["ex"] is not None else "NA",
                        f"{r['ey']:.2f}" if r["ey"] is not None else "NA",
                        f"{r['err_px']:.2f}" if r["err_px"] is not None else "NA",
                        f"{r['duration_s']:.4f}" if r.get("duration_s") is not None else "NA"])
        print(f"  CSV saved: {cp}")

        labels=[]; de=[]; dx_=[]; dy_=[]; mt=[]
        cols = plt.cm.tab10(np.linspace(0,1,len(METHOD_NAMES)))
        for mn in METHOD_NAMES:
            recs = self.benchmark_results.get(mn, [])
            v = [r for r in recs if r["err_px"] is not None and r["zoom"]>1.0]
            labels.append(mn); de.append([r["err_px"] for r in v])
            dx_.append([r["ex"] for r in v]); dy_.append([r["ey"] for r in v])
            ds = [r["duration_s"] for r in recs if r.get("duration_s") is not None and r["zoom"]>1.0]
            mt.append(np.mean(ds) if ds else 0.0)

        ttl = f"Benchmark {ts}\nROI {wr}x{hr}px @ ({rcx:.0f},{rcy:.0f})  zoom x{self.current_target_zoom:.1f}"
        fig,axes = plt.subplots(1,3,figsize=(16,7)); fig.suptitle(ttl, fontsize=11, y=1.02); fig.subplots_adjust(top=0.88)
        for ax,data,t,yl in [(axes[0],de,"Euclidean error","err (px)"),(axes[1],dx_,"X error","ex (px)"),(axes[2],dy_,"Y error","ey (px)")]:
            for i,(d,c) in enumerate(zip(data,cols)):
                if not d: continue
                bp=ax.boxplot([d],positions=[i+1],patch_artist=True,notch=False,medianprops=dict(color="black",linewidth=2),widths=0.6)
                bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.6)
            ax.set_xticks(range(1,len(labels)+1)); ax.set_xticklabels(labels,rotation=30,ha="right")
            ax.set_title(t); ax.set_ylabel(yl); ax.set_xlabel("Method"); ax.grid(True,axis="y",alpha=0.3)
            if t=="Euclidean error": ax.axhline(MICRO_CORRECT_PX,color="red",linestyle=":",alpha=0.6,label=f"micro thr ({MICRO_CORRECT_PX}px)"); ax.legend(fontsize=8)
            else: ax.axhline(0,color="gray",linestyle="-",alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(od,"comparison.png"),dpi=150,bbox_inches="tight"); plt.close()

        fig2,ax2=plt.subplots(figsize=(8,5)); fig2.suptitle(f"Execution time - {ttl}",fontsize=11,y=1.02); fig2.subplots_adjust(top=0.85)
        bars=ax2.bar(range(len(METHOD_NAMES)),mt,color=cols,alpha=0.7)
        ax2.set_xticks(range(len(METHOD_NAMES))); ax2.set_xticklabels(METHOD_NAMES,rotation=30,ha="right")
        ax2.set_title("Mean match time per step"); ax2.set_ylabel("Time (s)"); ax2.set_xlabel("Method"); ax2.grid(True,axis="y",alpha=0.3)
        for b,v in zip(bars,mt):
            if v>0: ax2.text(b.get_x()+b.get_width()/2,v+0.001,f"{v:.3f}s",ha="center",va="bottom",fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(od,"execution_time.png"),dpi=150,bbox_inches="tight"); plt.close()
        print(f"  Output dir: {od}")

    # ==========================================
    # MAIN LOOP — renders both windows
    # ==========================================
    def _main_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            frame = self._get_frame()
            if frame is not None:
                vis = frame.copy(); self._draw_overlay(vis)
                cv2.imshow("ORB Correct Once", vis)
                if not self.is_busy:
                    if self.template_saved: self.flowchart.set_phase(PHASE_PRESS_Z)
                    elif self.roi_selected: self.flowchart.set_phase(PHASE_PRESS_C)
                    else: self.flowchart.set_phase(PHASE_ROI_CHECK)
            else:
                self.flowchart.set_phase(PHASE_WAIT_CAMERA)

            cv2.imshow("Flowchart", self.flowchart.render())

            key = cv2.waitKeyEx(1)
            if key == -1: rate.sleep(); continue
            ck = key & 0xFF

            if   ck in (ord('w'),ord('W')): self._move_manual(0,-1)
            elif ck in (ord('s'),ord('S')): self._move_manual(0,1)
            elif ck in (ord('a'),ord('A')): self._move_manual(1,0)
            elif ck in (ord('d'),ord('D')): self._move_manual(-1,0)
            elif ck == 27: break
            elif ck == ord('r') and not self.is_busy:
                self.roi = None; self.roi_selected = self.template_saved = False
                self.template_gray = None; self.last_target_px = self.last_error_px = None
                self._set_zoom(1.0); self.current_zoom = 1.0; self.status = "Reset. Ready."
                self.flowchart.reset(); self.flowchart.set_phase(PHASE_WAIT_CAMERA)
            elif ck == ord('c') and self.roi_selected and not self.is_busy:
                self.flowchart.set_phase(PHASE_SAVE_TEMPLATE)
                frame = self._get_frame()
                if frame is not None and self.roi is not None:
                    x,y,w,h = self.roi
                    self.template_gray = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
                    self.template_saved = True
            elif ck == ord('z') and self.template_saved and not self.is_busy:
                self.is_busy = True; self.current_target_zoom = self.optimal_zoom
                self.last_target_px = self.last_error_px = None
                self.flowchart.set_phase(PHASE_PRESS_Z)
                threading.Thread(target=self._zoom_and_correct, daemon=True).start()
            elif ck in (ord('0'),ord('1'),ord('2'),ord('3')):
                self.cursor_sel = int(chr(ck))
            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try: OrbCorrectOnce()
    except rospy.ROSInterruptException: pass
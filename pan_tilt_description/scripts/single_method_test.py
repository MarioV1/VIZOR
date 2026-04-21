#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# single_method_test.py — Live PTZ zoom corrector, single method at a time
# Keys: m=cycle method  z=correct  r=reset  wasd=jog  +/-=step  1/2/3=cursor  ESC=quit

import rospy, cv2, numpy as np, os, math, subprocess, threading, time, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pan_tilt_msgs.msg import PanTiltCmdDeg
from pan_tilt_msgs.msg import PanTiltStatus
import rospkg

# ── Constants ─────────────────────────────────────────────────────────────────

IMG_W, IMG_H     = 1280, 720
CURSORS          = {1:(640,360), 2:(643,415), 3:(643,387)}
CURSOR_LABELS    = {1:"Centro imagen", 2:"Centro camara", 3:"Punto extra"}
CURSOR_COLORS    = {1:(0,255,0), 2:(255,0,0), 3:(255,0,255)}
ZOOM_FACTORES    = {
    1.0:(1.00,1.00),  2.0:(1.12,1.10),  3.0:(1.28,1.25),  4.0:(1.39,1.33),
    5.0:(1.61,1.53),  6.0:(1.74,1.67),  7.0:(1.98,1.89),  8.0:(2.19,2.10),
    9.0:(2.59,2.43),  10.0:(3.09,2.84), 11.0:(3.64,3.25), 12.0:(4.37,3.79),
    13.0:(6.29,5.15), 14.0:(7.95,6.48), 15.0:(9.89,7.95), 16.0:(12.16,9.35),
    17.0:(14.77,10.86),18.0:(16.82,12.10),19.0:(18.23,12.85),20.0:(18.56,13.00),
}
# ── Camera sensor parameters (replaces ZOOM_FOVS lookup table) ────────────────
# FoV equations: Z = K / tan(θ/2)  →  θ = 2·arctan(K/Z)
# K_H = sensor_width  / (2·f0) = 5.4 / (2·4.33)
# K_V = sensor_height / (2·f0) = 4.0 / (2·4.33)
SENSOR_W_MM      = 5.4
SENSOR_H_MM      = 4.0
F0_MM            = 4.33   # focal length at zoom = 1×
_KH              = SENSOR_W_MM / (2.0 * F0_MM)   # 0.6236
_KV              = SENSOR_H_MM / (2.0 * F0_MM)   # 0.4619
MIN_INLIERS      = 6
STABILISE_S      = 1.5
MICRO_CORRECT_PX  = 15
MAX_VALID_ERROR_PX = 200   # discard match if error exceeds this (likely false match)
EPICENTER_SIZE    = 450
PRE_CENTRE_PX    = 40
LOWE_RATIO       = 0.75
JOG_STEPS        = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
METHOD_NAMES     = ["ORB","SIFT","AKAZE","BRISK","OptFlow","NCC"]
MAX_SAFE_JUMP    = 4.0   # max zoom units per step

# ── Interpolated zoom factor table (0.01 step) ────────────────────────────────
# Only ZOOM_FACTORES needs interpolation; FoV is now computed analytically.

def _interp_zoom_factors():
    levels = sorted(ZOOM_FACTORES.keys())
    fac = {}
    for i in range(len(levels) - 1):
        z0, z1 = levels[i], levels[i+1]
        fx0, fy0 = ZOOM_FACTORES[z0]; fx1, fy1 = ZOOM_FACTORES[z1]
        steps = int(round((z1 - z0) / 0.01))
        for s in range(steps):
            t = s / steps
            z = round(z0 + t * (z1 - z0), 2)
            fac[z] = (fx0 + t*(fx1-fx0), fy0 + t*(fy1-fy0))
    fac[levels[-1]] = ZOOM_FACTORES[levels[-1]]
    return fac

ZOOM_FACTORES_DENSE = _interp_zoom_factors()

def get_fov(zoom_level):
    """Return (fov_h, fov_v) in degrees for any zoom level using the lens equations:
       θ_H = 2·arctan(K_H / Z),  θ_V = 2·arctan(K_V / Z)
    """
    z = max(zoom_level, 0.01)   # guard against division by zero
    fov_h = math.degrees(2.0 * math.atan(_KH / z))
    fov_v = math.degrees(2.0 * math.atan(_KV / z))
    return fov_h, fov_v

# ── Matchers ──────────────────────────────────────────────────────────────────

def texture_score(g):
    return cv2.Laplacian(g, cv2.CV_64F).var()

def _match_descriptor(det, norm, tg, sg, tx, ty, lw, mi):
    kp1, d1 = det.detectAndCompute(tg, None)
    kp2, d2 = det.detectAndCompute(sg, None)
    if d1 is None or d2 is None or len(kp1) < mi or len(kp2) < mi: return None
    ms = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    if len(ms) < mi: return None
    good = [m for m, n in ms if m.distance < lw * n.distance]
    if len(good) < mi: return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None: return None
    n = int(mask.sum()) if mask is not None else 0
    if n < mi: return None
    p = cv2.perspectiveTransform(np.float32([[tx, ty]]).reshape(-1, 1, 2), H)
    return float(p[0,0,0]), float(p[0,0,1]), n, H, mask, kp1, kp2, good

def _find_orb(tg, sg, tx, ty, ox=0, oy=0):
    for nf, lw, mi in [(2000, LOWE_RATIO, MIN_INLIERS), (4000, 0.85, max(4, MIN_INLIERS-2))]:
        r = _match_descriptor(cv2.ORB_create(nfeatures=nf), cv2.NORM_HAMMING, tg, sg, tx, ty, lw, mi)
        if r: return r
    return None

def _find_sift(tg, sg, tx, ty, ox=0, oy=0):
    for nf, lw, mi in [(2000, LOWE_RATIO, MIN_INLIERS), (4000, 0.85, max(4, MIN_INLIERS-2))]:
        r = _match_descriptor(cv2.SIFT_create(nfeatures=nf), cv2.NORM_L2, tg, sg, tx, ty, lw, mi)
        if r: return r
    return None

def _find_akaze(tg, sg, tx, ty, ox=0, oy=0):
    for thresh, lw, mi in [(0.001, LOWE_RATIO, MIN_INLIERS), (0.0005, 0.85, max(4, MIN_INLIERS-2))]:
        r = _match_descriptor(cv2.AKAZE_create(threshold=thresh), cv2.NORM_HAMMING, tg, sg, tx, ty, lw, mi)
        if r: return r
    return None

def _find_brisk(tg, sg, tx, ty, ox=0, oy=0):
    for thresh, lw, mi in [(30, LOWE_RATIO, MIN_INLIERS), (20, 0.85, max(4, MIN_INLIERS-2))]:
        r = _match_descriptor(cv2.BRISK_create(thresh=thresh), cv2.NORM_HAMMING, tg, sg, tx, ty, lw, mi)
        if r: return r
    return None

_LK = dict(winSize=(21,21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def _find_optflow(tg, sg, tx, ty, ox=0, oy=0):
    pts = cv2.goodFeaturesToTrack(tg, maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
    if pts is None or len(pts) < MIN_INLIERS: return None
    th, tw = tg.shape
    sh, sw = sg.shape
    sg_crop = sg[oy:oy+th, ox:ox+tw]
    if sg_crop.shape[0] != th or sg_crop.shape[1] != tw:
        sg_crop = cv2.resize(sg[max(0,oy):min(sh,oy+th), max(0,ox):min(sw,ox+tw)], (tw, th))
    tracked, st, _ = cv2.calcOpticalFlowPyrLK(tg, sg_crop, pts, None, **_LK)
    if tracked is None: return None
    s = st.ravel()
    src, dst = pts[s==1], tracked[s==1]
    if len(src) < MIN_INLIERS: return None
    A, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None: return None
    n = int(mask.sum()) if mask is not None else len(src)
    if n < MIN_INLIERS: return None
    H = np.vstack([A, [0, 0, 1]]).astype(np.float32)
    p = cv2.perspectiveTransform(np.float32([[tx, ty]]).reshape(-1, 1, 2), H)
    return float(p[0,0,0])+ox, float(p[0,0,1])+oy, n, H, mask, [], [], []

_NCC_PATCH, _NCC_SEARCH = 31, 80

def _find_ncc(tg, sg, tx, ty, ox=0, oy=0):
    corners = cv2.goodFeaturesToTrack(tg, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)
    if corners is None or len(corners) < MIN_INLIERS: return None
    th, tw = tg.shape
    sh, sw = sg.shape
    sp, dp = [], []
    for c in corners:
        cx_t, cy_t = int(c[0,0]), int(c[0,1])
        patch = tg[max(0,cy_t-_NCC_PATCH):min(th,cy_t+_NCC_PATCH+1),
                   max(0,cx_t-_NCC_PATCH):min(tw,cx_t+_NCC_PATCH+1)]
        if patch.shape[0] < 5 or patch.shape[1] < 5: continue
        fx, fy = cx_t+ox, cy_t+oy
        x1s = max(0, fx-_NCC_SEARCH); y1s = max(0, fy-_NCC_SEARCH)
        region = sg[y1s:min(sh,fy+_NCC_SEARCH+patch.shape[0]),
                    x1s:min(sw,fx+_NCC_SEARCH+patch.shape[1])]
        if region.shape[0] < patch.shape[0] or region.shape[1] < patch.shape[1]: continue
        _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED))
        if mx < 0.5: continue
        sp.append([cx_t, cy_t])
        dp.append([x1s+loc[0]+patch.shape[1]//2-ox, y1s+loc[1]+patch.shape[0]//2-oy])
    if len(sp) < MIN_INLIERS: return None
    A, mask = cv2.estimateAffinePartial2D(
        np.float32(sp).reshape(-1,1,2), np.float32(dp).reshape(-1,1,2),
        method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None: return None
    n = int(mask.sum()) if mask is not None else len(sp)
    if n < MIN_INLIERS: return None
    H = np.vstack([A, [0, 0, 1]]).astype(np.float32)
    p = cv2.perspectiveTransform(np.float32([[tx, ty]]).reshape(-1, 1, 2), H)
    return float(p[0,0,0])+ox, float(p[0,0,1])+oy, n, H, mask, [], [], []

_FINDERS = [_find_orb, _find_sift, _find_akaze, _find_brisk, _find_optflow, _find_ncc]

def find_template_point(tg, sg, tx, ty, method_idx, ox=0, oy=0):
    return _FINDERS[method_idx](tg, sg, tx, ty, ox, oy)

# ── Main node ─────────────────────────────────────────────────────────────────

class SingleMethodTest:
    def __init__(self):
        rospy.init_node("single_method_test", anonymous=True)
        self.bridge     = CvBridge()
        self.image      = None
        self.image_lock = threading.Lock()

        self.drawing   = False
        self.start_x   = self.start_y = -1
        self.current_x = self.current_y = -1

        self.roi          = None
        self.roi_selected = False
        self.optimal_zoom = 1.0
        self.target_zoom  = 1.0
        self.cursor_sel   = 1
        self.is_busy      = False

        self.last_target_px = None
        self.last_error_px  = None
        self.step_schedule  = []
        self.actual_zoom    = 1.0
        self.current_step   = 0
        self.roi_texture    = None
        self.live_roi       = None
        self.method_idx     = 0

        self.jog_step     = 1.0

        self.current_yaw   = 0.0
        self.current_pitch = 0.0

        self.PAN_MIN_DEG  = -60;  self.PAN_MAX_DEG  = 60
        self.TILT_MIN_DEG = -60;  self.TILT_MAX_DEG = 60
        self.status = "Draw ROI then press z | m=method"

        pkg_path = rospkg.RosPack().get_path("pan_tilt_description")
        self.image_dir  = os.path.join(pkg_path, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        self._video_recording = False
        self._vwriter         = None
        self._video_thread    = None

        self.pub_cmd = rospy.Publisher("/pan_tilt_cmd_deg", PanTiltCmdDeg, queue_size=1)
        rospy.Subscriber("/datavideo/video", Image, self._img_cb)
        rospy.Subscriber("/pan_tilt_status", PanTiltStatus, self._status_cb)
        threading.Thread(target=self._poll_zoom, daemon=True).start()

        cv2.namedWindow("Single Method Test")
        cv2.setMouseCallback("Single Method Test", self._mouse_cb)
        print("\nsingle_method_test started")
        print(f"   Methods: {METHOD_NAMES}")
        print("   m=cycle  z=correct  r=reset  wasd=jog  +/-=step  1/2/3=cursor  ESC=quit\n")
        self._main_loop()

    # ── ROS ───────────────────────────────────────────────────────────────────

    def _img_cb(self, msg):
        try:
            with self.image_lock:
                self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"Image error: {e}")

    def _status_cb(self, msg):
        self.current_yaw   = msg.yaw_now
        self.current_pitch = msg.pitch_now

    def _get_frame(self):
        with self.image_lock:
            return self.image.copy() if self.image is not None else None

    def _poll_zoom(self):
        """Fallback zoom poller in case status topic doesn't carry zoom_now."""
        while not rospy.is_shutdown():
            try:
                out = subprocess.getoutput("rostopic echo -n 1 /pan_tilt_status")
                for line in out.splitlines():
                    if "zoom_now:" in line:
                        self.actual_zoom = float(line.split("zoom_now:")[1].strip()); break
            except: pass
            time.sleep(1.0)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param):
        if self.is_busy: return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.current_x, self.current_y = x, y
            self.roi_selected = False
            self.last_target_px = self.last_error_px = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x, self.current_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x0, y0 = min(self.start_x, x), min(self.start_y, y)
            w,  h  = abs(x - self.start_x), abs(y - self.start_y)
            if w > 10 and h > 10:
                self.roi           = (x0, y0, w, h)
                self.roi_selected  = True
                self.optimal_zoom  = self._best_zoom(w, h)
                roi_cx, roi_cy     = x0 + w/2, y0 + h/2
                self.step_schedule = self._build_steps(self.optimal_zoom)
                self.current_step  = 0
                tex_warn = ""
                fn = self._get_frame()
                if fn is not None:
                    tscore = texture_score(cv2.cvtColor(fn[y0:y0+h, x0:x0+w], cv2.COLOR_BGR2GRAY))
                    self.roi_texture = tscore
                    if tscore < 100: tex_warn = " WARNING LOW TEXTURE"
                self.status = (f"ROI {w}x{h}px zoom x{self.optimal_zoom} "
                               f"steps={self.step_schedule} | z=correct{tex_warn}")

    # ── PTZ ───────────────────────────────────────────────────────────────────

    def _best_zoom(self, w, h):
        for zoom in sorted(ZOOM_FACTORES_DENSE.keys(), reverse=True):
            fx, fy = ZOOM_FACTORES_DENSE[zoom]
            if w*fx <= IMG_W*0.9 and h*fy <= IMG_H*0.9: return zoom
        return 1.0

    def _set_zoom(self, level):
        subprocess.call(f'rosservice call /set_zoom "level: {int(round(level))}"', shell=True)

    def _send_cmd(self, yaw, pitch, speed=20):
        yaw   = max(self.PAN_MIN_DEG,  min(self.PAN_MAX_DEG,  round(yaw,   3)))
        pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, round(pitch, 3)))
        cmd = PanTiltCmdDeg()
        cmd.yaw, cmd.pitch, cmd.speed = yaw, pitch, speed
        self.pub_cmd.publish(cmd)

    def _correct(self, ex, ey, zoom_level, label=""):
        fov_h, fov_v = get_fov(zoom_level)
        dyaw   = -ex / (IMG_W / fov_h)
        dpitch =  ey / (IMG_H / fov_v)

        if abs(dyaw) < 0.01 and abs(dpitch) < 0.01:
            print(f"   {label}negligible correction ({ex:+.1f},{ey:+.1f})px — skipped")
            return

        speed = 25 if math.hypot(dyaw, dpitch) > 5.0 else 15
        new_yaw   = max(self.PAN_MIN_DEG,  min(self.PAN_MAX_DEG,  self.current_yaw   + dyaw))
        new_pitch = max(self.TILT_MIN_DEG, min(self.TILT_MAX_DEG, self.current_pitch + dpitch))
        self._send_cmd(new_yaw, new_pitch, speed=speed)
        print(f"   {label}dy={dyaw:+.3f} dp={dpitch:+.3f} "
              f"({self.current_yaw:.2f},{self.current_pitch:.2f})"
              f"->({new_yaw:.3f},{new_pitch:.3f}) speed={speed}")

    def _jog(self, dyaw, dpitch):
        self._send_cmd(self.current_yaw + dyaw, self.current_pitch + dpitch)
        print(f"   jog dy={dyaw:+.1f} dp={dpitch:+.1f}  step={self.jog_step}°")

    def _build_steps(self, tz, roi_cx=None, roi_cy=None):
        """Equal steps with max jump of MAX_SAFE_JUMP zoom units."""
        total = tz - 1.0
        if total <= 0: return [1.0, tz]
        n = math.ceil(total / MAX_SAFE_JUMP)
        step = total / n
        steps = [1.0] + [round(1.0 + step*k, 2) for k in range(1, n+1)]
        steps[-1] = tz  # ensure exact target
        return steps

    def _update_live_roi(self, cx, cy, zoom_level):
        if self.roi is None: return
        _, _, wr, hr = self.roi
        z = round(zoom_level, 2)
        fx, fy = ZOOM_FACTORES_DENSE.get(z, ZOOM_FACTORES.get(round(zoom_level), (1.0, 1.0)))
        self.live_roi = (cx, cy, wr*fx, hr*fy)

    # ── Core correction thread ────────────────────────────────────────────────

    def _zoom_and_correct(self):
        midx = self.method_idx
        try:
            tz = self.target_zoom
            cx_c, cy_c = CURSORS[self.cursor_sel]
            xr, yr, wr, hr = self.roi
            roi_cx, roi_cy = xr + wr/2.0, yr + hr/2.0

            # Optical limiter
            msz = 1.0
            for z in sorted(ZOOM_FACTORES_DENSE.keys()):
                fx, fy = ZOOM_FACTORES_DENSE[z]
                if wr*fx > IMG_W*0.95 or hr*fy > IMG_H*0.95: break
                msz = z
            if tz > msz:
                self.status = f"Optical limit x{msz}"; tz = msz

            steps = self._build_steps(tz)
            self.step_schedule = steps

            # Create output folder and start recording
            ts      = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
            mname   = METHOD_NAMES[midx]
            out_dir = os.path.join(self.script_dir, f"[{mname}]_{ts}")
            os.makedirs(out_dir, exist_ok=True)
            self._start_video_recording(os.path.join(out_dir, "test.avi"))

            self.status = "Returning to x1..."
            self._set_zoom(1); time.sleep(0.8)
            # Wait until zoom x1 is confirmed before showing ROI
            for _ in range(10):
                if abs(self.actual_zoom - 1.0) < 0.1:
                    break
                time.sleep(0.3)
            self.roi_selected = True

            f0 = self._get_frame()
            if f0 is None: self.status = "No frame at x1"; return

            # Expanded template
            cw = max(int(wr*1.5), 250); ch = max(int(hr*1.5), 250)
            x1t = max(0, int(roi_cx - cw/2)); y1t = max(0, int(roi_cy - ch/2))
            tg  = cv2.cvtColor(f0[y1t:min(IMG_H,y1t+ch), x1t:min(IMG_W,x1t+cw)], cv2.COLOR_BGR2GRAY)
            lcx, lcy     = roi_cx - x1t, roi_cy - y1t
            tg_ox, tg_oy = x1t, y1t
            self._update_live_roi(roi_cx, roi_cy, 1.0)

            # Pre-centre
            ex0, ey0 = roi_cx - cx_c, roi_cy - cy_c
            if abs(ex0) > PRE_CENTRE_PX or abs(ey0) > PRE_CENTRE_PX:
                self.status = "Pre-centering..."
                self._correct(ex0, ey0, 1.0, "pre-centre ")
                time.sleep(2.0)
                f1 = self._get_frame()
                if f1 is not None:
                    r0 = find_template_point(tg, cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY),
                                             lcx, lcy, midx, tg_ox, tg_oy)
                    if r0: ccx, ccy = r0[0], r0[1]; self._update_live_roi(ccx, ccy, 1.0)
                    else:  ccx, ccy = float(cx_c), float(cy_c)
                    fb = f1
                else:
                    ccx, ccy = float(cx_c), float(cy_c); fb = f0
            else:
                ccx, ccy = roi_cx, roi_cy; fb = f0

            # Matryoshka relay
            sb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
            x1 = max(0, int(ccx - EPICENTER_SIZE/2)); y1 = max(0, int(ccy - EPICENTER_SIZE/2))
            x2 = min(IMG_W, int(ccx + EPICENTER_SIZE/2)); y2 = min(IMG_H, int(ccy + EPICENTER_SIZE/2))
            tg = sb[y1:y2, x1:x2]; lcx, lcy = ccx-x1, ccy-y1; tg_ox, tg_oy = x1, y1

            Hf = None; nif = 0; sf = fb
            records = []

            for si, sz in enumerate(steps[1:]):
                self.current_step = si + 1
                self.status = f"[{mname}] Step {si+1}/{len(steps)-1}: x{sz}..."
                self._set_zoom(sz); time.sleep(STABILISE_S)

                sf = self._get_frame()
                if sf is None: self.status = f"No frame x{sz}"; return
                sg = cv2.cvtColor(sf, cv2.COLOR_BGR2GRAY)

                t0  = time.time()
                res = find_template_point(tg, sg, lcx, lcy, midx, tg_ox, tg_oy)
                dur = time.time() - t0

                if res is None:
                    self.status = f"Match failed x{sz}"
                    records.append(dict(zoom=float(sz), label=f"step{si+1}_failed",
                                        ex=None, ey=None, err_px=None, duration_s=dur))
                    return

                cxf, cyf, ni, H, _, _, _, _ = res
                Hf = H; nif = ni
                self.last_target_px = (cxf, cyf)
                self._update_live_roi(cxf, cyf, sz)
                ex, ey = cxf - cx_c, cyf - cy_c
                self.last_error_px = (ex, ey)
                err_px = math.hypot(ex, ey)

                # Sanity check — discard implausibly large errors as false matches
                if err_px > MAX_VALID_ERROR_PX:
                    self.status = f"False match x{sz} err={err_px:.0f}px — skipping correction"
                    print(f"   [{mname}] x{sz} FALSE MATCH err={err_px:.0f}px — skipping")
                    records.append(dict(zoom=float(sz), label=f"step{si+1}_false_match",
                                        ex=ex, ey=ey, err_px=err_px, duration_s=dur))
                    ccx, ccy = float(cx_c), float(cy_c)
                    continue

                records.append(dict(zoom=float(sz), label=f"step{si+1}",
                                    ex=ex, ey=ey, err_px=err_px, duration_s=dur))
                print(f"   [{mname}] x{sz} err=({ex:+.1f},{ey:+.1f})px |{err_px:.1f}|px t={dur:.3f}s")

                if abs(ex) > MICRO_CORRECT_PX or abs(ey) > MICRO_CORRECT_PX:
                    self.status = f"Micro-correct x{sz}..."
                    self._correct(ex, ey, sz, "micro ")
                    time.sleep(2.0)
                    fz2 = self._get_frame()
                    if fz2 is not None:
                        sg2 = cv2.cvtColor(fz2, cv2.COLOR_BGR2GRAY)
                        t0  = time.time()
                        r2  = find_template_point(tg, sg2, lcx, lcy, midx, tg_ox, tg_oy)
                        dur2 = time.time() - t0
                        if r2:
                            cxf, cyf = r2[0], r2[1]; Hf, nif = r2[3], r2[2]
                            sg = sg2; sf = fz2
                            self._update_live_roi(cxf, cyf, sz)
                            ex2, ey2 = cxf - cx_c, cyf - cy_c
                            records.append(dict(zoom=float(sz), label=f"step{si+1}_after_micro",
                                                ex=ex2, ey=ey2, err_px=math.hypot(ex2,ey2),
                                                duration_s=dur2))
                        else:
                            cxf, cyf = float(cx_c), float(cy_c)

                ccx, ccy = cxf, cyf
                if si < len(steps) - 2:
                    # Always grab a fresh frame for the relay crop
                    sf_relay = self._get_frame()
                    if sf_relay is not None:
                        sg_relay = cv2.cvtColor(sf_relay, cv2.COLOR_BGR2GRAY)
                    else:
                        sg_relay = sg
                    x1 = max(0, int(ccx - EPICENTER_SIZE/2)); y1 = max(0, int(ccy - EPICENTER_SIZE/2))
                    x2 = min(IMG_W, int(ccx + EPICENTER_SIZE/2)); y2 = min(IMG_H, int(ccy + EPICENTER_SIZE/2))
                    tg = sg_relay[y1:y2, x1:x2]; lcx, lcy = ccx-x1, ccy-y1; tg_ox, tg_oy = x1, y1

            exf, eyf = ccx - cx_c, ccy - cy_c
            self.last_error_px = (exf, eyf)
            records.append(dict(zoom=float(tz), label="final_before",
                                ex=exf, ey=eyf, err_px=math.hypot(exf,eyf), duration_s=None))
            self._correct(exf, eyf, tz, "final ")
            self.status = f"Done err=({exf:+.0f},{eyf:+.0f})px in={nif} [{mname}]"
            print(self.status)

            if Hf is not None:
                self._save_debug(sf, tg, Hf, ccx, ccy, cx_c, cy_c, exf, eyf, midx)

            self._stop_video_recording()
            self._save_results(records, midx, out_dir, ts)

        except Exception as e:
            self.status = f"Error: {e}"; rospy.logerr(str(e))
            import traceback; traceback.print_exc()
        finally:
            if self._video_recording:
                self._stop_video_recording()
            self.last_target_px = None
            self.last_error_px  = None
            self.live_roi       = None
            self.roi_selected   = False
            self.is_busy = False; self.current_step = 0

    # ── Debug image ───────────────────────────────────────────────────────────

    def _save_debug(self, fz, tg, H, cxf, cyf, cx_c, cy_c, ex, ey, midx):
        dbg = fz.copy()
        th, tw = tg.shape[:2]
        proj = cv2.perspectiveTransform(
            np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1,1,2), H)
        cv2.polylines(dbg, [np.int32(proj)], True, (0,165,255), 2)
        cv2.drawMarker(dbg, (int(cxf), int(cyf)), (0,165,255), cv2.MARKER_CROSS, 16, 1)
        cv2.drawMarker(dbg, (cx_c, cy_c), CURSOR_COLORS[self.cursor_sel], cv2.MARKER_CROSS, 20, 2)
        cv2.arrowedLine(dbg, (int(cxf), int(cyf)), (cx_c, cy_c), (255,255,0), 2, tipLength=0.15)
        cv2.putText(dbg, f"err({ex:+.0f},{ey:+.0f})px", (cx_c+8, cy_c+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.imwrite(os.path.join(self.image_dir,
                    f"single_{METHOD_NAMES[midx].lower()}_debug.jpg"), dbg)

    # ── Overlay ───────────────────────────────────────────────────────────────

    # ── Video recording ───────────────────────────────────────────────────────

    def _start_video_recording(self, video_path):
        pass  # video recording disabled — codec issues on this system

    def _stop_video_recording(self):
        pass

    def _video_record_loop(self):
        pass

    # ── Results output ────────────────────────────────────────────────────────

    def _save_results(self, records, midx, out_dir, ts):
        mname = METHOD_NAMES[midx]

        # CSV
        csv_path = os.path.join(out_dir, "results.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# Test", ts])
            w.writerow(["# Method", mname])
            w.writerow(["# Target zoom", f"x{self.target_zoom}"])
            w.writerow([])
            w.writerow(["zoom","label","ex_px","ey_px","err_px","duration_s"])
            for r in records:
                w.writerow([
                    r["zoom"], r["label"],
                    f"{r['ex']:.2f}"         if r["ex"]       is not None else "NA",
                    f"{r['ey']:.2f}"         if r["ey"]       is not None else "NA",
                    f"{r['err_px']:.2f}"     if r["err_px"]   is not None else "NA",
                    f"{r['duration_s']:.4f}" if r.get("duration_s") is not None else "NA",
                ])
        print(f"  CSV saved: {csv_path}")

        valid      = [r for r in records if r["err_px"] is not None and r["zoom"] > 1.0]
        data_err   = [r["err_px"] for r in valid]
        data_ex    = [r["ex"]     for r in valid]
        data_ey    = [r["ey"]     for r in valid]
        durations  = [r["duration_s"] for r in records
                      if r.get("duration_s") is not None and r["zoom"] > 1.0]
        mean_time  = float(np.mean(durations)) if durations else 0.0

        col = plt.cm.tab10(METHOD_NAMES.index(mname) / len(METHOD_NAMES))
        title = f"[{mname}] {ts}  zoom x{self.target_zoom}"

        # Figure 1: error boxplots
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        fig.suptitle(title, fontsize=11, y=1.02)
        fig.subplots_adjust(top=0.88)
        for ax, data, ttl, ylabel in [
            (axes[0], data_err, "Euclidean error", "err (px)"),
            (axes[1], data_ex,  "X error (ex)",    "ex (px)"),
            (axes[2], data_ey,  "Y error (ey)",    "ey (px)"),
        ]:
            if data:
                bp = ax.boxplot([data], positions=[1], patch_artist=True, notch=False,
                                medianprops=dict(color="black", linewidth=2), widths=0.5)
                bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.7)
            ax.set_xticks([1]); ax.set_xticklabels([mname])
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

        # Figure 2: execution time bar
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.suptitle(f"Execution time — {title}", fontsize=10, y=1.02)
        fig2.subplots_adjust(top=0.85)
        bar = ax2.bar([0], [mean_time], color=[col], alpha=0.7)
        ax2.set_xticks([0]); ax2.set_xticklabels([mname])
        ax2.set_title("Mean match time per step")
        ax2.set_ylabel("Time (s)"); ax2.set_xlabel("Method")
        ax2.grid(True, axis="y", alpha=0.3)
        if mean_time > 0:
            ax2.text(0, mean_time + 0.001, f"{mean_time:.3f}s",
                     ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        time_path = os.path.join(out_dir, "execution_time.png")
        plt.savefig(time_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Time plot saved: {time_path}")
        print(f"  Output dir: {out_dir}")

    def _draw_overlay(self, img):
        if self.drawing:
            cv2.rectangle(img, (self.start_x, self.start_y),
                          (self.current_x, self.current_y), (0,255,255), 1)

        if self.roi_selected and self.roi:
            x, y, w, h = self.roi
            cx_roi, cy_roi = x + w//2, y + h//2
            if self.live_roi is not None:
                lx, ly, lw, lh = self.live_roi
                rx, ry = int(lx - lw/2), int(ly - lh/2)
                cv2.rectangle(img, (rx, ry), (rx+int(lw), ry+int(lh)), (0,255,0), 2)
                cv2.putText(img, "ROI", (rx, ry-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, "ROI", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
            # Always draw ROI centre cross
            cv2.drawMarker(img, (cx_roi, cy_roi), (0,200,0), cv2.MARKER_CROSS, 14, 1)

        last_tgt = self.last_target_px
        last_err = self.last_error_px
        if last_tgt is not None:
            tx, ty = int(last_tgt[0]), int(last_tgt[1])
            cv2.drawMarker(img, (tx, ty), (0,165,255), cv2.MARKER_CROSS, 16, 1)
            cv2.putText(img, f"Target({tx},{ty})", (tx+6, ty-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1)

        if last_tgt is not None and last_err is not None:
            cx_c, cy_c = CURSORS[self.cursor_sel]
            tx, ty = int(last_tgt[0]), int(last_tgt[1])
            cv2.arrowedLine(img, (tx, ty), (cx_c, cy_c), (255,255,0), 2, tipLength=0.15)
            ex, ey = last_err
            cv2.putText(img, f"err({ex:+.0f},{ey:+.0f})px", (cx_c+8, cy_c+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        for k, (cx, cy) in CURSORS.items():
            if self.cursor_sel in (0, k):
                cv2.drawMarker(img, (cx, cy), CURSOR_COLORS[k], cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img, CURSOR_LABELS[k], (cx+10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CURSOR_COLORS[k], 1)

        cv2.putText(img, self.status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        bar_x = 10; bar_y = IMG_H - 35
        for i, name in enumerate(METHOD_NAMES):
            active = (i == self.method_idx)
            col = (0,255,255) if active else (120,120,120)
            lbl = f"[{name}]" if active else f" {name} "
            (tw2, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(img, lbl, (bar_x, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 2 if active else 1)
            bar_x += tw2 + 8

        info = [f"Zoom:x{self.actual_zoom:.1f}  [{METHOD_NAMES[self.method_idx]}]  jog:{self.jog_step}°"]
        if self.step_schedule:
            info.append(f"Steps:{self.step_schedule}")
        if self.is_busy and self.current_step > 0:
            info.append(f"Step {self.current_step}/{len(self.step_schedule)-1}")
        if self.roi_texture is not None:
            lbl = f"Tex:{self.roi_texture:.0f}"
            if self.roi_texture < 100: lbl += " LOW"
            info.append(lbl)
        for j, line in enumerate(info):
            cv2.putText(img, line, (10, 60+j*26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,80,255) if "LOW" in line else (0,220,255), 2)

        cv2.putText(img, "z=correct  r=reset  m=method  wasd=jog  +/-=step  1/2/3=cursor  ESC=quit",
                    (10, IMG_H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _main_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            frame = self._get_frame()
            if frame is not None:
                vis = frame.copy()
                self._draw_overlay(vis)
                cv2.imshow("Single Method Test", vis)

            key = cv2.waitKey(1) & 0xFF
            if   key == 27: break
            elif key == ord("m") and not self.is_busy:
                self.method_idx = (self.method_idx + 1) % len(METHOD_NAMES)
                print(f"Method -> {METHOD_NAMES[self.method_idx]}")
            elif key == ord("r") and not self.is_busy:
                self.roi = None; self.roi_selected = False
                self.last_target_px = self.last_error_px = None
                self.roi_texture = None; self.step_schedule = []; self.live_roi = None
                self._set_zoom(1)
                self.status = "Reset. Draw a new ROI."
            elif key == ord("z") and self.roi_selected and not self.is_busy:
                self.target_zoom = self.optimal_zoom
                self.is_busy = True
                self.last_target_px = self.last_error_px = None
                threading.Thread(target=self._zoom_and_correct, daemon=True).start()
                self.target_zoom = self.optimal_zoom
                self.is_busy = True
                self.last_target_px = self.last_error_px = None
                threading.Thread(target=self._zoom_and_correct, daemon=True).start()
            elif key == ord("w") and not self.is_busy: self._jog(0,  -self.jog_step)
            elif key == ord("s") and not self.is_busy: self._jog(0,  +self.jog_step)
            elif key == ord("a") and not self.is_busy: self._jog(+self.jog_step, 0)
            elif key == ord("d") and not self.is_busy: self._jog(-self.jog_step, 0)
            elif key in (ord("+"), ord("=")):
                ji = JOG_STEPS.index(self.jog_step) if self.jog_step in JOG_STEPS else 2
                self.jog_step = JOG_STEPS[min(ji+1, len(JOG_STEPS)-1)]
                print(f"Jog step -> {self.jog_step}°")
            elif key == ord("-"):
                ji = JOG_STEPS.index(self.jog_step) if self.jog_step in JOG_STEPS else 2
                self.jog_step = JOG_STEPS[max(ji-1, 0)]
                print(f"Jog step -> {self.jog_step}°")
            elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
                self.cursor_sel = int(chr(key))

            rate.sleep()

        cv2.destroyAllWindows()
        print("Bye")


if __name__ == '__main__':
    try:
        SingleMethodTest()
    except rospy.ROSInterruptException:
        pass
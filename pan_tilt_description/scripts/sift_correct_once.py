#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy, cv2, numpy as np, os, math, subprocess, threading, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pan_tilt_msgs.msg import PanTiltCmdDeg
import rospkg

IMG_W, IMG_H = 1280, 720
CURSORS      = {1:(640,360), 2:(643,415), 3:(643,387)}
CURSOR_LABELS= {1:"Centro imagen", 2:"Centro camara", 3:"Punto extra"}
CURSOR_COLORS= {1:(0,255,0), 2:(255,0,0), 3:(255,0,255)}
ZOOM_FACTORES= {
    1.0:(1.00,1.00),2.0:(1.12,1.10),3.0:(1.28,1.25),4.0:(1.39,1.33),
    5.0:(1.61,1.53),6.0:(1.74,1.67),7.0:(1.98,1.89),8.0:(2.19,2.10),
    9.0:(2.59,2.43),10.0:(3.09,2.84),11.0:(3.64,3.25),12.0:(4.37,3.79),
    13.0:(6.29,5.15),14.0:(7.95,6.48),15.0:(9.89,7.95),16.0:(12.16,9.35),
    17.0:(14.77,10.86),18.0:(16.82,12.10),19.0:(18.23,12.85),20.0:(18.56,13.00),
}
ZOOM_FOVS = {
    1.0:(63.7,35.84),2.0:(56.9,31.2),3.0:(50.7,27.3),4.0:(45.9,24.5),
    5.0:(40.5,21.6),6.0:(37.4,19.6),7.0:(32.2,17.2),8.0:(29.1,15.2),
    9.0:(25.3,13.0),10.0:(21.7,11.1),11.0:(18.3,9.3),12.0:(15.2,7.7),
    13.0:(10.0,6.2),14.0:(7.8,4.8),15.0:(6.2,3.6),16.0:(5.2,2.9),
    17.0:(4.1,2.3),18.0:(3.5,1.9),19.0:(2.9,1.7),20.0:(2.3,1.3),
}
MIN_INLIERS      = 6
STABILISE_S      = 1.5
SERVO_RES        = 1.0
MICRO_CORRECT_PX = 15
EPICENTER_SIZE   = 450
PRE_CENTRE_PX    = 40

LOWE_RATIO = 0.75
def texture_score(g): return cv2.Laplacian(g,cv2.CV_64F).var()
def find_template_point(tg,sg,tx,ty):
    for nf,lw,mi in [(2000,LOWE_RATIO,MIN_INLIERS),(4000,0.85,max(4,MIN_INLIERS-2))]:
        det=cv2.SIFT_create(nfeatures=nf)
        kp1,d1=det.detectAndCompute(tg,None); kp2,d2=det.detectAndCompute(sg,None)
        if d1 is None or d2 is None: continue
        if len(kp1)<mi or len(kp2)<mi: continue
        bf=cv2.BFMatcher(cv2.NORM_L2)
        ms=bf.knnMatch(d1,d2,k=2)
        if len(ms)<mi: continue
        good=[m for m,n in ms if m.distance<lw*n.distance]
        if len(good)<mi: continue
        src=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        if H is None: continue
        n=int(mask.sum()) if mask is not None else 0
        if n<mi: continue
        pt=np.float32([[tx,ty]]).reshape(-1,1,2)
        p=cv2.perspectiveTransform(pt,H)
        return float(p[0,0,0]),float(p[0,0,1]),n,H,mask,kp1,kp2,good
    return None


class SiftCorrectOnce:
    def __init__(self):
        rospy.init_node("sift_correct_once", anonymous=True)
        self.bridge=CvBridge(); self.image=None; self.image_lock=threading.Lock()
        self.drawing=False; self.start_x=self.start_y=-1
        self.current_x=self.current_y=-1
        self.roi=None; self.roi_selected=False
        self.template_gray=None; self.template_saved=False
        self.optimal_zoom=1.0; self.current_zoom=1.0; self.target_zoom=1.0
        self.cursor_sel=1; self.is_busy=False
        self.last_target_px=None; self.last_error_px=None; self.last_inliers=0
        self.step_schedule=[]; self.actual_zoom=1.0; self.current_step=0
        self.roi_texture=None; self.live_roi=None
        self.PAN_MIN_DEG=-60; self.PAN_MAX_DEG=60
        self.TILT_MIN_DEG=-60; self.TILT_MAX_DEG=60
        self.status="Draw a ROI around the target, then press z"
        rospack=rospkg.RosPack()
        pkg_path=rospack.get_path("pan_tilt_description")
        self.image_dir=os.path.join(pkg_path,"images")
        os.makedirs(self.image_dir,exist_ok=True)
        self.pub_cmd=rospy.Publisher("/pan_tilt_cmd_deg",PanTiltCmdDeg,queue_size=1)
        rospy.Subscriber("/datavideo/video",Image,self._img_cb)
        threading.Thread(target=self._poll_zoom,daemon=True).start()
        cv2.namedWindow("SIFT Correct Once")
        cv2.setMouseCallback("SIFT Correct Once",self._mouse_cb)
        print("\nsift_correct_once started [SIFT]")
        print("   Draw ROI -> z=correct  1/2/3 cursor  r reset  ESC quit\n")
        self._main_loop()

    def _img_cb(self,msg):
        try:
            frame=self.bridge.imgmsg_to_cv2(msg,"bgr8")
            with self.image_lock: self.image=frame
        except Exception as e: rospy.logwarn(f"Image error: {e}")

    def _get_frame(self):
        with self.image_lock:
            return self.image.copy() if self.image is not None else None

    def _poll_zoom(self):
        while not rospy.is_shutdown():
            try:
                out=subprocess.getoutput("rostopic echo -n 1 /pan_tilt_status")
                for line in out.splitlines():
                    if "zoom_now:" in line:
                        self.actual_zoom=float(line.split("zoom_now:")[1].strip()); break
            except: pass
            time.sleep(0.5)

    def _mouse_cb(self,event,x,y,flags,param):
        if self.is_busy: return
        if event==cv2.EVENT_LBUTTONDOWN:
            self.drawing=True; self.start_x,self.start_y=x,y
            self.current_x,self.current_y=x,y
            self.roi_selected=False; self.template_saved=False
            self.template_gray=None; self.last_target_px=self.last_error_px=None
        elif event==cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x,self.current_y=x,y
        elif event==cv2.EVENT_LBUTTONUP:
            self.drawing=False
            x0,y0=min(self.start_x,x),min(self.start_y,y)
            w,h=abs(x-self.start_x),abs(y-self.start_y)
            if w>10 and h>10:
                self.roi=(x0,y0,w,h); self.roi_selected=True
                self.optimal_zoom=self._best_zoom(w,h)
                roi_cx,roi_cy=x0+w/2,y0+h/2
                self.step_schedule=self._build_steps(self.optimal_zoom,roi_cx,roi_cy)
                self.current_step=0
                fn=self._get_frame(); tex_warn=""
                if fn is not None:
                    crop=cv2.cvtColor(fn[y0:y0+h,x0:x0+w],cv2.COLOR_BGR2GRAY)
                    tscore=texture_score(crop); self.roi_texture=tscore
                    if tscore<100: tex_warn=" WARNING LOW TEXTURE"
                self.status=(f"ROI {w}x{h}px zoom x{self.optimal_zoom} steps={self.step_schedule} z{tex_warn}")

    def _best_zoom(self,w,h):
        for zoom in sorted(ZOOM_FACTORES.keys(),reverse=True):
            fx,fy=ZOOM_FACTORES[zoom]
            if w*fx<=IMG_W*0.9 and h*fy<=IMG_H*0.9: return zoom
        return 1.0

    def _set_zoom(self,level):
        subprocess.call(f'rosservice call /set_zoom "level: {int(level)}"',shell=True)

    def _read_pose(self):
        out=subprocess.getoutput("rostopic echo -n 1 /pan_tilt_status")
        yaw=pitch=0.0
        for line in out.splitlines():
            if "yaw_now:" in line and "yaw_now_" not in line:
                try: yaw=float(line.split("yaw_now:")[1].strip())
                except: pass
            if "pitch_now:" in line and "pitch_now_" not in line:
                try: pitch=float(line.split("pitch_now:")[1].strip())
                except: pass
            if "zoom_now:" in line:
                try: self.actual_zoom=float(line.split("zoom_now:")[1].strip())
                except: pass
        return yaw,pitch

    def _send_cmd(self,yaw,pitch,speed=20):
        yaw=max(self.PAN_MIN_DEG,min(self.PAN_MAX_DEG,round(yaw,3)))
        pitch=max(self.TILT_MIN_DEG,min(self.TILT_MAX_DEG,round(pitch,3)))
        cmd=PanTiltCmdDeg(); cmd.yaw,cmd.pitch,cmd.speed=yaw,pitch,speed
        self.pub_cmd.publish(cmd)

    def _angular_correction(self,ex,ey,zoom_level):
        fov_h,fov_v=ZOOM_FOVS[zoom_level]
        dyaw=-(ex/(IMG_W/fov_h)); dpitch=(ey/(IMG_H/fov_v))
        dyaw=round(dyaw/SERVO_RES)*SERVO_RES
        dpitch=round(dpitch/SERVO_RES)*SERVO_RES
        return dyaw,dpitch

    def _correct(self,ex,ey,zoom_level,label=""):
        dyaw,dpitch=self._angular_correction(ex,ey,zoom_level)
        if dyaw==0.0 and dpitch==0.0:
            print(f"   {label}sub-resolution skipped ({ex:+.1f},{ey:+.1f})px"); return dyaw,dpitch
        yaw_now,pitch_now=self._read_pose()
        self._send_cmd(yaw_now+dyaw,pitch_now+dpitch)
        print(f"   {label}dy={dyaw:+.2f} dp={dpitch:+.2f} ({yaw_now:.2f},{pitch_now:.2f})->({yaw_now+dyaw:.2f},{pitch_now+dpitch:.2f})")
        return dyaw,dpitch

    def _build_steps(self,target_zoom,roi_cx,roi_cy):
        d=math.sqrt(IMG_W**2+IMG_H**2)
        off=math.sqrt((roi_cx-IMG_W/2)**2+(roi_cy-IMG_H/2)**2)
        diff=(off/d)*(ZOOM_FOVS[1.0][0]/ZOOM_FOVS[target_zoom][0])
        if diff<0.05: n=0
        elif diff<0.25: n=1
        elif diff<0.60: n=2
        else: n=3
        if n==0: return [1,int(target_zoom)]
        sz=target_zoom/(n+1)
        wps=[1]+[int(round(sz*k)) for k in range(1,n+1)]+[int(target_zoom)]
        return sorted(set(max(1,min(int(target_zoom),w)) for w in wps))

    def _update_live_roi(self,cx,cy,zoom_level):
        if self.roi is None: return
        _,_,wr,hr=self.roi
        fx,fy=ZOOM_FACTORES.get(zoom_level,(1.0,1.0))
        self.live_roi=(cx,cy,wr*fx,hr*fy)

    def _zoom_and_correct(self):
        try:
            tz=self.target_zoom
            cx_c,cy_c=CURSORS[self.cursor_sel]
            xr,yr,wr,hr=self.roi
            roi_cx,roi_cy=xr+wr/2.0,yr+hr/2.0
            msz=1.0
            for z in sorted(ZOOM_FACTORES.keys()):
                fx,fy=ZOOM_FACTORES[z]
                if wr*fx>IMG_W*0.95 or hr*fy>IMG_H*0.95: break
                msz=z
            if tz>msz: self.status=f"Optical limit x{msz}"; tz=msz
            steps=self._build_steps(tz,roi_cx,roi_cy); self.step_schedule=steps
            self.status="Returning to x1..."; self._set_zoom(1); time.sleep(0.8)
            f0=self._get_frame()
            if f0 is None: self.status="No frame at x1"; return
            cw=max(int(wr*1.5),250); ch=max(int(hr*1.5),250)
            x1t=max(0,int(roi_cx-cw/2)); y1t=max(0,int(roi_cy-ch/2))
            x2t=min(IMG_W,x1t+cw); y2t=min(IMG_H,y1t+ch)
            tg=cv2.cvtColor(f0[y1t:y2t,x1t:x2t],cv2.COLOR_BGR2GRAY)
            lcx,lcy=roi_cx-x1t,roi_cy-y1t
            self._update_live_roi(roi_cx,roi_cy,1.0)
            ex0,ey0=roi_cx-cx_c,roi_cy-cy_c
            if abs(ex0)>PRE_CENTRE_PX or abs(ey0)>PRE_CENTRE_PX:
                self.status="Pre-centering..."
                self._correct(ex0,ey0,1.0,"pre-centre ")
                time.sleep(2.0)
                f1=self._get_frame()
                if f1 is not None:
                    s1=cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
                    r0=find_template_point(tg,s1,lcx,lcy)
                    if r0 is not None:
                        ccx,ccy=r0[0],r0[1]; self._update_live_roi(ccx,ccy,1.0)
                    else: ccx,ccy=float(cx_c),float(cy_c)
                    fb=f1
                else: ccx,ccy=float(cx_c),float(cy_c); fb=f0
            else: ccx,ccy=roi_cx,roi_cy; fb=f0
            sb=cv2.cvtColor(fb,cv2.COLOR_BGR2GRAY)
            x1=max(0,int(ccx-EPICENTER_SIZE/2)); y1=max(0,int(ccy-EPICENTER_SIZE/2))
            x2=min(IMG_W,int(ccx+EPICENTER_SIZE/2)); y2=min(IMG_H,int(ccy+EPICENTER_SIZE/2))
            tg=sb[y1:y2,x1:x2]; lcx,lcy=ccx-x1,ccy-y1
            Hf=None; nif=0; sf=fb
            zsteps=steps[1:]
            for si,sz in enumerate(zsteps):
                self.current_step=si+1
                self.status=f"Step {si+1}/{len(zsteps)}: x{sz}..."
                self._set_zoom(sz); time.sleep(STABILISE_S)
                sf=self._get_frame()
                if sf is None: self.status=f"No frame x{sz}"; return
                sg=cv2.cvtColor(sf,cv2.COLOR_BGR2GRAY)
                self.status=f"Matching x{sz}..."
                res=find_template_point(tg,sg,lcx,lcy)
                if res is None: self.status=f"Match failed x{sz}"; return
                cxf,cyf,ni,H,mask,kp1,kp2,good=res
                Hf=H; nif=ni
                self.last_target_px=(cxf,cyf); self.last_inliers=ni
                self._update_live_roi(cxf,cyf,sz)
                ex,ey=cxf-cx_c,cyf-cy_c; self.last_error_px=(ex,ey)
                if math.hypot(ex,ey)>MICRO_CORRECT_PX:
                    self.status=f"Micro-correct x{sz}..."
                    self._correct(ex,ey,sz,"micro ")
                    time.sleep(2.0)
                    fz2=self._get_frame()
                    if fz2 is not None:
                        sg2=cv2.cvtColor(fz2,cv2.COLOR_BGR2GRAY)
                        r2=find_template_point(tg,sg2,lcx,lcy)
                        if r2 is not None:
                            cxf,cyf=r2[0],r2[1]; Hf,nif=r2[3],r2[2]
                            sg=sg2; sf=fz2; self._update_live_roi(cxf,cyf,sz)
                        else: cxf,cyf=float(cx_c),float(cy_c)
                ccx,ccy=cxf,cyf
                if si<len(zsteps)-1:
                    x1=max(0,int(ccx-EPICENTER_SIZE/2)); y1=max(0,int(ccy-EPICENTER_SIZE/2))
                    x2=min(IMG_W,int(ccx+EPICENTER_SIZE/2)); y2=min(IMG_H,int(ccy+EPICENTER_SIZE/2))
                    tg=sg[y1:y2,x1:x2]; lcx,lcy=ccx-x1,ccy-y1
            exf,eyf=ccx-cx_c,ccy-cy_c; self.last_error_px=(exf,eyf)
            self._correct(exf,eyf,tz,"final ")
            err=self.last_error_px
            self.status=f"Done err=({err[0]:+.0f},{err[1]:+.0f})px in={nif}"
            print(self.status)
            if Hf is not None:
                self._save_debug(sf,tg,Hf,ccx,ccy,cx_c,cy_c,exf,eyf)
        except Exception as e:
            self.status=f"Error: {e}"; rospy.logerr(str(e))
            import traceback; traceback.print_exc()
        finally:
            self.is_busy=False; self.current_step=0

    def _save_debug(self,fz,tg,H,cxf,cyf,cx_c,cy_c,ex,ey):
        dbg=fz.copy()
        th,tw=tg.shape[:2]
        corners=np.float32([[0,0],[tw,0],[tw,th],[0,th]]).reshape(-1,1,2)
        proj=cv2.perspectiveTransform(corners,H)
        cv2.polylines(dbg,[np.int32(proj)],True,(0,165,255),2)
        cv2.drawMarker(dbg,(int(cxf),int(cyf)),(0,165,255),cv2.MARKER_CROSS,24,2)
        col=CURSOR_COLORS[self.cursor_sel]
        cv2.drawMarker(dbg,(cx_c,cy_c),col,cv2.MARKER_CROSS,24,2)
        cv2.arrowedLine(dbg,(int(cxf),int(cyf)),(cx_c,cy_c),(255,255,0),2,tipLength=0.15)
        cv2.putText(dbg,f"err({ex:+.0f},{ey:+.0f})px",(cx_c+8,cy_c+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        cv2.imwrite(os.path.join(self.image_dir,"sift_correct_once_debug.jpg"),dbg)

    def _draw_overlay(self,img):
        if self.drawing:
            cv2.rectangle(img,(self.start_x,self.start_y),(self.current_x,self.current_y),(0,255,255),1)
        if self.roi_selected and self.roi:
            x,y,w,h=self.roi
            if self.live_roi is not None:
                lx,ly,lw,lh=self.live_roi
                rx,ry=int(lx-lw/2),int(ly-lh/2)
                cv2.rectangle(img,(rx,ry),(rx+int(lw),ry+int(lh)),(0,255,0),2)
                cv2.putText(img,"ROI",(rx,ry-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,0),1)
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.drawMarker(img,(x+w//2,y+h//2),(0,200,0),cv2.MARKER_CROSS,14,1)
                cv2.putText(img,"ROI",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,200,0),1)
        if self.last_target_px is not None:
            tx,ty=int(self.last_target_px[0]),int(self.last_target_px[1])
            cv2.drawMarker(img,(tx,ty),(0,165,255),cv2.MARKER_CROSS,24,3)
            cv2.putText(img,f"Target({tx},{ty})",(tx+8,ty-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,165,255),2)
        if self.last_target_px is not None and self.last_error_px is not None:
            cx_c,cy_c=CURSORS[self.cursor_sel]
            tx,ty=int(self.last_target_px[0]),int(self.last_target_px[1])
            cv2.arrowedLine(img,(tx,ty),(cx_c,cy_c),(255,255,0),2,tipLength=0.15)
            ex,ey=self.last_error_px
            cv2.putText(img,f"err({ex:+.0f},{ey:+.0f})px",(cx_c+8,cy_c+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        for k,(cx,cy) in CURSORS.items():
            if self.cursor_sel in (0,k):
                cv2.drawMarker(img,(cx,cy),CURSOR_COLORS[k],cv2.MARKER_CROSS,20,2)
                cv2.putText(img,CURSOR_LABELS[k],(cx+10,cy-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,CURSOR_COLORS[k],1)
        cv2.putText(img,self.status,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
        info=[f"Zoom:x{self.actual_zoom:.1f} [SIFT]"]
        if self.step_schedule:
            info.append(f"Steps:{self.step_schedule}")
            if self.is_busy and self.current_step>0:
                info.append(f"Step {self.current_step}/{len(self.step_schedule)-1}")
        if self.roi_texture is not None:
            lbl=f"Tex:{self.roi_texture:.0f}"
            if self.roi_texture<100: lbl+=" LOW"
            info.append(lbl)
        for j,line in enumerate(info):
            col=(0,80,255) if "LOW" in line else (0,220,255)
            cv2.putText(img,line,(10,60+j*26),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
        cv2.putText(img,f"Cursor:{self.cursor_sel}|z=correct r=reset ESC=quit",
                    (10,IMG_H-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)

    def _main_loop(self):
        rate=rospy.Rate(30)
        while not rospy.is_shutdown():
            frame=self._get_frame()
            if frame is not None:
                vis=frame.copy(); self._draw_overlay(vis)
                cv2.imshow("SIFT Correct Once",vis)
            key=cv2.waitKey(1)&0xFF
            if key==27: break
            elif key==ord("r") and not self.is_busy:
                self.roi=None; self.roi_selected=self.template_saved=False
                self.template_gray=None; self.last_target_px=self.last_error_px=None
                self.roi_texture=None; self.step_schedule=[]; self.live_roi=None
                self.current_zoom=1.0; self._set_zoom(1)
                self.status="Reset. Draw a new ROI."
            elif key==ord("z") and self.roi_selected and not self.is_busy:
                self.target_zoom=self.optimal_zoom; self.is_busy=True
                self.current_zoom=self.optimal_zoom
                self.last_target_px=self.last_error_px=None
                threading.Thread(target=self._zoom_and_correct,daemon=True).start()
            elif key in (ord("0"),ord("1"),ord("2"),ord("3")):
                self.cursor_sel=int(chr(key))
            rate.sleep()
        cv2.destroyAllWindows(); print("Bye")

if __name__ == '__main__':
    try: SiftCorrectOnce()
    except rospy.ROSInterruptException: pass

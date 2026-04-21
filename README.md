# 🎯 VIZOR - Visual Iterative Zoom ORB/Optical Flow Re-centering

This project implements a **ORB/Optical Flow -based control system** for a Pan-Tilt-Zoom (PTZ) camera using ROS Noetic. It corrects the image center drift caused by the optical zoom of the video camera using both Optical Flow and ORB at user's choice. This project makes a significant contribution with providing an **auto-calibration algorithm** that configures a K parameter to describe the FOV/Zoom Level curve **independently of the model/manufacturer of the camera**. Also implements a benchmark that warrants high reproducibility and a tracking mode that is under development.

---

## 📷 Hardware Used

- **Camera**: Datavideo BC-50 (20x optical zoom)
- **Pan-Tilt Base**: IQR-PTU2 (2 DoF)

---

## 📦 Features

- Manual selection of ROI before via user interface.
- Real-time computation of pan/tilt angles based on TF transformations.
- Estimation of object size in pixels from real-world dimensions and distance.
- Calculation of **optimal zoom level** using auto-calibration algorithm.
- Matryoshka relay to improve the performance and acurracy of the algorithm
- Micro-corrections at each step, in case the error is bellow the sensitivity of the Pan-Tilt base they acummulate to the next step.
- **Report generation** at each excecution with .csv, boxplots and video of the errors across the steps.
- Interactive flowchart that shows each phase of the algorithm as the program excecutes.
- Option of tracking target at the end of the zoom.

---

## 🧩 System Requirements

- Ubuntu 20.04
- ROS Noetic
- Python 3.8
- ffmpeg
- A real hardware PTZ system using:
  - Datavideo BC-50 camera
  - IQR-PTU2 pan-tilt base

## 🚀 Installation & Setup

## 1. Install system dependencies
**Instructions for ROS Noetic**
```bash
https://sir.upc.edu/projects/rostutorials/1-ROS_basic_concepts/index.html#install-instructions
```
**Dependencies**
```bash

sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport
```
```bash
sudo apt-get install ros-noetic-serial
```
```bash
sudo apt install python3-pip
```
```bash
pip3 install opencv-python==4.5.5.64
```

## 2. Create and initialize your workspace

```bash
mkdir -p ~/ptz_ws/src
```
```bash
cd ~/ptz_ws/src
```
## 3. Clone the repository
```bash
git clone https://github.com/<your-username>/ptz_geometric_control.git
```
## 4. Build the workspace
```bash
cd ~/ptz_ws
```
```bash
catkin_make
```
## 5. Source the workspace
```bash
source devel/setup.bash
```

Add to `.bashrc` to persist:

```bash
echo "source ~/ptz_ws/devel/setup.bash" >> ~/.bashrc
```

## 6. Launch

```bash
cd ~/ptz_ws
```

```bash
roslaunch pan_tilt_description panTilt_view.launch
```
### Tracking
```bash
rosrun pan_tilt_description lk_correct_once_tracking.py
```
### Benchmark (ORB-Optical flow comparison)
```bash
rosrun pan_tilt_description ptz_benchmark.py
```
### Benchmark (Optical Flow)
```bash
rosrun pan_tilt_description optical_flow_benchmark.py
```

## 🎛️ Keyboard controls 

### Main Program

| Key | Action |
|-----|--------|
| `Mouse` | ROI selection |
|`K`| Auto-calibration|
| `C` | Save template|
| `Z` | Apply recommended zoom |
| `W/S` | Tilt manual controls |
| `A/D` | Pan manual controls |
| `R` | Reset |
| `Q` / `ESC` | Quit |

### Tracking
| Key | Action |
|-----|--------|
| `Mouse` | ROI selection |
|`K`| Auto-calibration|
| `C` | Save template|
| `Z` | Apply recommended zoom |
| `W/S` | Tilt manual controls |
| `A/D` | Pan manual controls |
| `T` | Track|
| `G` | Record screen|
| `R` | Reset |
| `Q` / `ESC` | Quit |

## CUDA Video Processing Pipeline

Real-time video processing pipeline based on **CUDA** and **OpenCV**. The program captures a live stream (webcam or video file), converts frames to **grayscale**, and applies one GPU-accelerated operation, displaying the processed output in a window.

### Available operations (basic versions)

* **Edge Detection** (Sobel + threshold)
* **Crop** (ROI)
* **Scale**
* **Rotate**
* **Optical Flow** (**implemented from scratch in CUDA using the Lucas–Kanade sparse method**) and visualized with motion vectors (arrows)

---

## ⚙️ Project usage

The project consists of:

* **Main pipeline (C++/OpenCV + CUDA):** video capture, Host↔Device transfers, kernel launches, and visualization.
* **CUDA kernels (.cu files):** GPU implementations of the operations called from the main program.

---

## How it works

At startup, the program:

1. Opens the video source via OpenCV.
2. Reads one frame to determine resolution and format.
3. Converts frames to grayscale (**CV_8UC1**).
4. Allocates GPU input/output buffers and a CPU output buffer.
5. Enters the real-time loop:

   * capture frame
   * convert to grayscale
   * copy Host → Device
   * run the selected CUDA operation
   * copy Device → Host
   * display output using `cv::imshow`

---

## 🧰 Main commands

### 🔨 Build (single command)

From the project root directory:

```bash
make
```

This command:

* configures CMake (creates `build/`)
* compiles the executable `app`

Expected output:

* `build/app`

### ▶️ Run

```bash
Run from build/: cd build && ./app
```

At startup, select the operation:

* `1` → Edge Detection (Sobel)
* `2` → Crop
* `3` → Scale
* `4` → Rotate
* `5` → Optical Flow (Lucas–Kanade, CUDA)

---

## ⌨️ Runtime controls

* **ESC**: quit the program
* **Edge Detection only**: `w/s` (or `u/d`, depending on your version) to increase/decrease the threshold during execution
  *(Note: to capture keyboard input you must click on the OpenCV output window “Result” to give it focus.)*

---

## 🧹 Clean generated files

To remove the build folder and generated files:

```bash
make clean
```

---

## ✅ Dependencies

* **CUDA Toolkit** (nvcc)
* **OpenCV 4** (linked via CMake)

In `CMakeLists.txt`:

* `OpenCV_DIR="/usr/local/lib/cmake/opencv4"`

If OpenCV is installed elsewhere, update `OpenCV_DIR` accordingly.

---

## 🔚 Termination

Press **ESC** while the OpenCV window is active to exit.
If keyboard input is not detected, click the “Result” window first (focus issue).


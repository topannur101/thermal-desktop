import os
import cv2
import numpy as np

class ThermalCamera:
    def __init__(self, camera_index=1):
        """
        Initialize the thermal camera.
        Index 1 is often the USB thermal camera if the laptop has a built-in webcam at 0.
        """
        self.camera_index = camera_index
        self.cap = None

    def connect(self):
        print(f"Connecting to thermal camera index {self.camera_index}...")
        
        # Use DSHOW on Windows, V4L2 on Linux
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        
        if not self.cap.isOpened():
            print(f"Failed to open camera index {self.camera_index}.")
            return False

        # Attempt to request raw 16-bit format (Y16)
        fourcc = cv2.VideoWriter_fourcc(*'Y16 ')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        
        # Standard resolution for UV256
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

        # Check for black screen issue
        valid_frame = False
        for _ in range(10):
            ret, frame = self.cap.read()
            if ret and frame is not None and np.max(frame) > 0:
                valid_frame = True
                break
                
        if not valid_frame:
            print("Black screen detected in raw mode. Reconnecting in standard mode...")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

        return True

    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None

        # Print frame info once so the developer can see what the camera delivers
        if not hasattr(self, '_frame_info_printed'):
            print(f"[Camera] frame shape={frame.shape}, dtype={frame.dtype}")
            self._frame_info_printed = True

        # ── 16-bit raw (best quality) ─────────────────────────────────────────
        if frame.dtype == np.uint16:
            raw = frame.astype(np.float32)
            temp_celsius = (raw / 100.0) - 273.15   # raw≈Kelvin*100
            norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return temp_celsius, cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

        # ── 2-D uint8 ─────────────────────────────────────────────────────────
        if frame.ndim == 2:
            h, w = frame.shape
            # V4L2 often returns YUYV as (H, W*2): Y bytes are at even columns
            if w >= h * 2:
                y = frame[:, 0::2]          # extract Y samples → true (H, W/2)
            else:
                y = frame                   # already true greyscale
            raw = y.astype(np.float32)
            temp_celsius = 10.0 + (raw / 255.0) * 30.0
            norm = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return temp_celsius, cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

        # ── 3-D array ─────────────────────────────────────────────────────────
        channels = frame.shape[2]

        if channels == 2:
            # (H, W, 2) YUYV: channel-0 is Y (luminance)
            y = frame[:, :, 0]
            raw = y.astype(np.float32)
            temp_celsius = 10.0 + (raw / 255.0) * 30.0
            norm = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return temp_celsius, cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

        # channels == 3 or 4 — standard BGR / BGRA color image
        if channels == 4:
            frame = frame[:, :, :3]
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw = grey.astype(np.float32)
        temp_celsius = 15.0 + (raw / 255.0) * 25.0
        return temp_celsius, frame

    def release(self):
        if self.cap:
            self.cap.release()

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

        # Determine how to process based on what Windows actually gave us
        if frame.dtype == np.uint16 or (frame.dtype == np.uint8 and len(frame.shape) == 2):
            # We got raw GREYSCALE data! (16-bit or 8-bit)
            raw_data = frame.astype(np.float32)

            # --- ESTIMATED TEMPERATURE CALCULATION ---
            # NOTE: Every camera manufacturer maps grey values to Celsius differently.
            # This is a generic estimation. If 16-bit, it often represents Kelvin * 10 or * 100.
            # If 8-bit, it's heavily compressed and we have to guess based on standard room temps.
            if frame.dtype == np.uint16:
                 # E.g. raw value 30000 -> 300.00 K -> ~26.85 C
                 temp_celsius = (raw_data / 100.0) - 273.15
            else:
                 # Very rough estimate for 8-bit grey: map 0-255 to 10C-40C roughly
                 temp_celsius = 10.0 + (raw_data / 255.0) * 30.0

            # --- APPLY PSEUDO-COLOR ---
            # To turn the grey image into a thermal look, we normalize it to 0-255 first
            normalized_grey = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Apply a color map (COLORMAP_INFERNO or COLORMAP_JET)
            color_frame = cv2.applyColorMap(normalized_grey, cv2.COLORMAP_INFERNO)
            
            return temp_celsius, color_frame

        elif len(frame.shape) == 3 and frame.shape[2] == 2:
            # 2-channel frame (e.g. YUYV / YUV422 packed) — extract luma channel
            grey = frame[:, :, 0]
            raw_data = grey.astype(np.float32)

            # Rough estimate from the luma channel
            temp_celsius = 15.0 + (raw_data / 255.0) * 25.0

            normalized_grey = cv2.normalize(grey, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_frame = cv2.applyColorMap(normalized_grey, cv2.COLORMAP_INFERNO)
            return temp_celsius, color_frame

        else:
            # We got a standard color image (Windows transcoded it or camera's built-in feed).
            # We convert it back to grey to estimate temp.
            if len(frame.shape) == 2:
                # Already single-channel greyscale
                grey = frame
            else:
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_data = grey.astype(np.float32)

            # Rough estimate from transcoded grey
            temp_celsius = 15.0 + (raw_data / 255.0) * 25.0

            # Return the original frame directly to preserve its native colors
            return temp_celsius, frame

    def release(self):
        if self.cap:
            self.cap.release()

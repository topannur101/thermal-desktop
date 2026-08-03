import cv2
import numpy as np

print("Testing Thermal Camera Split Screen Mode...")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    # fallback to 0 if 1 is not opened
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open any camera.")
    exit()

print("Setting resolution to 256x384 and disabling RGB conversion...")
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

ret, frame = cap.read()
if ret and frame is not None:
    print(f"Frame received successfully!")
    print(f"Shape: {frame.shape}")
    print(f"Data type: {frame.dtype}")
    print(f"Max value: {np.max(frame)}")
    
    if len(frame.shape) == 3:
        print("Frame has 3 dimensions.")
    elif len(frame.shape) == 2:
        print("Frame has 2 dimensions.")
        if frame.shape[1] == 512:
            print("Width is 512 (256 * 2 bytes). This is raw YUY2 format!")
    
    # Save a small dump of the raw data to check
    with open("raw_frame_dump.npy", "wb") as f:
        np.save(f, frame)
    print("Saved raw frame to raw_frame_dump.npy for analysis.")
else:
    print("Failed to read frame at 256x384 resolution.")

cap.release()

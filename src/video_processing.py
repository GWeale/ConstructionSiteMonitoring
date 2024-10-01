import numpy as np
import struct

def read_video_frames(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    frames = []
    frame_size = 1920 * 1080 * 3
    for i in range(0, len(data), frame_size):
        frame = data[i:i+frame_size]
        if len(frame) < frame_size:
            break
        frame = np.frombuffer(frame, dtype=np.uint8).reshape((1080, 1920, 3))
        frames.append(frame)
    return frames

def equirectangular_to_cartesian(frame):
    h, w, _ = frame.shape
    theta = np.linspace(0, 2 * np.pi, w)
    phi = np.linspace(0, np.pi, h)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    cartesian = np.stack((x, y, z), axis=-1)
    return cartesian, frame

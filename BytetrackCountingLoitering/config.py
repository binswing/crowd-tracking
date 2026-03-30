# config.py
import numpy as np

CONF_THRESHOLD = 0.4
THRESHOLD_TIME = 5  # seconds
VELOCITY_THRESHOLD = 0.5  # pixels per frame, adjust based on FPS

# Polygon definition
polygon = np.array([
    (100,100),
    (500,100),
    (500,400),
    (100,400)
])
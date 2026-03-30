# loitering.py
import time
import numpy as np
from config import THRESHOLD_TIME, VELOCITY_THRESHOLD
from utils import inside_polygon

loiter_dict = {}

def check_loitering(tracks, polygon, loiter_dict, current_time):
    alerts = []
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = track
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if inside_polygon((cx, cy), polygon) and class_id == 0:  # Only for persons
            if track_id not in loiter_dict:
                loiter_dict[track_id] = {'start_time': current_time, 'positions': [(cx, cy)]}
            else:
                # Update positions for velocity check
                loiter_dict[track_id]['positions'].append((cx, cy))
                if len(loiter_dict[track_id]['positions']) > 5:  # Keep last 5 positions
                    loiter_dict[track_id]['positions'].pop(0)

                # Calculate velocity (average movement over last positions)
                if len(loiter_dict[track_id]['positions']) > 1:
                    displacements = [np.linalg.norm(np.array(p2) - np.array(p1)) for p1, p2 in zip(loiter_dict[track_id]['positions'][:-1], loiter_dict[track_id]['positions'][1:])]
                    avg_velocity = np.mean(displacements)
                else:
                    avg_velocity = 0

                duration = current_time - loiter_dict[track_id]['start_time']
                if duration > THRESHOLD_TIME and avg_velocity < VELOCITY_THRESHOLD:  # Time + low velocity
                    alerts.append(track_id)
        else:
            if track_id in loiter_dict:
                del loiter_dict[track_id]
    return alerts
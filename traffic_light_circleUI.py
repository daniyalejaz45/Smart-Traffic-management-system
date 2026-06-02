import cv2
from cv2 import putText


def draw_traffic_light(frame, state, x=10, y=60):
    color_map = {
        'Red': (0, 0, 255),
        'Yellow': (0, 255, 255),
        'Green': (0, 255, 0)
    }
    color = color_map.get(state, (255, 255, 255))

    # Draw circle for the light
    cv2.circle(frame, (x + 15, y), 10, color, -1)

    # Draw light label
    putText(frame, (x + 30, y + 5), state, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
import cv2


def draw_vehicle_info(frame, direction, count, signal_state, timer_text, color):
    # Text overlays
    cv2.putText(frame, f"Dir: {direction}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Count: {count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Signal: {signal_state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(frame, f"Timer: {timer_text}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return frame
import numpy as np
import time
import cv2
import dearpygui.dearpygui as dpg
from vehicle_detector import VehicleDetector
from ui_overlay import draw_vehicle_info

# -------------------- Constants --------------------
FRAME_WIDTH, FRAME_HEIGHT = 256, 192
texture_width, texture_height = FRAME_WIDTH * 2, FRAME_HEIGHT * 2
directions = ['North', 'South', 'East', 'West']
TIMER_UPDATE_INTERVAL = 5

# -------------------- Initial States --------------------
detector = VehicleDetector()
cams = {
    'North': cv2.VideoCapture('SampleVideos2/north.mp4'),
    'South': cv2.VideoCapture('SampleVideos2/south.mp4'),
    'East':  cv2.VideoCapture('SampleVideos/east.mp4'),
    'West':  cv2.VideoCapture('SampleVideos2/west.mp4'),
}
signal_states = {d: 'Red' for d in directions}
signal_timers = {d: 40 for d in directions}
vehicle_counts = {d: 0 for d in directions}
emergency_flags = {d: False for d in directions}

current_index = 0
start_time = time.time()
last_timer_update = time.time()

# -------------------- Signal Logic --------------------
def update_signal_timers():
    global last_timer_update
    now = time.time()
    if now - last_timer_update >= TIMER_UPDATE_INTERVAL:
        total = sum(vehicle_counts.values()) or 1
        for d in directions:
            ratio = vehicle_counts[d] / total
            signal_timers[d] = max(20, min(int(ratio * 100), 60))
        print("Updated timers:", signal_timers)
        last_timer_update = now

def update_signals():
    global current_index, start_time
    current_dir = directions[current_index]
    if time.time() - start_time >= signal_timers[current_dir]:
        for i, dir in enumerate(directions):
            if emergency_flags[dir]:
                current_index = i
                emergency_flags[dir] = False
                start_time = time.time()
                break
        else:
            current_index = (current_index + 1) % len(directions)
            start_time = time.time()
    for d in directions:
        signal_states[d] = 'Red'
    signal_states[directions[current_index]] = 'Green'

# -------------------- Dear PyGui Setup --------------------
dpg.create_context()
dpg.create_viewport(title="Traffic Camera Grid", width=texture_width + 50, height=texture_height + 100)

with dpg.texture_registry(show=False):
    texture_id = dpg.add_dynamic_texture(texture_width, texture_height, [0.0] * texture_width * texture_height * 4)

with dpg.window(label="Traffic Dashboard"):
    dpg.add_image(texture_id)

dpg.setup_dearpygui()
dpg.show_viewport()

# -------------------- Main Loop --------------------

last_frame_hash = None

while dpg.is_dearpygui_running():
    update_signal_timers()
    update_signals()

    frames = []
    for direction in directions:
        cap = cams[direction]
        ret, frame = cap.read()
        if not ret:
            print(f"[{direction}] Restarting video")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print(f"[{direction}] Failed to read even after restart")
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame, count, is_emergency = detector.detect(frame)
        vehicle_counts[direction] = count
        emergency_flags[direction] = is_emergency

        if direction == directions[current_index]:
            time_left = max(0, int(signal_timers[direction] - (time.time() - start_time)))
            timer_text = f"{time_left}s"
            color = (0, 255, 0)
        else:
            timer_text = f"{signal_timers[direction]}s"
            color = (0, 0, 255)

        frame = draw_vehicle_info(frame, direction, count, signal_states[direction], timer_text, color)
        frames.append(frame)

    # Handle case where fewer than 4 frames are present
    while len(frames) < 4:
        frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))

    top = np.hstack((frames[0], frames[1]))
    bottom = np.hstack((frames[2], frames[3]))
    grid = np.vstack((top, bottom))

    # Convert BGR to RGBA and normalize
    rgba = cv2.cvtColor(grid, cv2.COLOR_BGR2RGBA)
    rgba = rgba.flatten() / 255.0  # efficient flattening + normalization

    # Only update texture if changed
    new_hash = hash(rgba.tobytes())
    if new_hash != last_frame_hash:
        dpg.set_value(texture_id, rgba.tolist())
        last_frame_hash = new_hash

    dpg.render_dearpygui_frame()
    time.sleep(0.06)  # ~16 FPS max

dpg.destroy_context()

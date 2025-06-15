import cv2
import mediapipe as mp
import numpy as np
import math

# ---- CONFIGURABLE PARAMETERS ----
INPUT_VIDEO_PATH = "C:\\Users\\..."           # Set your input video path here
OUTPUT_VIDEO_PATH = "C:\\Users\\..."
OUTPUT_MINMAX_IMAGE_PATH = "C:\\..." # in *.png
OUTPUT_TABLE_IMAGE_PATH = "C:\\..."  # in *.png
SIDE = "right"
FRAME_STEP = 1  # 1=every frame, 2=every second frame, etc.

MAX_ROW_IMAGES_WIDTH = 500

MAIN_CIRCLE_RADIUS = 15
MAIN_CIRCLE_FILL_RADIUS = 8
MAIN_CIRCLE_COLOR = (255, 0, 0)
MAIN_LINE_COLOR = (255, 255, 255)
MAIN_ARC_COLOR = (255, 0, 0)
MAIN_ARC_FILL = (255, 0, 0)
MAIN_LETTER_COLOR = (255, 255, 255)
MAIN_NUM_FONT_SCALE = 0.7
MAIN_NUM_FONT_THICKNESS = 2

TABLE_WIDTH = 510
TABLE_ALPHA = 0.6
TABLE_BG_COLOR = (255,255,255)
TABLE_BORDER_COLOR = (255,0,0)
TABLE_BORDER_THICKNESS = 2
TABLE_HEADER_BG = (230, 230, 250)
TABLE_HEADER_FONT_COLOR = (0,0,0)
TABLE_HEADER_FONT_SCALE = 0.52
TABLE_HEADER_FONT_THICKNESS = 1
TABLE_CELL_FONT_SCALE = 0.48
TABLE_CELL_FONT_THICKNESS = 1
GREEN = (0, 180, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
GRAY = (90, 90, 90)
ROW_HEIGHT = 35

MIN_DETECTION_CONFIDENCE = 1
MIN_TRACKING_CONFIDENCE = 1

TABLE_COLUMNS = [
    {"name": "#", "width": 40},
    {"name": "Angle name", "width": 170},
    {"name": "Min", "width": 100},
    {"name": "Max", "width": 100},
    {"name": "Recommended", "width": 100},
]

ANGLE_SPECS = [
    {"label": "Elbow angle", "min": 150, "max": 160},
    {"label": "Shoulder angle", "min": 85, "max": 90},
    {"label": "Torso angle", "min": 40, "max": 50},
    {"label": "Hip angle", "min": 60, "max": 110},
    {"label": "Knee angle", "min": 65, "max": 145},
    {"label": "Ankle angle", "min": 75, "max": 105},
]

def get_body_side_landmarks(landmarks, side='right'):
    side_ids = {
        'right': [
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
            mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
        ]
    }
    landmark_dict = {}
    for lm in side_ids[side]:
        landmark = landmarks[lm.value]
        landmark_dict[lm.name] = (landmark.x, landmark.y)
    return landmark_dict

def calc_angle_internal(ptA, ptB, ptC):
    a = np.array(ptA)
    b = np.array(ptB)
    c = np.array(ptC)
    ba = a - b
    bc = c - b
    angle_rad = np.arccos(
        np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)
    )
    angle_deg = np.degrees(angle_rad)
    return angle_deg, ba, bc

def get_angle_points_arc_from_start_to_end(center, start, end, arc_radius, npoints=30):
    v1 = np.array(start) - np.array(center)
    v2 = np.array(end) - np.array(center)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    dtheta = angle2 - angle1
    while dtheta <= -np.pi:
        angle2 += 2 * np.pi
        dtheta = (angle2 - angle1)
    while dtheta > np.pi:
        angle2 -= 2 * np.pi
        dtheta = (angle2 - angle1)
    if dtheta < 0:
        angle1, angle2 = angle2, angle1
    arc_points = []
    for t in np.linspace(angle1, angle2, npoints):
        x = int(center[0] + arc_radius * math.cos(t))
        y = int(center[1] + arc_radius * math.sin(t))
        arc_points.append((x, y))
    return np.array(arc_points, dtype=np.int32), angle1, angle2

def draw_filled_semiblue_arc(image, pA, pB, pC, alpha=0.5):
    dist_BA = np.linalg.norm(np.array(pA) - np.array(pB))
    dist_BC = np.linalg.norm(np.array(pC) - np.array(pB))
    arc_radius = int(min(dist_BA, dist_BC) / 3)
    arc_points, angle1, angle2 = get_angle_points_arc_from_start_to_end(pB, pA, pC, arc_radius)
    polygon = np.vstack([[pB], arc_points])
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], MAIN_ARC_FILL)
    cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
    radius = arc_radius * 0.55
    angle_mid = angle1 + (angle2 - angle1) * 0.5
    x = int(pB[0] + radius * math.cos(angle_mid))
    y = int(pB[1] + radius * math.sin(angle_mid))
    return (x, y)

def draw_arc_from_to(image, center, start, end, alpha=0.5, arc_radius=None):
    if arc_radius is None:
        dist_start = np.linalg.norm(np.array(start) - np.array(center))
        dist_end = np.linalg.norm(np.array(end) - np.array(center))
        arc_radius = int(min(dist_start, dist_end) / 3)
    else:
        arc_radius = int(arc_radius)
    arc_points, angle1, angle2 = get_angle_points_arc_from_start_to_end(center, start, end, arc_radius)
    polygon = np.vstack([[center], arc_points])
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], MAIN_ARC_FILL)
    cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
    radius = arc_radius * 0.55
    angle_mid = angle1 + (angle2 - angle1) * 0.5
    x = int(center[0] + radius * math.cos(angle_mid))
    y = int(center[1] + radius * math.sin(angle_mid))
    return (x, y)

def get_circle_edge_point(center, next_center, radius):
    vector = np.array(next_center) - np.array(center)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return center
    dir_vec = vector / norm
    edge_x = int(center[0] + dir_vec[0]*radius)
    edge_y = int(center[1] + dir_vec[1]*radius)
    return (edge_x, edge_y)

def elongate_line(ptA, ptB, factor=0.5):
    vec = np.array(ptB) - np.array(ptA)
    ptC = np.array(ptB) + factor * vec
    return tuple(ptC.astype(int))

def intersection_point(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (int(px), int(py))

def draw_points_and_lines(image, pts, order, white_radius=15, thickness=1):
    n = len(order)
    for idx, k in enumerate(order):
        cv2.circle(image, pts[k], white_radius, (255,255,255), 2, lineType=cv2.LINE_AA)
        cv2.circle(image, pts[k], MAIN_CIRCLE_FILL_RADIUS, MAIN_CIRCLE_COLOR, -1, lineType=cv2.LINE_AA)
    for idx in range(n-1):
        k1 = order[idx]
        k2 = order[idx+1]
        edge1 = get_circle_edge_point(pts[k1], pts[k2], white_radius)
        edge2 = get_circle_edge_point(pts[k2], pts[k1], white_radius)
        cv2.line(image, edge1, edge2, MAIN_LINE_COLOR, thickness, lineType=cv2.LINE_AA)

def annotate_pose_frame(frame, pose_landmarks, side='right', summary_angle_idx=None):
    h, w = frame.shape[:2]
    landmark_dict = get_body_side_landmarks(pose_landmarks, side=side)
    get = lambda name: (int(landmark_dict[name][0]*w), int(landmark_dict[name][1]*h))
    pts = {
        'shoulder': get('RIGHT_SHOULDER'),
        'elbow': get('RIGHT_ELBOW'),
        'wrist': get('RIGHT_WRIST'),
        'hip': get('RIGHT_HIP'),
        'knee': get('RIGHT_KNEE'),
        'ankle': get('RIGHT_ANKLE'),
        'heel': get('RIGHT_HEEL'),
        'foot_index': get('RIGHT_FOOT_INDEX'),
    }
    points_order = ['wrist', 'elbow', 'shoulder', 'hip', 'knee', 'ankle', 'heel', 'foot_index']
    draw_points_and_lines(frame, pts, points_order, white_radius=MAIN_CIRCLE_RADIUS, thickness=2)
    digit_positions = []
    # 1. Elbow
    pos1 = draw_filled_semiblue_arc(frame, pts['wrist'], pts['elbow'], pts['shoulder'], alpha=0.5) if summary_angle_idx in [None,0] else None
    digit_positions.append(pos1)
    # 2. Shoulder
    pos2 = draw_filled_semiblue_arc(frame, pts['elbow'], pts['shoulder'], pts['hip'], alpha=0.5) if summary_angle_idx in [None,1] else None
    digit_positions.append(pos2)
    # 3. Shoulder-Hip-Horizontal
    p2 = pts['shoulder']
    p3 = pts['hip']
    line_vec = np.array(p2) - np.array(p3)
    line_length = np.linalg.norm(line_vec)
    horiz_len = line_length / 2
    p3_edge = get_circle_edge_point(p3, (p3[0]+100, p3[1]), MAIN_CIRCLE_RADIUS)
    p_horiz_end = (int(p3_edge[0] + horiz_len), p3_edge[1])
    angle3, _, _ = calc_angle_internal(p_horiz_end, p3, p2)
    arc_radius = int(np.linalg.norm(np.array(p_horiz_end) - np.array(p3)))
    pos3 = draw_arc_from_to(frame, p3, p_horiz_end, p2, alpha=0.5, arc_radius=arc_radius) if summary_angle_idx in [None,2] else None
    digit_positions.append(pos3)
    # 4. Hip
    pos4 = draw_filled_semiblue_arc(frame, pts['shoulder'], pts['hip'], pts['knee'], alpha=0.5) if summary_angle_idx in [None,3] else None
    digit_positions.append(pos4)
    # 5. Knee
    pos5 = draw_filled_semiblue_arc(frame, pts['hip'], pts['knee'], pts['ankle'], alpha=0.5) if summary_angle_idx in [None,4] else None
    digit_positions.append(pos5)
    # 6. Ankle (intersection arc)
    elongated_ankle = tuple((np.array(pts['ankle']) + 0.5 * (np.array(pts['ankle']) - np.array(pts['knee']))).astype(int))
    intersec = intersection_point(pts['heel'], pts['foot_index'], pts['knee'], elongated_ankle)
    if intersec is not None and summary_angle_idx in [None,5]:
        angle6, _, _ = calc_angle_internal(pts['ankle'], intersec, pts['foot_index'])
        ankle_footindex_dist = np.linalg.norm(np.array(pts['ankle']) - np.array(pts['foot_index']))
        arc6_radius = int((2/3) * ankle_footindex_dist)
        pos6 = draw_arc_from_to(frame, intersec, pts['ankle'], pts['foot_index'], alpha=0.5, arc_radius=arc6_radius)
        digit_positions.append(pos6)
    else:
        digit_positions.append(None)
        angle6 = 0.0
    for i, pos in enumerate(digit_positions):
        if pos is not None and (summary_angle_idx is None or i == summary_angle_idx):
            x, y = pos
            cv2.putText(frame, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, MAIN_NUM_FONT_SCALE, (0,0,0), MAIN_NUM_FONT_THICKNESS+2, cv2.LINE_AA)
            cv2.putText(frame, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, MAIN_NUM_FONT_SCALE, MAIN_LETTER_COLOR, MAIN_NUM_FONT_THICKNESS, cv2.LINE_AA)
    angle1, _, _ = calc_angle_internal(pts['wrist'], pts['elbow'], pts['shoulder'])
    angle2, _, _ = calc_angle_internal(pts['elbow'], pts['shoulder'], pts['hip'])
    angle4, _, _ = calc_angle_internal(pts['shoulder'], pts['hip'], pts['knee'])
    angle5, _, _ = calc_angle_internal(pts['hip'], pts['knee'], pts['ankle'])
    angle3 = angle3
    return [angle1, angle2, angle3, angle4, angle5, angle6], frame

def process_video(input_path, output_path, side='right', frame_step=1):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
    min_vals = [float('inf')] * 6
    max_vals = [float('-inf')] * 6
    min_frames = [None] * 6
    max_frames = [None] * 6
    min_landmarks = [None] * 6
    max_landmarks = [None] * 6
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_step != 0:
            i += 1
            continue
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            angles, annotated = annotate_pose_frame(frame.copy(), results.pose_landmarks.landmark, side=side)
            out.write(annotated)
            for j in range(6):
                if angles[j] < min_vals[j]:
                    min_vals[j] = angles[j]
                    min_frames[j] = frame.copy()
                    min_landmarks[j] = results.pose_landmarks.landmark
                if angles[j] > max_vals[j]:
                    max_vals[j] = angles[j]
                    max_frames[j] = frame.copy()
                    max_landmarks[j] = results.pose_landmarks.landmark
        else:
            out.write(frame)
        i += 1
    cap.release()
    out.release()
    return min_frames, max_frames, min_landmarks, max_landmarks, min_vals, max_vals

def pad_width(img, target_width):
    h, w = img.shape[:2]
    if w == target_width:
        return img
    elif w < target_width:
        pad = target_width - w
        return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
    else:
        return img[:, :target_width]

def save_summary_minmax_image(min_frames, max_frames, min_landmarks, max_landmarks, min_vals, max_vals, out_path, side='right'):
    rows = len(min_frames)
    img_rows = []
    for k in range(rows):
        # Safeguard against missing frames
        if min_frames[k] is not None:
            min_img = min_frames[k].copy()
            if min_landmarks[k] is not None:
                _, min_img = annotate_pose_frame(min_img, min_landmarks[k], side=side, summary_angle_idx=k)
        else:
            min_img = np.ones((400, 500, 3), dtype=np.uint8) * 255
            cv2.putText(min_img, f"No min frame for {ANGLE_SPECS[k]['label']}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if max_frames[k] is not None:
            max_img = max_frames[k].copy()
            if max_landmarks[k] is not None:
                _, max_img = annotate_pose_frame(max_img, max_landmarks[k], side=side, summary_angle_idx=k)
        else:
            max_img = np.ones((400, 500, 3), dtype=np.uint8) * 255
            cv2.putText(max_img, f"No max frame for {ANGLE_SPECS[k]['label']}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        # Resize to target width
        min_h, min_w = min_img.shape[:2]
        scale = MAX_ROW_IMAGES_WIDTH / min_w
        min_img_resized = cv2.resize(min_img, (MAX_ROW_IMAGES_WIDTH, int(min_h*scale)))
        max_img_resized = cv2.resize(max_img, (MAX_ROW_IMAGES_WIDTH, int(min_h*scale)))
        # Title
        label = ANGLE_SPECS[k]["label"]
        min_title = np.ones((40, MAX_ROW_IMAGES_WIDTH, 3), dtype=np.uint8) * 255
        max_title = np.ones((40, MAX_ROW_IMAGES_WIDTH, 3), dtype=np.uint8) * 255
        cv2.putText(min_title, f"{label} min: {min_vals[k]:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(max_title, f"{label} max: {max_vals[k]:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        min_title = pad_width(min_title, MAX_ROW_IMAGES_WIDTH)
        max_title = pad_width(max_title, MAX_ROW_IMAGES_WIDTH)
        min_img_resized = pad_width(min_img_resized, MAX_ROW_IMAGES_WIDTH)
        max_img_resized = pad_width(max_img_resized, MAX_ROW_IMAGES_WIDTH)
        min_full = np.vstack([min_title, min_img_resized])
        max_full = np.vstack([max_title, max_img_resized])
        if min_full.shape[0] != max_full.shape[0]:
            hmax = max(min_full.shape[0], max_full.shape[0])
            min_full = cv2.copyMakeBorder(min_full, 0, hmax-min_full.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
            max_full = cv2.copyMakeBorder(max_full, 0, hmax-max_full.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        min_full = pad_width(min_full, MAX_ROW_IMAGES_WIDTH)
        max_full = pad_width(max_full, MAX_ROW_IMAGES_WIDTH)
        row_img = np.hstack([min_full, max_full])
        img_rows.append(row_img)
    out_img = np.vstack(img_rows)
    cv2.imwrite(out_path, out_img)

def extract_and_annotate_frame(video_path, side='right', frame_index=2):
    cap = cv2.VideoCapture(video_path)
    frame = None
    i = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if i == frame_index:
            frame = f
            break
        i += 1
    cap.release()
    if frame is None:
        return None
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        _, annotated = annotate_pose_frame(frame.copy(), results.pose_landmarks.landmark, side=side)
        return annotated
    else:
        return frame

def save_table_image_overlay(min_vals, max_vals, out_path, video_path, side='right', header_frame_idx=2, overlay_alpha=0.7):
    rows = len(ANGLE_SPECS)
    table_height = ROW_HEIGHT * (rows+1)
    table_width = sum([col["width"] for col in TABLE_COLUMNS]) + 2

    # Get the annotated header image
    header_img = extract_and_annotate_frame(video_path, side=side, frame_index=header_frame_idx)
    if header_img is None:
        print("Warning: Could not extract header image. Creating blank image.")
        header_img = np.ones((400, table_width, 3), dtype=np.uint8) * 255

    # Resize header image to table width
    header_h, header_w = header_img.shape[:2]
    scale = table_width / header_w
    new_h = int(header_h * scale)
    header_img_resized = cv2.resize(header_img, (table_width, new_h))

    # Create the table as before
    table_img = np.ones((table_height+60, table_width, 3), dtype=np.uint8)*255
    overlay = table_img.copy()
    cv2.rectangle(overlay, (0, 0), (table_width, table_height), TABLE_BG_COLOR, -1)
    table_img = cv2.addWeighted(overlay, TABLE_ALPHA, table_img, 1-TABLE_ALPHA, 0)
    cv2.rectangle(table_img, (0, 0), (table_width, ROW_HEIGHT), TABLE_HEADER_BG, -1)
    cell_x = 0
    for idx, col in enumerate(TABLE_COLUMNS):
        col_text = col["name"]
        font_scale = TABLE_HEADER_FONT_SCALE
        font_thick = TABLE_HEADER_FONT_THICKNESS
        cv2.putText(table_img, col_text, (cell_x+8, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TABLE_HEADER_FONT_COLOR, font_thick, cv2.LINE_AA)
        cell_x += col["width"]
    cell_x = 0
    for col in TABLE_COLUMNS:
        cv2.rectangle(table_img, (cell_x, 0), (cell_x+col["width"], table_height), TABLE_BORDER_COLOR, TABLE_BORDER_THICKNESS)
        cell_x += col["width"]
    for row in range(rows+1):
        row_y = row * ROW_HEIGHT
        cv2.line(table_img, (0, row_y), (table_width, row_y), TABLE_BORDER_COLOR, TABLE_BORDER_THICKNESS)
    for i, spec in enumerate(ANGLE_SPECS):
        row_y = (i+1)*ROW_HEIGHT
        cell_x = 0
        cv2.putText(table_img, str(i+1), (cell_x+12, row_y+25), cv2.FONT_HERSHEY_SIMPLEX, TABLE_CELL_FONT_SCALE, BLACK, TABLE_CELL_FONT_THICKNESS, cv2.LINE_AA)
        cell_x += TABLE_COLUMNS[0]["width"]
        label = spec["label"]
        cv2.putText(table_img, label, (cell_x+3, row_y+25), cv2.FONT_HERSHEY_SIMPLEX, TABLE_CELL_FONT_SCALE, BLACK, TABLE_CELL_FONT_THICKNESS, cv2.LINE_AA)
        cell_x += TABLE_COLUMNS[1]["width"]
        color_min = GREEN if spec["min"] <= min_vals[i] <= spec["max"] else RED
        cv2.putText(table_img, f"{min_vals[i]:.1f}", (cell_x+15, row_y+25), cv2.FONT_HERSHEY_SIMPLEX, TABLE_CELL_FONT_SCALE, color_min, TABLE_CELL_FONT_THICKNESS+1, cv2.LINE_AA)
        cell_x += TABLE_COLUMNS[2]["width"]
        color_max = GREEN if spec["min"] <= max_vals[i] <= spec["max"] else RED
        cv2.putText(table_img, f"{max_vals[i]:.1f}", (cell_x+15, row_y+25), cv2.FONT_HERSHEY_SIMPLEX, TABLE_CELL_FONT_SCALE, color_max, TABLE_CELL_FONT_THICKNESS+1, cv2.LINE_AA)
        cell_x += TABLE_COLUMNS[3]["width"]
        rec_text = f"{spec['min']} to {spec['max']}"
        cv2.putText(table_img, rec_text, (cell_x+11, row_y+25), cv2.FONT_HERSHEY_SIMPLEX, TABLE_CELL_FONT_SCALE, GRAY, TABLE_CELL_FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(table_img, "*) Recommend Elbow and Shoulder angles for hands on bar (not on drops).", (15, table_height+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

    # Overlay the table on the bottom of the header image with alpha
    canvas_height = header_img_resized.shape[0] + table_img.shape[0]
    canvas = np.ones((canvas_height, table_width, 3), dtype=np.uint8) * 255
    canvas[:header_img_resized.shape[0], :, :] = header_img_resized
    # Overlay semi-transparent table at the bottom
    y0 = canvas_height - table_img.shape[0]
    overlay = canvas[y0:y0+table_img.shape[0], :, :].copy()
    cv2.addWeighted(table_img, overlay_alpha, overlay, 1-overlay_alpha, 0, overlay)
    canvas[y0:y0+table_img.shape[0], :, :] = overlay

    cv2.imwrite(out_path, canvas)

if __name__ == "__main__":
    print("Processing video. This may take a while...")
    min_frames, max_frames, min_landmarks, max_landmarks, min_vals, max_vals = process_video(
        INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, side=SIDE, frame_step=FRAME_STEP)
    print("Saving min/max composite image (summary arcs only)...")
    save_summary_minmax_image(
        min_frames, max_frames, min_landmarks, max_landmarks, min_vals, max_vals, OUTPUT_MINMAX_IMAGE_PATH, side=SIDE)
    print("Saving table image (overlay style)...")
    save_table_image_overlay(
        min_vals, max_vals, OUTPUT_TABLE_IMAGE_PATH, INPUT_VIDEO_PATH, side=SIDE, header_frame_idx=2, overlay_alpha=0.7)
    print("Done!")
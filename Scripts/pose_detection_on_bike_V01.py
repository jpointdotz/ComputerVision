import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose

# =========================================
# ---- Editable COLOR and SIZE PARAMETERS
# =========================================

# Colors are in BGR (OpenCV format)
SKELETON_COLOR = (255, 0, 0)        # Blue for lines/arcs/circles in skeleton model
SKELETON_BG = (255, 255, 255, 153)  # White (with alpha) for skeleton background
SKELETON_CIRCLE_COLOR = (255, 0, 0, 255)  # Blue, opaque for circles in skeleton
SKELETON_LETTER_COLOR = (255, 0, 0, 255)  # Blue, opaque for numbers in skeleton
SKELETON_LINE_COLOR = (255, 0, 0, 255)    # Blue, opaque for lines/arcs in skeleton

MAIN_CIRCLE_COLOR = (255, 0, 0)     # Blue for joint circles in main picture
MAIN_LETTER_COLOR = (255, 255, 255) # White for angle numbers in main picture
MAIN_LINE_COLOR = (255, 255, 255)   # White for lines in main picture
MAIN_BG_COLOR = (255, 255, 255)     # White for overlays (not used for main, just as reference)
MAIN_ARC_COLOR = (255, 0, 0)        # Blue for arc outline/dashed in main
MAIN_ARC_FILL = (255, 0, 0)         # Blue for filled arc in main

TABLE_BG_COLOR = (255,255,255)      # White for angle table background
TABLE_BORDER_COLOR = (255,0,0)      # Blue for angle table border
TABLE_LETTER_COLOR = (255,0,0)      # Blue for text in table

# ---- Sizing (change as needed)
SKELETON_MARGIN = 20                # Margin for skeleton model in rectangle (px)
SKELETON_CIRCLE_RADIUS = 0          # Joint circle radius in skeleton model
SKELETON_CIRCLE_RADIUS_DRAW = 4     # Filled blue part in skeleton
SKELETON_CIRCLE_OUTLINE = 0         # White outline for joint in skeleton

MAIN_CIRCLE_RADIUS = 15             # White outline for joint in main
MAIN_CIRCLE_FILL_RADIUS = 8         # Blue fill for joint in main
SKELETON_LINE_THICKNESS = 1
MAIN_LINE_THICKNESS = 2

SKELETON_NUM_FONT_SCALE = 0.48
SKELETON_NUM_FONT_THICKNESS = 1
MAIN_NUM_FONT_SCALE = 0.7
MAIN_NUM_FONT_THICKNESS = 2

TABLE_WIDTH = 150
TABLE_ALPHA = 0.6
TABLE_BORDER_THICKNESS = 1

# =========================================

def get_body_side_landmarks(landmarks, side='right'):
    side_ids = {
        'right': [
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.RIGHT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_HEEL,
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
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

def get_angle_points(pA, pB, pC, arc_radius, npoints=30):
    v1 = np.array(pA) - np.array(pB)
    v2 = np.array(pC) - np.array(pB)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    dtheta = (angle2 - angle1)
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
        x = int(pB[0] + arc_radius * math.cos(t))
        y = int(pB[1] + arc_radius * math.sin(t))
        arc_points.append((x, y))
    return np.array(arc_points, dtype=np.int32), angle1, angle2

def draw_filled_semiblue_arc(image, pA, pB, pC, number=None, alpha=0.5):
    """Filled blue arc with white number at midpoint for main picture."""
    dist_BA = np.linalg.norm(np.array(pA) - np.array(pB))
    dist_BC = np.linalg.norm(np.array(pC) - np.array(pB))
    arc_radius = int(min(dist_BA, dist_BC) / 3)
    arc_points, angle1, angle2 = get_angle_points(pA, pB, pC, arc_radius)
    polygon = np.vstack([[pB], arc_points])
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], MAIN_ARC_FILL)
    cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)
    if number is not None:
        radius = arc_radius * 0.55
        angle_mid = angle1 + (angle2 - angle1) * 0.5
        x = int(pB[0] + radius * math.cos(angle_mid))
        y = int(pB[1] + radius * math.sin(angle_mid))
        cv2.putText(image, str(number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, MAIN_NUM_FONT_SCALE, MAIN_LETTER_COLOR, MAIN_NUM_FONT_THICKNESS, cv2.LINE_AA)

def draw_dashed_arc(img, center, radius, angle1, angle2, color, thickness=2, dash_length=12, gap_length=8):
    arc_length = abs(angle2 - angle1) * radius
    n_dashes = max(1, int(arc_length // (dash_length + gap_length)))
    for i in range(n_dashes+1):
        t1 = angle1 + (angle2 - angle1) * (i * (dash_length + gap_length) / arc_length)
        t2 = t1 + (angle2 - angle1) * (dash_length / arc_length)
        if (angle2 > angle1 and t2 > angle2) or (angle2 < angle1 and t2 < angle2):
            t2 = angle2
        p1 = (int(center[0] + radius * math.cos(t1)), int(center[1] + radius * math.sin(t1)))
        p2 = (int(center[0] + radius * math.cos(t2)), int(center[1] + radius * math.sin(t2)))
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=12, gap_length=8):
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    dist = np.linalg.norm(pt2 - pt1)
    direction = (pt2 - pt1) / dist if dist > 0 else np.zeros(2)
    n_dashes = int(dist // (dash_length + gap_length))
    for i in range(n_dashes+1):
        start = pt1 + direction * (i * (dash_length + gap_length))
        end = start + direction * dash_length
        if np.linalg.norm(end - pt1) > dist:
            end = pt2
        cv2.line(img, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), color, thickness, lineType=cv2.LINE_AA)

def draw_arc_outline_and_number(image, pA, pB, pC, color, number, number_outside=False, font_scale=0.55, font_thickness=1, arc_thickness=1, npoints=40, avoid_points=None):
    dist_BA = np.linalg.norm(np.array(pA) - np.array(pB))
    dist_BC = np.linalg.norm(np.array(pC) - np.array(pB))
    arc_radius = int(min(dist_BA, dist_BC) / 3)
    arc_points, angle1, angle2 = get_angle_points(pA, pB, pC, arc_radius)
    prev = None
    for t in np.linspace(angle1, angle2, npoints):
        x = int(pB[0] + arc_radius * math.cos(t))
        y = int(pB[1] + arc_radius * math.sin(t))
        if prev is not None:
            cv2.line(image, prev, (x, y), color, arc_thickness, lineType=cv2.LINE_AA)
        prev = (x, y)
    base_radius = arc_radius * (0.55 if not number_outside else 1.12)
    angle_mid = angle1 + (angle2 - angle1) * 0.5
    x = int(pB[0] + base_radius * math.cos(angle_mid))
    y = int(pB[1] + base_radius * math.sin(angle_mid))
    if avoid_points is not None:
        for ap in avoid_points:
            if np.linalg.norm(np.array([x, y]) - np.array(ap)) < 18:
                base_radius = arc_radius * 1.18
                x = int(pB[0] + base_radius * math.cos(angle_mid))
                y = int(pB[1] + base_radius * math.sin(angle_mid))
    cv2.putText(image, str(number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

def draw_circle_point(image, point, blue_radius=5, white_radius=8, color=(255,0,0,255), white=True):
    # Joint circle: white (outline) and blue (filled)
    if white:
        cv2.circle(image, point, white_radius, (255,255,255,255), 2, lineType=cv2.LINE_AA)
    cv2.circle(image, point, blue_radius, color, -1, lineType=cv2.LINE_AA)

def get_circle_edge_point(center, next_center, radius):
    # Returns a point on the circumference toward next_center
    vector = np.array(next_center) - np.array(center)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return center
    dir_vec = vector / norm
    edge_x = int(center[0] + dir_vec[0]*radius)
    edge_y = int(center[1] + dir_vec[1]*radius)
    return (edge_x, edge_y)

def draw_points_and_lines(image, pts, order, white_radius=15, color=(255,0,0), white=True, thickness=1):
    n = len(order)
    for idx, k in enumerate(order):
        # Draw white outline then blue fill
        cv2.circle(image, pts[k], white_radius, (255,255,255), 2, lineType=cv2.LINE_AA) if white else None
        cv2.circle(image, pts[k], MAIN_CIRCLE_FILL_RADIUS, MAIN_CIRCLE_COLOR, -1, lineType=cv2.LINE_AA)
    for idx in range(n-1):
        k1 = order[idx]
        k2 = order[idx+1]
        # Draw line from edge of one circle to edge of next
        edge1 = get_circle_edge_point(pts[k1], pts[k2], white_radius)
        edge2 = get_circle_edge_point(pts[k2], pts[k1], white_radius)
        cv2.line(image, edge1, edge2, MAIN_LINE_COLOR, MAIN_LINE_THICKNESS, lineType=cv2.LINE_AA)

def draw_text_and_table(image, angles, angle_labels, table_width, skel_box_x, skel_box_y, skel_box_w):
    # Table width matches skeleton model rectangle
    table_height = 30*len(angles) + 30
    x_start = skel_box_x
    y_start = 10
    overlay = image.copy()
    # Table background
    cv2.rectangle(overlay, (x_start, y_start), (x_start+table_width, y_start+table_height), TABLE_BG_COLOR, -1)
    image = cv2.addWeighted(overlay, TABLE_ALPHA, image, 1-TABLE_ALPHA, 0)
    # Blue rectangle border matching skeleton model rectangle width
    cv2.rectangle(image, (x_start, y_start), (x_start+skel_box_w, y_start+table_height), TABLE_BORDER_COLOR, TABLE_BORDER_THICKNESS)
    cv2.putText(image, "Angles Table", (x_start+9, y_start+20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, TABLE_LETTER_COLOR, 1, cv2.LINE_AA)
    for idx, (label, val) in enumerate(zip(angle_labels, angles)):
        y = y_start+30+idx*30
        cv2.putText(image, f"{idx+1}: {val:.1f}", (x_start+11, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, TABLE_LETTER_COLOR, 1, cv2.LINE_AA)
    # Return new y for skeleton view placement
    return image, (x_start, y_start+table_height+5), skel_box_w, table_height

def draw_skeleton_view(box_pos, box_width, box_height, pts, blue=SKELETON_COLOR):
    img_skel = np.full((box_height, box_width, 4), SKELETON_BG, dtype=np.uint8)
    margin = SKELETON_MARGIN
    points_order = ['wrist', 'elbow', 'shoulder', 'hip', 'knee', 'ankle', 'heel', 'foot_index']
    raw_pts = np.array([pts[k] for k in points_order])
    minx, maxx = np.min(raw_pts[:,0]), np.max(raw_pts[:,0])
    miny, maxy = np.min(raw_pts[:,1]), np.max(raw_pts[:,1])
    width_span = maxx - minx
    height_span = maxy - miny
    scale = min((box_width-2*margin)/(width_span+1e-5), (box_height-2*margin)/(height_span+1e-5))
    offset_x = (box_width - scale*width_span)/2 - minx*scale
    offset_y = (box_height - scale*height_span)/2 - miny*scale
    def skel_pt(pt):
        return (int(pt[0]*scale + offset_x), int(pt[1]*scale + offset_y))
    skel_pts = {k: skel_pt(v) for k,v in pts.items()}
    # At least 5px from sides
    xs = [p[0] for p in skel_pts.values()]
    min_x = min(xs)
    max_x = max(xs)
    adjust_left = 5-min_x if min_x < 5 else 0
    adjust_right = (box_width-5)-max_x if max_x > (box_width-5) else 0
    for k in skel_pts:
        skel_pts[k] = (skel_pts[k][0]+adjust_left+adjust_right, skel_pts[k][1])
    # Draw lines from edge of circle to edge of next
    for idx in range(len(points_order)-1):
        k1 = points_order[idx]
        k2 = points_order[idx+1]
        edge1 = get_circle_edge_point(skel_pts[k1], skel_pts[k2], SKELETON_CIRCLE_RADIUS)
        edge2 = get_circle_edge_point(skel_pts[k2], skel_pts[k1], SKELETON_CIRCLE_RADIUS)
        cv2.line(img_skel, edge1, edge2, SKELETON_LINE_COLOR, SKELETON_LINE_THICKNESS, lineType=cv2.LINE_AA)
    for k in points_order:
        draw_circle_point(img_skel, skel_pts[k], blue_radius=SKELETON_CIRCLE_RADIUS_DRAW, white_radius=SKELETON_CIRCLE_RADIUS, color=SKELETON_CIRCLE_COLOR, white=True)
    # Draw arcs and numbers, all numbers outward
    draw_arc_outline_and_number(img_skel, skel_pts['wrist'], skel_pts['elbow'], skel_pts['shoulder'], SKELETON_LETTER_COLOR, 1, number_outside=True, font_scale=SKELETON_NUM_FONT_SCALE, font_thickness=SKELETON_NUM_FONT_THICKNESS, arc_thickness=1, npoints=34)
    draw_arc_outline_and_number(img_skel, skel_pts['elbow'], skel_pts['shoulder'], skel_pts['hip'], SKELETON_LETTER_COLOR, 2, number_outside=True, font_scale=SKELETON_NUM_FONT_SCALE, font_thickness=SKELETON_NUM_FONT_THICKNESS, arc_thickness=1, npoints=34)
    # Dashed arc and line for angle 3
    p2 = skel_pts['shoulder']
    p3 = skel_pts['hip']
    line_length = np.linalg.norm(np.array(p2) - np.array(p3))
    x3, y3 = p3
    pt_left = (int(x3 - line_length*0.35), y3)
    pt_right = (int(x3 + line_length*0.65), y3)
    draw_dashed_line(img_skel, pt_left, pt_right, SKELETON_LINE_COLOR, thickness=1, dash_length=7, gap_length=6)
    v1 = np.array(p2) - np.array(p3)
    v2 = np.array([1.0, 0.0])
    v1 = v1 / np.linalg.norm(v1)
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = 0
    dtheta = angle2 - angle1
    while dtheta <= -np.pi:
        angle2 += 2 * np.pi
        dtheta = angle2 - angle1
    while dtheta > np.pi:
        angle2 -= 2 * np.pi
        dtheta = angle2 - angle1
    if dtheta < 0:
        angle1, angle2 = angle2, angle1
    arc_radius = int(line_length / 2.2)
    draw_dashed_arc(img_skel, p3, arc_radius, angle1, angle2, SKELETON_LINE_COLOR, thickness=1, dash_length=5, gap_length=5)
    num3_radius = arc_radius * 1.25
    angle_mid = angle1 + (angle2 - angle1) * 0.5
    x = int(p3[0] + num3_radius * math.cos(angle_mid))
    y = int(p3[1] + num3_radius * math.sin(angle_mid))
    cv2.putText(img_skel, "3", (x, y), cv2.FONT_HERSHEY_SIMPLEX, SKELETON_NUM_FONT_SCALE, SKELETON_LETTER_COLOR, SKELETON_NUM_FONT_THICKNESS, cv2.LINE_AA)
    draw_arc_outline_and_number(img_skel, skel_pts['shoulder'], skel_pts['hip'], skel_pts['knee'], SKELETON_LETTER_COLOR, 4, number_outside=True, font_scale=SKELETON_NUM_FONT_SCALE, font_thickness=SKELETON_NUM_FONT_THICKNESS, arc_thickness=1, npoints=34)
    draw_arc_outline_and_number(img_skel, skel_pts['hip'], skel_pts['knee'], skel_pts['ankle'], SKELETON_LETTER_COLOR, 5, number_outside=True, font_scale=SKELETON_NUM_FONT_SCALE, font_thickness=SKELETON_NUM_FONT_THICKNESS, arc_thickness=1, npoints=34)
    # Blue rectangle border
    cv2.rectangle(img_skel, (0,0), (box_width-1,box_height-1), SKELETON_LINE_COLOR, 2)
    return img_skel

def process_image(image_path, side='right'):
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            landmark_dict = get_body_side_landmarks(results.pose_landmarks.landmark, side=side)
            h, w = image.shape[:2]
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
            # Draw main skeleton: blue joints, white outlines, white lines
            draw_points_and_lines(image, pts, points_order, white_radius=MAIN_CIRCLE_RADIUS, color=MAIN_LINE_COLOR, white=True, thickness=MAIN_LINE_THICKNESS)
            for k in points_order:
                cv2.circle(image, pts[k], MAIN_CIRCLE_FILL_RADIUS, MAIN_CIRCLE_COLOR, -1, lineType=cv2.LINE_AA)
            # Calculate angles for table and values
            angle1, _, _ = calc_angle_internal(pts['wrist'], pts['elbow'], pts['shoulder'])
            angle2, _, _ = calc_angle_internal(pts['elbow'], pts['shoulder'], pts['hip'])
            p2 = pts['shoulder']
            p3 = pts['hip']
            line_length = np.linalg.norm(np.array(p2) - np.array(p3))
            p_horizontal = (int(p3[0] + line_length), p3[1])
            angle3, _, _ = calc_angle_internal(p2, p3, p_horizontal)
            angle4, _, _ = calc_angle_internal(pts['shoulder'], pts['hip'], pts['knee'])
            angle5, _, _ = calc_angle_internal(pts['hip'], pts['knee'], pts['ankle'])
            # Draw blue filled arcs with white numbers at each main angle, except arc 3
            draw_filled_semiblue_arc(image, pts['wrist'], pts['elbow'], pts['shoulder'], number=1, alpha=0.5)
            draw_filled_semiblue_arc(image, pts['elbow'], pts['shoulder'], pts['hip'], number=2, alpha=0.5)
            # Arc 3: dashed outline, no fill, number outside
            pt_left = (int(p3[0] - line_length*0.35), p3[1])
            pt_right = (int(p3[0] + line_length*0.65), p3[1])
            draw_dashed_line(image, pt_left, pt_right, MAIN_ARC_COLOR, thickness=2, dash_length=12, gap_length=8)
            v1 = np.array(p2) - np.array(p3)
            v1 = v1 / np.linalg.norm(v1)
            angle1_3 = math.atan2(v1[1], v1[0])
            angle2_3 = 0
            dtheta = angle2_3 - angle1_3
            while dtheta <= -np.pi:
                angle2_3 += 2 * np.pi
                dtheta = angle2_3 - angle1_3
            while dtheta > np.pi:
                angle2_3 -= 2 * np.pi
                dtheta = angle2_3 - angle1_3
            if dtheta < 0:
                angle1_3, angle2_3 = angle2_3, angle1_3
            arc_radius = int(line_length / 2.2)
            draw_dashed_arc(image, p3, arc_radius, angle1_3, angle2_3, MAIN_ARC_COLOR, thickness=2, dash_length=8, gap_length=8)
            num3_radius = arc_radius * 1.15
            angle_mid = angle1_3 + (angle2_3 - angle1_3) * 0.5
            x_num3 = int(p3[0] + num3_radius * math.cos(angle_mid))
            y_num3 = int(p3[1] + num3_radius * math.sin(angle_mid))
            cv2.putText(image, "3", (x_num3, y_num3), cv2.FONT_HERSHEY_SIMPLEX, MAIN_NUM_FONT_SCALE, MAIN_LETTER_COLOR, MAIN_NUM_FONT_THICKNESS, cv2.LINE_AA)
            # Remaining arcs
            draw_filled_semiblue_arc(image, pts['shoulder'], pts['hip'], pts['knee'], number=4, alpha=0.5)
            draw_filled_semiblue_arc(image, pts['hip'], pts['knee'], pts['ankle'], number=5, alpha=0.5)
            angles = [angle1, angle2, angle3, angle4, angle5]
            angle_labels = [
                "Wrist-Elbow vs Elbow-Shoulder",
                "Shoulder-Elbow vs Shoulder-Hip",
                "Shoulder-Hip vs Hip-Horizontal",
                "Hip-Shoulder vs Hip-Knee",
                "Knee-Hip vs Knee-Ankle"
            ]
            # --- Resize image to width 1024, keep aspect ratio ---
            orig_h, orig_w = image.shape[:2]
            new_w = 1024
            scale = new_w / orig_w
            new_h = int(orig_h * scale)
            img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # --- Table and skeleton box position and dims ---
            skel_box_w = TABLE_WIDTH
            skel_box_h = TABLE_WIDTH * 2
            skel_box_x = 10
            skel_box_y = 10 + 30*len(angles) + 40 + 2  # Space for table above
            # Draw table with same width as skeleton box
            img_out, (skel_box_x, skel_box_y), _, _ = draw_text_and_table(
                img_resized, angles, angle_labels, skel_box_w, skel_box_x, 10, skel_box_w
            )
            # --- Skeleton view ---
            img_skel = draw_skeleton_view(
                (skel_box_x, skel_box_y), skel_box_w, skel_box_h,
                pts, blue=SKELETON_COLOR
            )
            # Place skeleton view in the image (RGBA over BGR)
            if img_out.shape[2] == 3:
                img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2BGRA)
            y1, y2 = skel_box_y, skel_box_y+skel_box_h
            x1, x2 = skel_box_x, skel_box_x+skel_box_w
            alpha_mask = img_skel[:,:,3]/255.0
            for c in range(3):
                img_out[y1:y2, x1:x2, c] = (
                    alpha_mask * img_skel[:,:,c] + (1-alpha_mask) * img_out[y1:y2, x1:x2, c]
                )
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGRA2BGR)
            return img_out
        else:
            print("No pose detected!")
            return image

def show_pose_image(image):
    cv2.imshow("Pose Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = "C:\\Users\\jan.zarnay\\Python\\Hyundai\\cycling\\cyclist_5.jpg"  # Replace with your image path
    out_image = process_image(img_path, side='right')
    cv2.imwrite("output_pose_blue_arcs_white_numbers_FINAL_TWEAKABLE.png", out_image)
    show_pose_image(out_image)
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import math
import uuid 

app = Flask(__name__)
CORS(app)

polls_db = {} 

# ==========================================
# REUSABLE VTON FUNCTIONS
# ==========================================
def overlay_transparent(background, overlay, x, y):
    h_bg, w_bg = background.shape[:2]
    h_ol, w_ol = overlay.shape[:2]
    y1, y2 = max(0, y), min(h_bg, y + h_ol)
    x1, x2 = max(0, x), min(w_bg, x + w_ol)
    y1_o, y2_o = max(0, -y), min(h_ol, h_bg - y)
    x1_o, x2_o = max(0, -x), min(w_ol, w_bg - x)
    if y1 >= y2 or x1 >= x2: return background
    overlay_rgb = overlay[y1_o:y2_o, x1_o:x2_o, :3]
    alpha_mask = overlay[y1_o:y2_o, x1_o:x2_o, 3] / 255.0
    alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
    background[y1:y2, x1:x2] = (alpha_mask * overlay_rgb + (1 - alpha_mask) * background[y1:y2, x1:x2]).astype(np.uint8)
    return background

def match_lighting(overlay_bgra, frame_bgr, target_x, target_y, target_w, target_h):
    h_bg, w_bg = frame_bgr.shape[:2]
    sample_y1, sample_y2 = max(0, target_y + int(target_h * 0.2)), min(h_bg, target_y + int(target_h * 0.5))
    sample_x1, sample_x2 = max(0, target_x + int(target_w * 0.3)), min(w_bg, target_x + int(target_w * 0.7))
    chest_region = frame_bgr[sample_y1:sample_y2, sample_x1:sample_x2]
    if chest_region.size == 0: return overlay_bgra
    gray_chest = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)
    room_brightness = np.mean(gray_chest)
    shirt_bgr, shirt_alpha = overlay_bgra[:, :, :3], overlay_bgra[:, :, 3]
    hsv_shirt = cv2.cvtColor(shirt_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_shirt)
    lighting_ratio = max(0.3, min(room_brightness / 170.0, 1.5))
    v = np.clip(v * lighting_ratio, 0, 255).astype(np.uint8)
    final_bgr = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    return np.dstack((final_bgr, shirt_alpha))

def load_shirt(filepath):
    img = cv2.imread(f"static/{filepath}", cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] != 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

# ==========================================
# INITIALIZE AI MODELS
# ==========================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

previous_shirt_corners = None

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    global previous_shirt_corners
    data = request.json
    
    raw_base64 = data.get('image')
    shirt_id = data.get('shirt')
    
    if not raw_base64 or not shirt_id:
        return jsonify({"success": False})

    image_data = raw_base64.split(",")[1] 
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = frame.shape

    current_shirt = load_shirt(shirt_id)
    if current_shirt is None:
        return jsonify({"success": False, "error": "Shirt not found"})
    orig_h, orig_w = current_shirt.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    hand_result = hands.process(rgb)
    
    active_gesture = None
    predicted_size = "Scanning..."

    # 1. Process Hand Gestures
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            index_mcp = hand_landmarks.landmark[5] 
            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]
            ring_tip = hand_landmarks.landmark[16]
            ring_pip = hand_landmarks.landmark[14]
            pinky_tip = hand_landmarks.landmark[20]
            pinky_pip = hand_landmarks.landmark[18]

            index_folded = index_tip.y > index_pip.y
            middle_folded = middle_tip.y > middle_pip.y
            ring_folded = ring_tip.y > ring_pip.y
            pinky_folded = pinky_tip.y > pinky_pip.y

            all_folded = index_folded and middle_folded and ring_folded and pinky_folded
            strict_open = (index_tip.y < index_pip.y - 0.04) and (middle_tip.y < middle_pip.y - 0.04) and (ring_tip.y < ring_pip.y - 0.04) and (pinky_tip.y < pinky_pip.y - 0.04)

            if all_folded and thumb_tip.y < hand_landmarks.landmark[3].y - 0.03:
                active_gesture = "thumbs_up"
            elif strict_open:
                active_gesture = "open_palm"
            elif not index_folded and middle_folded and ring_folded and pinky_folded:
                if index_tip.x < index_mcp.x - 0.01:
                    active_gesture = "point_left"
                elif index_tip.x > index_mcp.x + 0.01:
                    active_gesture = "point_right"
# ✨ 2. UPGRADED: Full-Body Torso & Dress Mapping
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        sl, sr = landmarks[12], landmarks[11] # Shoulders
        hl, hr = landmarks[24], landmarks[23] # Hips

        sl_x, sl_y = int(sl.x * w), int(sl.y * h)
        sr_x, sr_y = int(sr.x * w), int(sr.y * h)
        hl_x, hl_y = int(hl.x * w), int(hl.y * h)
        hr_x, hr_y = int(hr.x * w), int(hr.y * h)

        shoulder_width = math.sqrt((sr_x - sl_x)**2 + (sr_y - sl_y)**2)
        torso_length = math.sqrt((hl_x - sl_x)**2 + (hl_y - sl_y)**2)

        if shoulder_width > 0 and torso_length > 0:
            body_ratio = shoulder_width / torso_length
            if body_ratio > 0.65: predicted_size = "Large (Broad Fit)"
            elif body_ratio < 0.50: predicted_size = "Small (Slim Fit)"
            else: predicted_size = "Medium (Regular Fit)"

            # 1. Is it a Shirt or a Dress? (Check image aspect ratio)
            aspect_ratio = orig_h / orig_w
            is_dress = aspect_ratio > 1.3 # If it's tall, treat it as a dress

            # 2. Add padding to wrap AROUND the body, not just pin to joints
            pad_x = int(shoulder_width * 0.35) # Wider wrap for shoulders
            pad_y_top = int(shoulder_width * 0.25) # Go slightly above shoulders for collar

            # 3. Map the top corners to the shoulders
            top_left_x = sl_x - pad_x
            top_left_y = sl_y - pad_y_top
            top_right_x = sr_x + pad_x
            top_right_y = sr_y - pad_y_top

            # 4. Map the bottom corners down past the hips
            # If it's a dress, stretch it 1.8x the torso length down the legs
            length_multiplier = 1.8 if is_dress else 1.15
            
            bottom_left_x = hl_x - pad_x
            bottom_left_y = sl_y + int(torso_length * length_multiplier)
            bottom_right_x = hr_x + pad_x
            bottom_right_y = sr_y + int(torso_length * length_multiplier)

            raw_dst_pts = np.float32([
                [top_left_x, top_left_y], 
                [top_right_x, top_right_y],
                [bottom_left_x, bottom_left_y], 
                [bottom_right_x, bottom_right_y]
            ])

            # 5. Smooth the movement so it doesn't jitter
            if previous_shirt_corners is not None:
                alpha = 0.5 
                dst_pts = (raw_dst_pts * alpha) + (previous_shirt_corners * (1.0 - alpha))
                dst_pts = np.float32(dst_pts)
            else:
                dst_pts = raw_dst_pts
            previous_shirt_corners = dst_pts.copy()

            src_pts = np.float32([[0, 0], [orig_w, 0], [0, orig_h], [orig_w, orig_h]])
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Warp the clothing image to fit the new body box
            rotated_shirt = cv2.warpPerspective(current_shirt, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            # Calculate the bounding box for the lighting filter
            min_x = int(min(top_left_x, bottom_left_x))
            min_y = int(min(top_left_y, top_right_y))
            max_x = int(max(top_right_x, bottom_right_x))
            max_y = int(max(bottom_left_y, bottom_right_y))
            
            shirt_w = max_x - min_x
            shirt_h = max_y - min_y

            if shirt_w > 0 and shirt_h > 0:
                lit_shirt = match_lighting(rotated_shirt, frame, max(0, min_x), max(0, min_y), shirt_w, shirt_h)
                frame = overlay_transparent(frame, lit_shirt, 0, 0)

                
    _, buffer = cv2.imencode('.jpg', frame)
    result_img = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "success": True, 
        "image": result_img, 
        "size": predicted_size,
        "gesture": active_gesture
    })

@app.route('/create_poll', methods=['POST'])
def create_poll():
    data = request.json
    poll_id = str(uuid.uuid4())[:6].upper()
    polls_db[poll_id] = { "image": data['image'], "name": data['name'], "yes": 0, "no": 0 }
    return jsonify({"success": True, "poll_id": poll_id})

@app.route('/poll/<poll_id>')
def view_poll(poll_id):
    poll_data = polls_db.get(poll_id)
    if not poll_data: return "<h1>Poll not found or expired!</h1>", 404
    return render_template('poll.html', poll_id=poll_id, poll=poll_data)

if __name__ == '__main__':
    from waitress import serve
    print("=========================================")
    print(" 🚀 StyleAR Production Server is ACTIVE")
    print(" 🌐 Running on: http://127.0.0.1:5000")
    print("=========================================")
    serve(app, host="0.0.0.0", port=10000)
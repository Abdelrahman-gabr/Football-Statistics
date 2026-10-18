# import cv2
# import numpy as np
# from ultralytics import YOLO
# from sklearn.cluster import KMeans

# # Load your trained model weights
# model = YOLO("best.pt")

# video_path = "Milan.mp4"
# cap = cv2.VideoCapture(video_path)

# # Video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # Output video writer
# output_path = "C:/Users/gabra/Downloads/Football_Statistics/Milaaanoooo.mp4"
# out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# def get_clean_jersey_color(frame, box):
#     """Extracts upper torso and filters out grass (green) and extreme shadows/highlights."""
#     x1, y1, x2, y2 = map(int, box[:4])
#     h = y2 - y1
#     w = x2 - x1
    
#     # Target upper middle torso (jersey area)
#     y_start = y1 + int(0.15 * h)
#     y_end = y1 + int(0.55 * h)
#     x_start = x1 + int(0.25 * w)
#     x_end = x2 - int(0.25 * w)
    
#     y_start, y_end = max(0, y_start), min(frame.shape[0], y_end)
#     x_start, x_end = max(0, x_start), min(frame.shape[1], x_end)
    
#     if y_end <= y_start or x_end <= x_start:
#         return None
        
#     torso = frame[y_start:y_end, x_start:x_end]
#     hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    
#     # Mask out green grass pixels (Hue roughly between 35 and 85)
#     # Also mask out extreme dark/bright pixels to handle shadows better
#     h_channel = hsv_torso[:, :, 0]
#     s_channel = hsv_torso[:, :, 1]
#     v_channel = hsv_torso[:, :, 2]
    
#     # Non-grass and valid lighting mask
#     valid_mask = (h_channel < 35) | (h_channel > 85)
#     valid_mask &= (v_channel > 40) & (v_channel < 240)  # Filter extreme shadows/glare
    
#     if np.sum(valid_mask) > 10:
#         # Return mean of non-grass, lighting-normalized pixels (Focus more on Hue & Saturation)
#         mean_h = np.mean(h_channel[valid_mask])
#         mean_s = np.mean(s_channel[valid_mask])
#         return np.array([mean_h, mean_s]) # Using Hue and Saturation avoids shadow brightness bias
    
#     return None

# # --- Step 1: Collect robust color samples ---
# print("🔍 Collecting robust jersey color samples (ignoring grass/shadows)...")
# color_samples = []
# sample_frames = min(50, total_frames)  # Sample more frames for better distribution

# for _ in range(sample_frames):
#     ret, frame = cap.read()
#     if not ret: break
#     results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
#     if results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy()
        
#         for box, cls_id in zip(boxes, classes):
#             if model.names[int(cls_id)] == 'player':
#                 color_feat = get_clean_jersey_color(frame, box)
#                 if color_feat is not None:
#                     color_samples.append(color_feat)

# cap.release()

# kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
# if len(color_samples) > 15:
#     kmeans.fit(np.array(color_samples))
#     print("✅ Team color clusters successfully re-initialized with shadow-robust filtering!")
# else:
#     print("⚠️ Warning: Low sample count. Defaulting clusters.")

# # --- Step 2: Process Video ---
# print("🚀 Processing video with corrected clustering & vertical HUD...")
# cap = cv2.VideoCapture(video_path)
# track_team_cache = {}

# team_a_frames = 0
# team_b_frames = 0
# possession_threshold_px = 75

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
    
#     results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
    
#     ball_pos = None
#     players_data = []
    
#     if results[0].boxes.id is not None:
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         track_ids = results[0].boxes.id.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy()
        
#         for box, track_id, cls_id in zip(boxes, track_ids, classes):
#             class_name = model.names[int(cls_id)]
#             x1, y1, x2, y2 = map(int, box[:4])
            
#             if class_name == 'player':
#                 if track_id in track_team_cache:
#                     team = track_team_cache[track_id]
#                 else:
#                     color_feat = get_clean_jersey_color(frame, box)
#                     if color_feat is not None:
#                         team = kmeans.predict(color_feat.reshape(1, -1))[0]
#                         track_team_cache[track_id] = team
#                     else:
#                         team = 0
                
#                 # Colors & Labels (Team A: Blue, Team B: Red)
#                 color = (255, 0, 0) if team == 0 else (0, 0, 255)
#                 label = f"A #{int(track_id)}" if team == 0 else f"B #{int(track_id)}"
                
#                 center_x = (x1 + x2) / 2
#                 center_y = y2
#                 players_data.append((team, center_x, center_y))
                
#             elif class_name == 'goalkeeper':
#                 color = (0, 255, 255)  # Yellow
#                 label = f"GK #{int(track_id)}"
#             elif class_name == 'referee':
#                 color = (255, 255, 255)  # White
#                 label = f"Ref #{int(track_id)}"
#             else:  # ball
#                 color = (0, 255, 0)  # Green
#                 label = "Ball"
#                 ball_pos = ((x1 + x2) / 2, (y1 + y2) / 2)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, y1 - 6), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

#     # Ball Possession Calculation
#     if ball_pos is not None and len(players_data) > 0:
#         bx, by = ball_pos
#         min_dist = float('inf')
#         possessing_team = None
        
#         for team, px, py in players_data:
#             dist = np.sqrt((bx - px)**2 + (by - py)**2)
#             if dist < min_dist:
#                 min_dist = dist
#                 possessing_team = team
                
#         if min_dist < possession_threshold_px:
#             if possessing_team == 0:
#                 team_a_frames += 1
#             else:
#                 team_b_frames += 1

#     total_poss_frames = team_a_frames + team_b_frames
#     if total_poss_frames > 0:
#         team_a_pct = int((team_a_frames / total_poss_frames) * 100)
#         team_b_pct = 100 - team_a_pct
#     else:
#         team_a_pct, team_b_pct = 50, 50

#     # --- Large & Bold Top-Right HUD Overlay ---
#     hud_w = 340
#     hud_h = 120
#     hud_x = width - hud_w - 30
#     hud_y = 30
    
#     # Dark solid background with thick bold border
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
#     cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (255, 255, 255), 3)
    
#     # HUD Title
#     cv2.putText(frame, "MATCH POSSESSION", (hud_x + 15, hud_y + 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
#     # Team A Row (Blue indicator box + Bold text)
#     cv2.rectangle(frame, (hud_x + 15, hud_y + 48), (hud_x + 35, hud_y + 72), (255, 0, 0), -1)
#     cv2.putText(frame, f"Team A: {team_a_pct}%", (hud_x + 45, hud_y + 68), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
#     # Team B Row (Red indicator box + Bold text)
#     cv2.rectangle(frame, (hud_x + 15, hud_y + 82), (hud_x + 35, hud_y + 106), (0, 0, 255), -1)
#     cv2.putText(frame, f"Team B: {team_b_pct}%", (hud_x + 45, hud_y + 102), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

#     out.write(frame)

# cap.release()
# out.release()
# print(f"✅ Video successfully processed and saved to {output_path}")







import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Load your trained model weights
model = YOLO("best.pt")

video_path = "world_cup.mp4"
cap = cv2.VideoCapture(video_path)

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video writer
output_path = "C:/Users/gabra/Downloads/Football_Statistics/wcf.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

def get_clean_jersey_color(frame, box):
    """Extracts upper torso and filters out grass (green) and extreme shadows/highlights."""
    x1, y1, x2, y2 = map(int, box[:4])
    h = y2 - y1
    w = x2 - x1
    
    y_start = y1 + int(0.15 * h)
    y_end = y1 + int(0.55 * h)
    x_start = x1 + int(0.25 * w)
    x_end = x2 - int(0.25 * w)
    
    y_start, y_end = max(0, y_start), min(frame.shape[0], y_end)
    x_start, x_end = max(0, x_start), min(frame.shape[1], x_end)
    
    if y_end <= y_start or x_end <= x_start:
        return None
        
    torso = frame[y_start:y_end, x_start:x_end]
    hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    
    h_channel = hsv_torso[:, :, 0]
    s_channel = hsv_torso[:, :, 1]
    v_channel = hsv_torso[:, :, 2]
    
    valid_mask = (h_channel < 35) | (h_channel > 85)
    valid_mask &= (v_channel > 40) & (v_channel < 240) 
    
    if np.sum(valid_mask) > 10:
        mean_h = np.mean(h_channel[valid_mask])
        mean_s = np.mean(s_channel[valid_mask])
        return np.array([mean_h, mean_s])
    
    return None

# --- Step 1: Collect robust color samples ---
print("🔍 Collecting robust jersey color samples...")
color_samples = []
sample_frames = min(50, total_frames)

for _ in range(sample_frames):
    ret, frame = cap.read()
    if not ret: break
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        for box, cls_id in zip(boxes, classes):
            if model.names[int(cls_id)] == 'player':
                color_feat = get_clean_jersey_color(frame, box)
                if color_feat is not None:
                    color_samples.append(color_feat)

cap.release()

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
if len(color_samples) > 15:
    kmeans.fit(np.array(color_samples))
    print("✅ Team color clusters initialized successfully!")
else:
    print("⚠️ Warning: Low sample count. Defaulting clusters.")

# --- Step 2: Process Video with Sticky Possession ---
print("🚀 Processing video with Sticky Possession & Large HUD...")
cap = cv2.VideoCapture(video_path)
track_team_cache = {}

team_a_frames = 0
team_b_frames = 0
possession_threshold_px = 85  # Slightly wider catchment area for loose balls
current_possessing_team = None  # Remembers the last team to touch the ball

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
    
    ball_pos = None
    players_data = []
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        for box, track_id, cls_id in zip(boxes, track_ids, classes):
            class_name = model.names[int(cls_id)]
            x1, y1, x2, y2 = map(int, box[:4])
            
            if class_name == 'player':
                if track_id in track_team_cache:
                    team = track_team_cache[track_id]
                else:
                    color_feat = get_clean_jersey_color(frame, box)
                    if color_feat is not None:
                        team = kmeans.predict(color_feat.reshape(1, -1))[0]
                        track_team_cache[track_id] = team
                    else:
                        team = 0
                
                color = (255, 0, 0) if team == 0 else (0, 0, 255)
                label = f"Team A #{int(track_id)}" if team == 0 else f"Team B #{int(track_id)}"
                
                center_x = (x1 + x2) / 2
                center_y = y2
                players_data.append((team, center_x, center_y))
                
            elif class_name == 'goalkeeper':
                color = (0, 255, 255)  # Yellow
                label = f"GK #{int(track_id)}"
            elif class_name == 'referee':
                color = (255, 255, 255)  # White
                label = f"Ref #{int(track_id)}"
            else:  # ball
                color = (0, 255, 0)  # Green
                label = "Ball"
                ball_pos = ((x1 + x2) / 2, (y1 + y2) / 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # --- Sticky Possession Logic ---
    if ball_pos is not None and len(players_data) > 0:
        bx, by = ball_pos
        min_dist = float('inf')
        closest_team = None
        
        for team, px, py in players_data:
            dist = np.sqrt((bx - px)**2 + (by - py)**2)
            if dist < min_dist:
                min_dist = dist
                closest_team = team
                
        # If a player is close enough, update sticky possession to this team
        if min_dist < possession_threshold_px:
            current_possessing_team = closest_team

    # Increment frame count for whichever team currently holds sticky possession
    if current_possessing_team == 0:
        team_a_frames += 1
    elif current_possessing_team == 1:
        team_b_frames += 1

    total_poss_frames = team_a_frames + team_b_frames
    if total_poss_frames > 0:
        team_a_pct = int((team_a_frames / total_poss_frames) * 100)
        team_b_pct = 100 - team_a_pct
    else:
        team_a_pct, team_b_pct = 50, 50

    # --- Large & Bold Top-Right HUD Overlay ---
    hud_w = 340
    hud_h = 120
    hud_x = width - hud_w - 30
    hud_y = 30
    
    # Dark solid background with thick bold border
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (255, 255, 255), 3)
    
    # HUD Title
    cv2.putText(frame, "MATCH POSSESSION", (hud_x + 15, hud_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Team A Row (Blue indicator box + Bold text)
    cv2.rectangle(frame, (hud_x + 15, hud_y + 48), (hud_x + 35, hud_y + 72), (255, 0, 0), -1)
    cv2.putText(frame, f"Team A: {team_a_pct}%", (hud_x + 45, hud_y + 68), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
    # Team B Row (Red indicator box + Bold text)
    cv2.rectangle(frame, (hud_x + 15, hud_y + 82), (hud_x + 35, hud_y + 106), (0, 0, 255), -1)
    cv2.putText(frame, f"Team B: {team_b_pct}%", (hud_x + 45, hud_y + 102), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    out.write(frame)

cap.release()
out.release()
print(f"✅ Video successfully processed and saved to {output_path}")
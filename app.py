import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PitchIQ AI | Elite Soccer Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .saas-card {
        background-color: #162238;
        border: 1px solid #26354D;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #162238;
        border: 1px solid #26354D;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #3B82F6;
    }
    .metric-label {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: #F8FAFC;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #2563EB;
        color: #ffffff;
    }
    [data-testid="stFileUploader"] {
        background-color: #162238;
        border: 2px dashed #26354D;
        border-radius: 12px;
        padding: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")


try:
    model = load_yolo_model()
except Exception as e:
    st.error(
        f"⚠️ Critical Error loading model weights (`best.pt`). Please ensure `best.pt` is in the same directory as `app.py`. Details: {e}"
    )


# -----------------------------------------------------------------------------
# 3. COMPUTER VISION & STICKY POSSESSION LOGIC
# -----------------------------------------------------------------------------
def get_clean_jersey_color(frame, box):
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


def process_football_video(input_video_path, output_video_path, progress_bar, status_text):
    # Use WebM container with VP80 codec for safe browser compatibility without extra DLLs
    if output_video_path.endswith(".mp4"):
        output_video_path = output_video_path.replace(".mp4", ".webm")

    cap = cv2.VideoCapture(input_video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 30, (width, height))
    
    if not out.isOpened():
        # Fallback to mp4v if VP80 fails
        output_video_path = output_video_path.replace(".webm", ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 30, (width, height))

    status_text.text("🔍 Phase 1/2: Collecting robust jersey color samples...")
    color_samples = []
    sample_frames = min(50, total_frames)

    for _ in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            for box, cls_id in zip(boxes, classes):
                if model.names[int(cls_id)] == "player":
                    color_feat = get_clean_jersey_color(frame, box)
                    if color_feat is not None:
                        color_samples.append(color_feat)

    cap.release()

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    if len(color_samples) > 15:
        kmeans.fit(np.array(color_samples))

    cap = cv2.VideoCapture(input_video_path)
    track_team_cache = {}

    team_a_frames = 0
    team_b_frames = 0
    possession_threshold_px = 85
    current_possessing_team = None

    frame_idx = 0
    status_text.text("🚀 Phase 2/2: Processing video with Sticky Possession & HUD overlay...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        progress_bar.progress(min(float(frame_idx) / float(total_frames), 1.0))

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

                if class_name == "player":
                    if track_id in track_team_cache:
                        team = track_team_cache[track_id]
                    else:
                        color_feat = get_clean_jersey_color(frame, box)
                        if color_feat is not None and len(color_samples) > 15:
                            team = kmeans.predict(color_feat.reshape(1, -1))[0]
                            track_team_cache[track_id] = team
                        else:
                            team = 0

                    color = (255, 0, 0) if team == 0 else (0, 0, 255)
                    label = f"Team A #{int(track_id)}" if team == 0 else f"Team B #{int(track_id)}"
                    players_data.append((team, (x1 + x2) / 2, y2))

                elif class_name == "goalkeeper":
                    color = (0, 255, 255)
                    label = f"GK #{int(track_id)}"
                elif class_name == "referee":
                    color = (255, 255, 255)
                    label = f"Ref #{int(track_id)}"
                else:
                    color = (0, 255, 0)
                    label = "Ball"
                    ball_pos = ((x1 + x2) / 2, (y1 + y2) / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Sticky Possession Logic
        if ball_pos is not None and len(players_data) > 0:
            bx, by = ball_pos
            min_dist = float("inf")
            closest_team = None
            for team, px, py in players_data:
                dist = np.sqrt((bx - px) ** 2 + (by - py) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_team = team
            if min_dist < possession_threshold_px:
                current_possessing_team = closest_team

        if current_possessing_team == 0:
            team_a_frames += 1
        elif current_possessing_team == 1:
            team_b_frames += 1

        total_poss = team_a_frames + team_b_frames
        team_a_pct = int((team_a_frames / total_poss) * 100) if total_poss > 0 else 50
        team_b_pct = 100 - team_a_pct

        # HUD Overlay
        hud_w, hud_h = 340, 120
        hud_x, hud_y = width - hud_w - 30, 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (255, 255, 255), 3)

        cv2.putText(frame, "MATCH POSSESSION", (hud_x + 15, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (hud_x + 15, hud_y + 48), (hud_x + 35, hud_y + 72), (255, 0, 0), -1)
        cv2.putText(frame, f"Team A: {team_a_pct}%", (hud_x + 45, hud_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (hud_x + 15, hud_y + 82), (hud_x + 35, hud_y + 106), (0, 0, 255), -1)
        cv2.putText(frame, f"Team B: {team_b_pct}%", (hud_x + 45, hud_y + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    return team_a_pct, team_b_pct, output_video_path


# -----------------------------------------------------------------------------
# 4. SINGLE-PAGE CLEAN MODERN INTERFACE
# -----------------------------------------------------------------------------
st.markdown("# ⚽ PitchIQ Analytics Suite")
st.markdown(
    "<p style='color: #94A3B8; font-size: 16px;'>Professional computer vision suite for multi-object tracking, shadow-robust team color clustering, and live sticky possession metrics.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

col_upload, col_display = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="saas-card"><h3>📁 Match Video Input</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload footage (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        st.success(f"Loaded: **{uploaded_file.name}** ({round(uploaded_file.size / (1024*1024), 2)} MB)")
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        input_video_path = tfile.name

        if st.button("🚀 Run AI Tracking & Possession Pipeline"):
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
            output_video_path = output_temp.name

            st.markdown("---")
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                ta_res, tb_res, final_vid_path = process_football_video(
                    input_video_path, output_video_path, progress_bar, status_text
                )
                status_text.text("✅ Processing complete successfully!")
                st.session_state["processed_video"] = final_vid_path
                st.session_state["team_a_res"] = ta_res
                st.session_state["team_b_res"] = tb_res
                st.rerun()
            except Exception as ex:
                st.error(f"Inference error encountered: {ex}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="saas-card">
            <h3>⚙️ Engine Specifications</h3>
            <div style="color: #94A3B8; font-size: 14px; line-height: 1.6;">
                • <b>Backbone:</b> YOLOv11 Custom Trained <br>
                • <b>Tracker:</b> ByteTrack Multi-Object Engine<br>
                • <b>Clustering:</b> HSV Shadow-Masked K-Means<br>
                • <b>Possession Logic:</b> Sticky Last-Touch Frame Attribution
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with col_display:
    st.markdown('<div class="saas-card"><h3>📊 Live Intelligence & Results</h3>', unsafe_allow_html=True)

    if "processed_video" in st.session_state:
        st.markdown("#### Annotated Video Output")
        st.video(st.session_state["processed_video"])

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{st.session_state.get("team_a_res", 50)}%</div><div class="metric-label">Team A Possession</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{st.session_state.get("team_b_res", 50)}%</div><div class="metric-label">Team B Possession</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        with open(st.session_state["processed_video"], "rb") as f:
            st.download_button(
                label="📥 Download Processed Match Video",
                data=f,
                file_name="PitchIQ_Analyzed_Match.webm",
                mime="video/webm",
            )
    else:
        st.info("Awaiting video upload and execution. Upload your match clip to generate tracking overlays and analytics.")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">--%</div><div class="metric-label">Team A Possession</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                '<div class="metric-card"><div class="metric-value">--%</div><div class="metric-label">Team B Possession</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
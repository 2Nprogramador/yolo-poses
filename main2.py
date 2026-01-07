import streamlit as st
import cv2
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


# ==========================================
# 1. Thresholds
# ==========================================

def get_thresholds_beginner():
    return {
        'NORMAL': (0, 32),
        'TRANS': (35, 65),
        'PASS': (70, 95),
        'TOO_LOW': 95
    }


def get_thresholds_pro():
    return {
        'NORMAL': (0, 32),
        'TRANS': (35, 65),
        'PASS': (80, 95),
        'TOO_LOW': 95
    }


# ==========================================
# 2. Funções Matemáticas
# ==========================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def calculate_vertical_angle(hip, knee):
    vertical_point = [knee[0], knee[1] - 100]
    return calculate_angle(hip, knee, vertical_point)


# ==========================================
# 3. Desenho do Esqueleto
# ==========================================

def draw_pose_landmarks(frame, landmarks, w, h):
    connections = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 12),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28)
    ]

    for s, e in connections:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


# ==========================================
# 4. Streamlit UI
# ==========================================

st.set_page_config(page_title="Visão computacional Agachamento", layout="wide")
st.title("🏋️ Visão computacional Agachamento")

mode = st.sidebar.radio("Nível:", ["Iniciante", "Pro"])
limits = get_thresholds_beginner() if mode == "Iniciante" else get_thresholds_pro()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "gravando4.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

run_analysis = st.sidebar.button("Iniciar Análise")

# memória de estado (histerese)
if "last_state" not in st.session_state:
    st.session_state.last_state = "EM PE"


# ==========================================
# 5. MediaPipe Tasks
# ==========================================

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False
)


# ==========================================
# 6. Processamento
# ==========================================

if run_analysis:
    if not os.path.exists(VIDEO_PATH):
        st.error("❌ Vídeo não encontrado.")
    elif not os.path.exists(MODEL_PATH):
        st.error("❌ Modelo .task não encontrado.")
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        detector = vision.PoseLandmarker.create_from_options(options)

        stframe = st.empty()
        kpi1, kpi2, kpi3 = st.columns(3)

        timestamp = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            result = detector.detect_for_video(mp_image, timestamp)
            timestamp += 33

            vertical_angle = 0
            current_state = st.session_state.last_state

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_pose_landmarks(frame, lm, w, h)

                hip = [lm[23].x * w, lm[23].y * h]
                knee = [lm[25].x * w, lm[25].y * h]

                vertical_angle = calculate_vertical_angle(hip, knee)

                # -------- LÓGICA CORRIGIDA --------
                if vertical_angle <= limits['NORMAL'][1]:
                    current_state = "EM PE"

                elif limits['TRANS'][0] <= vertical_angle <= limits['TRANS'][1]:
                    current_state = "DESCENDO"

                elif limits['PASS'][0] <= vertical_angle <= limits['PASS'][1]:
                    current_state = "AGACHAMENTO OK"

                elif vertical_angle > limits['TOO_LOW']:
                    current_state = "MUITO BAIXO"

                # senão: mantém o estado anterior (histerese)
                st.session_state.last_state = current_state
                # ----------------------------------

                color = {
                    "EM PE": (0, 255, 255),
                    "DESCENDO": (255, 165, 0),
                    "AGACHAMENTO OK": (0, 255, 0),
                    "MUITO BAIXO": (0, 0, 255)
                }.get(current_state, (255, 255, 255))

                cv2.rectangle(frame, (0, 0), (360, 100), (0, 0, 0), -1)
                cv2.putText(frame, current_state, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(frame, f"Angulo: {int(vertical_angle)}",
                            (200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

            stframe.image(frame, channels="BGR", use_container_width=True)
            kpi1.metric("Ângulo Vertical", f"{int(vertical_angle)}°")
            kpi2.metric("Estado", current_state)
            kpi3.metric("Modo", mode)

        cap.release()
        detector.close()

else:
    st.info(f"📹 Vídeo configurado: {VIDEO_PATH}")

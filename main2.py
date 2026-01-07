import streamlit as st
import cv2
import numpy as np
import os
import tempfile

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ==========================================
# 1. Configurações
# ==========================================

st.set_page_config(page_title="Processamento Offline - Agachamento", layout="wide")
st.title("🏋️ Análise Completa de Agachamento")
st.markdown("Este modo processa o vídeo inteiro primeiro para garantir **reprodução fluida** no final.")

# Funções de Threshold (Mantidas)
def get_thresholds_beginner():
    return {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (70, 95), 'TOO_LOW': 95}

def get_thresholds_pro():
    return {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (80, 95), 'TOO_LOW': 95}

# ==========================================
# 2. Funções Matemáticas e Desenho
# ==========================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def calculate_vertical_angle(hip, knee):
    vertical_point = [knee[0], knee[1] - 100]
    return calculate_angle(hip, knee, vertical_point)

def draw_pose_landmarks(frame, landmarks, w, h):
    connections = [
        (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
    ]
    for s, e in connections:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

# ==========================================
# 3. Interface e Execução
# ==========================================

st.sidebar.header("Configurações")
mode = st.sidebar.radio("Nível:", ["Iniciante", "Pro"])
limits = get_thresholds_beginner() if mode == "Iniciante" else get_thresholds_pro()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "gravando4.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_final.mp4")

run_analysis = st.sidebar.button("⚙️ Processar Vídeo Completo")

if "last_state" not in st.session_state:
    st.session_state.last_state = "EM PE"

if run_analysis:
    if not os.path.exists(VIDEO_PATH) or not os.path.exists(MODEL_PATH):
        st.error("Erro: Arquivos de vídeo ou modelo não encontrados.")
    else:
        # Configuração do Modelo
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        # Configuração de Vídeo
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        # Propriedades do vídeo original
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Redimensionamento (Mantendo 640px de largura para processar rápido)
        target_width = 640
        scale = target_width / width_orig
        target_height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

        # Configuração do Gravador (VideoWriter)
        # mp4v é um codec genérico que costuma funcionar bem
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (target_width, target_height))

        # Barra de Progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        timestamp_ms = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Resize
            frame = cv2.resize(frame, (target_width, target_height))
            h, w, _ = frame.shape

            # 2. MediaPipe Detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect_for_video(mp_image, int(timestamp_ms))
            timestamp_ms += (1000.0 / fps)

            # 3. Lógica de Negócio (Ângulos e Estados)
            vertical_angle = 0
            current_state = st.session_state.last_state

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_pose_landmarks(frame, lm, w, h)

                hip = [lm[23].x * w, lm[23].y * h]
                knee = [lm[25].x * w, lm[25].y * h]
                vertical_angle = calculate_vertical_angle(hip, knee)

                if vertical_angle <= limits['NORMAL'][1]: current_state = "EM PE"
                elif limits['TRANS'][0] <= vertical_angle <= limits['TRANS'][1]: current_state = "DESCENDO"
                elif limits['PASS'][0] <= vertical_angle <= limits['PASS'][1]: current_state = "AGACHAMENTO OK"
                elif vertical_angle > limits['TOO_LOW']: current_state = "MUITO BAIXO"

                st.session_state.last_state = current_state

                # Desenho do Overlay
                color = {"EM PE": (0, 255, 255), "DESCENDO": (255, 165, 0), "AGACHAMENTO OK": (0, 255, 0), "MUITO BAIXO": (0, 0, 255)}.get(current_state, (255, 255, 255))
                cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(frame, f"{current_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"{int(vertical_angle)} deg", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 4. Gravar Frame no arquivo final
            out.write(frame)

            # Atualizar barra de progresso
            frame_count += 1
            if total_frames > 0:
                prog = min(frame_count / total_frames, 1.0)
                progress_bar.progress(prog)
                status_text.text(f"Processando frame {frame_count}/{total_frames}...")

        # Finalização
        cap.release()
        out.release()
        detector.close()
        
        status_text.text("Processamento concluído! Carregando player...")
        progress_bar.empty()

        # 5. Exibir o vídeo final
        # Precisamos ler o arquivo binário para o Streamlit exibir corretamente
        if os.path.exists(OUTPUT_PATH):
            st.success("Vídeo processado com sucesso!")
            st.video(OUTPUT_PATH)
        else:
            st.error("Erro ao salvar o vídeo processado.")

else:
    st.info("Clique no botão na barra lateral para iniciar o processamento completo.")

import streamlit as st
import cv2
import numpy as np
import os
import time  # Importante para controlar a velocidade do vídeo

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ==========================================
# 1. Configurações e Thresholds
# ==========================================

st.set_page_config(page_title="Visão Computacional - Agachamento", layout="wide")

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
# 2. Funções Matemáticas e Visuais
# ==========================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def calculate_vertical_angle(hip, knee):
    # Cria um ponto virtual verticalmente abaixo do joelho para medir a flexão
    vertical_point = [knee[0], knee[1] - 100]
    return calculate_angle(hip, knee, vertical_point)

def draw_pose_landmarks(frame, landmarks, w, h):
    # Conexões principais para desenhar o esqueleto
    connections = [
        (11, 13), (13, 15), # Braço esquerdo
        (12, 14), (14, 16), # Braço direito
        (11, 12),           # Ombros
        (11, 23), (12, 24), # Tronco
        (23, 24),           # Quadril
        (23, 25), (25, 27), # Perna esquerda
        (24, 26), (26, 28)  # Perna direita
    ]

    # Desenha linhas
    for s, e in connections:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Desenha pontos
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

# ==========================================
# 3. Interface Streamlit
# ==========================================

st.title("🏋️ Visão Computacional: Análise de Agachamento")

# Sidebar
st.sidebar.header("Configurações")
mode = st.sidebar.radio("Nível de Dificuldade:", ["Iniciante", "Pro"])
limits = get_thresholds_beginner() if mode == "Iniciante" else get_thresholds_pro()

run_analysis = st.sidebar.button("▶️ Iniciar Análise")

# Caminhos dos arquivos (certifique-se que estão na mesma pasta ou ajuste aqui)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "gravando4.mp4")
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

# Inicializa estado se não existir
if "last_state" not in st.session_state:
    st.session_state.last_state = "EM PE"

# ==========================================
# 4. Lógica Principal (Pipeline)
# ==========================================

if run_analysis:
    # Verificações de segurança
    if not os.path.exists(VIDEO_PATH):
        st.error(f"❌ Vídeo não encontrado: {VIDEO_PATH}")
    elif not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modelo não encontrado: {MODEL_PATH}")
    else:
        # Configuração do MediaPipe
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False
        )

        try:
            detector = vision.PoseLandmarker.create_from_options(options)
            cap = cv2.VideoCapture(VIDEO_PATH)

            # Informações do vídeo para sincronia
            fps_original = cap.get(cv2.CAP_PROP_FPS)
            # Se não conseguir ler o FPS, assume 30
            if fps_original <= 0: fps_original = 30.0
            
            frame_duration = 1.0 / fps_original

            # Placeholders da interface
            stframe = st.empty()
            col1, col2, col3 = st.columns(3)
            
            # Métricas iniciais
            kpi_angle = col1.empty()
            kpi_state = col2.empty()
            kpi_mode = col3.empty()
            kpi_mode.metric("Modo Selecionado", mode)

            timestamp_ms = 0 # MediaPipe usa milissegundos

            while cap.isOpened():
                start_process_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                # --- 1. REDIMENSIONAMENTO (CRÍTICO PARA PERFORMANCE) ---
                h_orig, w_orig, _ = frame.shape
                target_width = 640
                scale_factor = target_width / w_orig
                target_height = int(h_orig * scale_factor)
                
                frame = cv2.resize(frame, (target_width, target_height))
                h, w, _ = frame.shape
                # -------------------------------------------------------

                # Prepara imagem para o MediaPipe
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                # Detecção
                result = detector.detect_for_video(mp_image, int(timestamp_ms))
                
                # Incrementa timestamp baseado no FPS do vídeo
                timestamp_ms += (1000.0 / fps_original)

                # Variáveis de lógica
                vertical_angle = 0
                current_state = st.session_state.last_state

                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    
                    # Desenha esqueleto
                    draw_pose_landmarks(frame, lm, w, h)

                    # Pega coordenadas (já ajustadas para o novo tamanho w, h)
                    hip = [lm[23].x * w, lm[23].y * h]
                    knee = [lm[25].x * w, lm[25].y * h]

                    vertical_angle = calculate_vertical_angle(hip, knee)

                    # --- LÓGICA DE ESTADOS ---
                    if vertical_angle <= limits['NORMAL'][1]:
                        current_state = "EM PE"
                    elif limits['TRANS'][0] <= vertical_angle <= limits['TRANS'][1]:
                        current_state = "DESCENDO"
                    elif limits['PASS'][0] <= vertical_angle <= limits['PASS'][1]:
                        current_state = "AGACHAMENTO OK"
                    elif vertical_angle > limits['TOO_LOW']:
                        current_state = "MUITO BAIXO"
                    
                    st.session_state.last_state = current_state

                    # Define cores baseadas no estado
                    color_map = {
                        "EM PE": (0, 255, 255),       # Amarelo
                        "DESCENDO": (255, 165, 0),    # Laranja
                        "AGACHAMENTO OK": (0, 255, 0),# Verde
                        "MUITO BAIXO": (0, 0, 255)    # Vermelho
                    }
                    color = color_map.get(current_state, (255, 255, 255))

                    # Desenha infos na tela (Overlay)
                    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
                    cv2.putText(frame, f"Status: {current_state}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Ang: {int(vertical_angle)}", (w - 150, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Atualiza Interface
                stframe.image(frame, channels="BGR", use_container_width=True)
                kpi_angle.metric("Ângulo Joelho", f"{int(vertical_angle)}°")
                kpi_state.metric("Estado Atual", current_state)

                # --- 2. CONTROLE DE FPS (Sincronia) ---
                processing_time = time.time() - start_process_time
                wait_time = frame_duration - processing_time
                if wait_time > 0:
                    time.sleep(wait_time)
                # --------------------------------------

            cap.release()
            detector.close()
            st.success("Análise do vídeo concluída!")

        except Exception as e:
            st.error(f"Ocorreu um erro durante o processamento: {e}")

else:
    st.info("Aguardando início. Certifique-se que o arquivo 'gravando4.mp4' e o modelo '.task' estão no diretório.")

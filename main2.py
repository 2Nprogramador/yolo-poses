import streamlit as st
import cv2
import numpy as np
import os
import tempfile

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ==========================================
# 1. Configurações e Layout
# ==========================================

st.set_page_config(page_title="Visão Computacional - 33 pontos", layout="wide")
st.markdown("""
    <h1>Análise de Exercícios<br>com Visão Computacional</h1>
    """, unsafe_allow_html=True)
st.markdown("""
Faça o upload do seu vídeo MP4  
Clique em Processar Vídeo  
Obtenha a Análise Completa
""")

# Funções de Threshold (Mantidas para o Agachamento Padrão)
def get_thresholds_beginner():
    return {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (70, 95), 'TOO_LOW': 95}

def get_thresholds_pro():
    return {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (80, 95), 'TOO_LOW': 95}

# ==========================================
# 2. Funções Matemáticas
# ==========================================

def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos (a, b, c). b é o vértice."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_vertical_angle(hip, knee):
    """Calcula o ângulo do fêmur em relação à vertical (usado no agachamento padrão)."""
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
# 3. Interface e Upload
# ==========================================

st.sidebar.header("Configurações")

# --- MUDANÇA: Seleção do Exercício ---
exercise_type = st.sidebar.selectbox(
    "Escolha o Exercício:", 
    ["Agachamento Padrão", "Agachamento Búlgaro"]
)

mode = st.sidebar.radio("Nível:", ["Iniciante", "Pro"])
limits = get_thresholds_beginner() if mode == "Iniciante" else get_thresholds_pro()

uploaded_file = st.sidebar.file_uploader("Carregar Vídeo (MP4/MOV)", type=["mp4", "mov", "avi"])

# Paths Locais
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_final.webm")

# Variável para controlar qual vídeo processar
video_source_path = None

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_source_path = tfile.name
    st.sidebar.success("Vídeo carregado!")
else:
    default_video = os.path.join(BASE_DIR, "gravando4.mp4")
    if os.path.exists(default_video):
        video_source_path = default_video
        st.sidebar.info("Usando vídeo padrão (nenhum upload detectado).")

run_analysis = st.sidebar.button("⚙️ Processar Vídeo")

if "last_state" not in st.session_state:
    st.session_state.last_state = "EM PE"

# ==========================================
# 4. Execução do Processamento
# ==========================================

if run_analysis and video_source_path:
    if not os.path.exists(MODEL_PATH):
        st.error("Erro: Modelo MediaPipe (.task) não encontrado na pasta.")
    else:
        # Configuração do Modelo
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(video_source_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        if width_orig == 0:
            st.error("Erro ao ler o vídeo. O arquivo pode estar corrompido.")
            st.stop()

        target_width = 640
        scale = target_width / width_orig
        target_height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

        fourcc = cv2.VideoWriter_fourcc(*'vp80') 
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (target_width, target_height))

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        timestamp_ms = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (target_width, target_height))
            h, w, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect_for_video(mp_image, int(timestamp_ms))
            timestamp_ms += (1000.0 / fps)

            # Inicializa variáveis
            display_angle = 0
            current_state = st.session_state.last_state
            warning_msg = "" # Para alertas extras (Ex: Tronco caindo)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_pose_landmarks(frame, lm, w, h)

                # ==========================================
                # LÓGICA 1: AGACHAMENTO PADRÃO
                # ==========================================
                if exercise_type == "Agachamento Padrão":
                    # Usa o quadril e joelho esquerdo como referência padrão (ou média)
                    hip = [lm[23].x * w, lm[23].y * h]
                    knee = [lm[25].x * w, lm[25].y * h]
                    
                    # Ângulo Vertical (Fêmur vs Chão)
                    display_angle = calculate_vertical_angle(hip, knee)

                    if display_angle <= limits['NORMAL'][1]: current_state = "EM PE"
                    elif limits['TRANS'][0] <= display_angle <= limits['TRANS'][1]: current_state = "DESCENDO"
                    elif limits['PASS'][0] <= display_angle <= limits['PASS'][1]: current_state = "AGACHAMENTO OK"
                    elif display_angle > limits['TOO_LOW']: current_state = "MUITO BAIXO"

                # ==========================================
                # LÓGICA 2: AGACHAMENTO BÚLGARO
                # ==========================================
                elif exercise_type == "Agachamento Búlgaro":
                    # 1. Identificar perna da frente (maior Y = mais para baixo na imagem)
                    left_ankle_y = lm[27].y
                    right_ankle_y = lm[28].y
                    
                    # Define os índices baseados no lado
                    if left_ankle_y > right_ankle_y: # Esquerda está na frente
                        idx_shoulder, idx_hip, idx_knee, idx_ankle = 11, 23, 25, 27
                    else: # Direita está na frente
                        idx_shoulder, idx_hip, idx_knee, idx_ankle = 12, 24, 26, 28

                    # Extrair coordenadas
                    shoulder = [lm[idx_shoulder].x * w, lm[idx_shoulder].y * h]
                    hip = [lm[idx_hip].x * w, lm[idx_hip].y * h]
                    knee = [lm[idx_knee].x * w, lm[idx_knee].y * h]
                    ankle = [lm[idx_ankle].x * w, lm[idx_ankle].y * h]

                    # 2. Calcular Ângulos
                    # Ângulo do Joelho (Quadril-Joelho-Tornozelo)
                    knee_angle = calculate_angle(hip, knee, ankle)
                    # Ângulo do Tronco (Ombro-Quadril-Joelho) - Para validar postura
                    torso_angle = calculate_angle(shoulder, hip, knee)
                    
                    # Usamos o knee_angle para exibir
                    display_angle = knee_angle 

                    # 3. Máquina de Estados (Baseada na flexão do joelho)
                    if knee_angle > 160:
                        current_state = "EM PE"
                    elif 100 < knee_angle <= 160:
                        current_state = "DESCENDO"
                    elif 75 <= knee_angle <= 100:
                        current_state = "AGACHAMENTO OK"
                    elif knee_angle < 75:
                        current_state = "MUITO BAIXO"
                    
                    # 4. Verificação de Erro (Tronco)
                    # Se o tronco estiver muito fechado (<70 graus), emite alerta
                    if torso_angle < 70:
                        warning_msg = "ALERTA: TRONCO CAINDO!"

                # Atualiza estado na sessão
                st.session_state.last_state = current_state

                # ==========================================
                # RENDERIZAÇÃO (Visual)
                # ==========================================
                
                # Definição de cores
                color = {"EM PE": (0, 255, 255), "DESCENDO": (255, 165, 0), "AGACHAMENTO OK": (0, 255, 0), "MUITO BAIXO": (0, 0, 255)}.get(current_state, (255, 255, 255))
                
                # Se houver um alerta de erro crítico (ex: tronco), forçamos cor vermelha no texto de alerta
                alert_color = (0, 0, 255)

                texto_estado = f"{current_state}"
                texto_angulo = f"Angulo {int(display_angle)}o" # 'o' minúsculo simulando grau

                # Desenhar barra de fundo (aumentei um pouco a altura para caber alertas se houver)
                bar_height = 90 if warning_msg else 60
                cv2.rectangle(frame, (0, 0), (w, bar_height), (0, 0, 0), -1)

                # Texto Estado
                cv2.putText(frame, texto_estado, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Texto Ângulo (Alinhado à direita)
                (largura_texto, _), _ = cv2.getTextSize(texto_angulo, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame, texto_angulo, (w - largura_texto - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Texto de Alerta Extra (Se existir)
                if warning_msg:
                     cv2.putText(frame, warning_msg, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

            out.write(frame)

            frame_count += 1
            if total_frames > 0:
                prog = min(frame_count / total_frames, 1.0)
                progress_bar.progress(prog)
                status_text.text(f"Processando frame {frame_count}/{total_frames}...")

        cap.release()
        out.release()
        detector.close()
        
        status_text.text("Concluído! Renderizando player...")
        progress_bar.empty()

        if os.path.exists(OUTPUT_PATH):
            st.success("Análise Finalizada!")
            st.video(OUTPUT_PATH, format="video/webm")
        else:
            st.error("Erro ao gerar o arquivo final.")

elif run_analysis and not video_source_path:
    st.warning("Por favor, faça o upload de um vídeo primeiro.")
else:
    st.info("Aguardando ação do usuário.")

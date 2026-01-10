import streamlit as st
import cv2
import numpy as np
import os
import tempfile

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ==========================================
# 1. Funções Matemáticas e Auxiliares
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

def draw_visual_angle(frame, p1, p2, p3, angle_text, color=(255, 255, 255)):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    p3 = (int(p3[0]), int(p3[1]))
    cv2.line(frame, p1, p2, (255, 255, 255), 2)
    cv2.line(frame, p2, p3, (255, 255, 255), 2)
    cv2.circle(frame, p2, 6, color, -1)
    cv2.putText(frame, angle_text, (p2[0] + 15, p2[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# ==========================================
# 2. Configuração da Página e Instruções
# ==========================================

st.set_page_config(page_title="Análise Personalizável", layout="wide")
st.title("🏋️ Análise de Exercícios Personalizável")

# Seção de Instruções (compatível com Dark Mode)
st.markdown("""
<div style="background-color: #f0f2f6; color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 25px;">
    <h4 style="margin-top:0; color: #333333;">Como usar:</h4>
    <ol>
        <li style="color: #333333;">Escolha o exercício na barra lateral.</li>
        <li style="color: #333333;">Personalize com as suas Regras.</li>
        <li style="color: #333333;">Faça o upload do vídeo.</li>
        <li style="color: #333333;">Clique em <b>Processar Vídeo</b> para ver a análise com métricas visuais.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. Sidebar Dinâmica
# ==========================================

st.sidebar.header("1. Seleção do Exercício")
# --- ATUALIZAÇÃO: Adicionado 'Supino Máquina' na lista ---
exercise_type = st.sidebar.selectbox(
    "Qual exercício analisar?", 
    ["Agachamento Búlgaro", "Agachamento Padrão", "Supino Máquina"]
)

# Dicionário para armazenar as regras do usuário
user_rules = {}

st.sidebar.markdown("---")
st.sidebar.header(f"2. Regras: {exercise_type}")

# --- LÓGICA DE PERSONALIZAÇÃO ---

if exercise_type == "Agachamento Búlgaro":
    st.sidebar.info("Configure os ângulos para o Búlgaro.")
    
    # Regra 1: Profundidade do Joelho
    check_knee = st.sidebar.checkbox("Analisar Profundidade (Joelho)", value=True)
    if check_knee:
        col1, col2 = st.sidebar.columns(2)
        min_knee = col1.number_input("Ângulo Mín (Agachou)", value=75, min_value=40, max_value=120)
        max_knee = col2.number_input("Ângulo Máx (Em pé)", value=160, min_value=130, max_value=180)
        user_rules['knee'] = {'active': True, 'min': min_knee, 'max': max_knee}
    else:
        user_rules['knee'] = {'active': False}

    # Regra 2: Inclinação do Tronco
    check_torso = st.sidebar.checkbox("Alerta de Tronco (Postura)", value=True)
    if check_torso:
        min_torso = st.sidebar.slider("Ângulo Mínimo Tronco (Segurança)", 50, 90, 70)
        user_rules['torso'] = {'active': True, 'limit': min_torso}
    else:
        user_rules['torso'] = {'active': False}

elif exercise_type == "Agachamento Padrão":
    st.sidebar.info("Configure os ângulos para o Agachamento.")
    
    # Regra Única: Profundidade
    st.sidebar.markdown("**Limites de Estado:**")
    val_stand = st.sidebar.slider("Limite 'Em Pé' (graus)", 0, 40, 32)
    val_ok = st.sidebar.slider("Limite 'Agachamento OK' (graus)", 70, 110, 80)
    
    user_rules['squat_limits'] = {
        'stand_max': val_stand,
        'pass_min': val_ok
    }

# --- ATUALIZAÇÃO: Regras do Supino Máquina ---
elif exercise_type == "Supino Máquina":
    st.sidebar.info("Análise de braço (Ombro-Cotovelo-Punho).")
    
    st.sidebar.markdown("**Regras de Amplitude:**")
    
    # Regra 1: Extensão total (Fim do movimento)
    val_extended = st.sidebar.slider("Ângulo Braço Esticado (Min)", 140, 180, 160, help="Ângulo considerado como extensão total do braço.")
    
    # Regra 2: Flexão/Retorno (Início do movimento)
    val_flexed = st.sidebar.slider("Ângulo Braço na Base (Max)", 40, 100, 80, help="Ângulo quando o peso está próximo ao peito.")
    
    user_rules['bench_press'] = {
        'extended_min': val_extended,
        'flexed_max': val_flexed
    }

# ==========================================
# 4. Upload e Setup
# ==========================================

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("3. Carregar Vídeo", type=["mp4", "mov", "avi"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_custom.webm")

video_path = None
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
else:
    default = os.path.join(BASE_DIR, "gravando4.mp4")
    if os.path.exists(default):
        video_path = default

run_btn = st.sidebar.button("⚙️ PROCESSAR VÍDEO")

if "last_state" not in st.session_state:
    st.session_state.last_state = "INICIO"

# ==========================================
# 5. Loop de Processamento
# ==========================================

if run_btn and video_path:
    if not os.path.exists(MODEL_PATH):
        st.error("Modelo MediaPipe não encontrado.")
    else:
        # Setup MediaPipe
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        detector = vision.PoseLandmarker.create_from_options(options)

        # Setup Video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_width = 640
        scale = target_width / width_orig
        target_height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

        fourcc = cv2.VideoWriter_fourcc(*'vp80') 
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (target_width, target_height))

        # UI
        progress = st.progress(0)
        status = st.empty()
        
        timestamp_ms = 0
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (target_width, target_height))
            h, w, _ = frame.shape
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect_for_video(mp_image, int(timestamp_ms))
            timestamp_ms += (1000.0 / fps)

            # Variáveis visuais
            current_state = st.session_state.last_state
            main_angle_display = 0
            alert_msg = ""
            vis_p1, vis_p2, vis_p3 = None, None, None

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_pose_landmarks(frame, lm, w, h)

                # --- APLICAÇÃO DAS REGRAS PERSONALIZADAS ---
                
                # CASO 1: BULGARIAN SPLIT SQUAT
                if exercise_type == "Agachamento Búlgaro":
                    # Detectar perna da frente
                    left_y, right_y = lm[27].y, lm[28].y
                    if left_y > right_y: # Esq na frente
                        s, h_pt, k, a = 11, 23, 25, 27
                    else:
                        s, h_pt, k, a = 12, 24, 26, 28

                    shoulder = [lm[s].x * w, lm[s].y * h]
                    hip = [lm[h_pt].x * w, lm[h_pt].y * h]
                    knee = [lm[k].x * w, lm[k].y * h]
                    ankle = [lm[a].x * w, lm[a].y * h]

                    if user_rules['knee']['active']:
                        knee_angle = calculate_angle(hip, knee, ankle)
                        main_angle_display = knee_angle
                        vis_p1, vis_p2, vis_p3 = hip, knee, ankle

                        limit_max_stand = user_rules['knee']['max'] 
                        limit_min_squat = user_rules['knee']['min'] 

                        if knee_angle > limit_max_stand:
                            current_state = "EM PE"
                        elif limit_min_squat <= knee_angle <= limit_max_stand:
                            current_state = "DESCENDO"
                        elif knee_angle < limit_min_squat:
                            current_state = "AGACHAMENTO OK"
                    
                    if user_rules['torso']['active']:
                        torso_angle = calculate_angle(shoulder, hip, knee)
                        limit_torso = user_rules['torso']['limit']
                        if torso_angle < limit_torso:
                            alert_msg = f"TRONCO < {limit_torso}"
                            cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), 
                                     (int(hip[0]), int(hip[1])), (0, 0, 255), 4)

                # CASO 2: AGACHAMENTO PADRÃO
                elif exercise_type == "Agachamento Padrão":
                    hip = [lm[23].x * w, lm[23].y * h]
                    knee = [lm[25].x * w, lm[25].y * h]
                    vertical_ref = [knee[0], knee[1] - 100]
                    
                    femur_angle = calculate_angle(hip, knee, vertical_ref)
                    main_angle_display = femur_angle
                    vis_p1, vis_p2, vis_p3 = hip, knee, vertical_ref

                    lim_stand = user_rules['squat_limits']['stand_max']
                    lim_pass = user_rules['squat_limits']['pass_min']

                    if femur_angle <= lim_stand:
                        current_state = "EM PE"
                    elif lim_stand < femur_angle < lim_pass:
                        current_state = "DESCENDO"
                    elif femur_angle >= lim_pass:
                        current_state = "AGACHAMENTO OK"

                # --- ATUALIZAÇÃO: Lógica do Supino Máquina ---
                elif exercise_type == "Supino Máquina":
                    # Usamos o lado esquerdo (padrão) - Pontos 11, 13, 15
                    # Se fosse necessário, poderíamos detectar qual braço está visível
                    shoulder = [lm[11].x * w, lm[11].y * h]
                    elbow = [lm[13].x * w, lm[13].y * h]
                    wrist = [lm[15].x * w, lm[15].y * h]

                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    main_angle_display = elbow_angle
                    vis_p1, vis_p2, vis_p3 = shoulder, elbow, wrist

                    # Recupera as regras do usuário
                    limit_ext = user_rules['bench_press']['extended_min'] # ex: 160
                    limit_flex = user_rules['bench_press']['flexed_max']  # ex: 80

                    # Lógica de Estados
                    if elbow_angle >= limit_ext:
                        current_state = "BRACO ESTICADO"
                    elif elbow_angle <= limit_flex:
                        current_state = "NA BASE"
                    else:
                        current_state = "EMPURRANDO"

                st.session_state.last_state = current_state

                # --- DESENHO FINAL ---
                # Adicionei cores específicas para os estados do supino também
                color_map = {
                    "EM PE": (0, 255, 255), 
                    "DESCENDO": (255, 165, 0), 
                    "AGACHAMENTO OK": (0, 255, 0),
                    "BRACO ESTICADO": (0, 255, 0), # Verde
                    "NA BASE": (0, 255, 255),      # Amarelo
                    "EMPURRANDO": (255, 165, 0)    # Laranja
                }
                s_color = color_map.get(current_state, (255, 255, 255))
                
                if vis_p1:
                    draw_visual_angle(frame, vis_p1, vis_p2, vis_p3, f"{int(main_angle_display)}", s_color)

                cv2.rectangle(frame, (0, 0), (w, 80 if alert_msg else 50), (0, 0, 0), -1)
                cv2.putText(frame, f"Estado: {current_state}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_color, 2)
                cv2.putText(frame, f"Ang: {int(main_angle_display)}", (w - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if alert_msg:
                    cv2.putText(frame, f"ALERTA: {alert_msg}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out.write(frame)
            frame_idx += 1
            if frames_total > 0: progress.progress(frame_idx / frames_total)
            status.text(f"Processando {frame_idx}/{frames_total}...")

        cap.release()
        out.release()
        detector.close()
        
        status.text("Concluído!")
        if os.path.exists(OUTPUT_PATH):
            st.success("Processamento Finalizado com Suas Regras!")
            st.video(OUTPUT_PATH, format="video/webm")

import streamlit as st
import cv2
import numpy as np
import os
import tempfile

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

# ==========================================
# 1. CONSTANTES DE MOVIMENTO (A FÍSICA DO EXERCÍCIO)
# ==========================================
# Estas regras definem O QUE É o movimento.
# Elas ficam separadas das preferências do usuário.

MOVEMENT_CONSTANTS = {
    "Agachamento Búlgaro": {
        "state_variable": "knee_angle",
        "stages": {"UP": "EM PE", "DOWN": "AGACHAMENTO OK", "TRANSITION": "DESCENDO"}
    },
    "Agachamento Padrão": {
        "state_variable": "femur_angle",
        "stages": {"UP": "EM PE", "DOWN": "AGACHAMENTO OK", "TRANSITION": "DESCENDO"}
    },
    "Supino Máquina": {
        "state_variable": "elbow_angle",
        "stages": {"UP": "BRACO ESTICADO", "DOWN": "NA BASE", "TRANSITION": "EMPURRANDO"}
    },
    "Flexão de Braço": {
        "state_variable": "elbow_angle",
        "stages": {"UP": "EM CIMA (OK)", "DOWN": "EMBAIXO (OK)", "TRANSITION": "MOVIMENTO"}
    },
    "Rosca Direta": {
        "state_variable": "elbow_angle",
        "stages": {"UP": "ESTICADO", "DOWN": "CONTRAIDO", "TRANSITION": "EM ACAO"}
    },
    "Desenvolvimento (Ombro)": {
        "state_variable": "elbow_angle",
        "stages": {"UP": "TOPO (LOCKOUT)", "DOWN": "BASE", "TRANSITION": "MOVIMENTO"}
    },
    "Afundo (Lunge)": {
        "state_variable": "knee_angle",
        "stages": {"UP": "DESCENDO", "DOWN": "BOM AFUNDO", "TRANSITION": "DESCENDO"} 
    },
    "Levantamento Terra": {
        "state_variable": "hip_angle",
        "stages": {"UP": "TOPO (ERETO)", "DOWN": "POSICAO INICIAL", "TRANSITION": "LEVANTANDO"}
    },
    "Prancha (Plank)": {
        "state_variable": "body_angle",
        "stages": {"UP": "QUADRIL ALTO", "DOWN": "QUADRIL CAINDO", "TRANSITION": "PERFEITO"}
    },
    "Abdominal (Crunch)": {
        "state_variable": "crunch_angle",
        "stages": {"UP": "DEITADO", "DOWN": "CONTRAIDO", "TRANSITION": "MOVIMENTO"}
    },
    "Elevação Lateral": {
        "state_variable": "shoulder_abd_angle",
        "stages": {"UP": "ALTURA CORRETA", "DOWN": "DESCANSO", "TRANSITION": "SUBINDO"}
    }
}

# ==========================================
# 2. Funções Matemáticas e Auxiliares
# ==========================================

def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos (a, b, c). b é o vértice."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def draw_pose_landmarks(frame, landmarks, w, h):
    connections = [
        (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
    ]
    for s, e in connections:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

def draw_visual_angle(frame, p1, p2, p3, angle_text, color=(255, 255, 255), label=""):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    p3 = (int(p3[0]), int(p3[1]))
    cv2.line(frame, p1, p2, (255, 255, 255), 2)
    cv2.line(frame, p2, p3, (255, 255, 255), 2)
    cv2.circle(frame, p2, 6, color, -1)
    
    display_text = f"{label}: {angle_text}" if label else angle_text
    cv2.putText(frame, display_text, (p2[0] + 15, p2[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# ==========================================
# 3. Configuração da Página
# ==========================================

st.set_page_config(
    page_title="Treino Completo AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        @keyframes pulse-red {
            0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
        }
        [data-testid="stSidebarCollapsedControl"] {
            animation: pulse-red 2s infinite;
            background-color: #FF4B4B;
            color: white;
            border-radius: 50%;
        }
        [data-testid="stSidebarNav"] > button {
             border: 2px solid #FF4B4B;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Análise de Exercícios com Visão Computacional")

# ==========================================
# 4. Sidebar: Seleção e Regras do Usuário
# ==========================================

st.sidebar.header("1. Seleção do Exercício")

EXERCISE_OPTIONS = list(MOVEMENT_CONSTANTS.keys())
exercise_type = st.sidebar.selectbox("Qual exercício analisar?", EXERCISE_OPTIONS)

# Dicionário que armazena APENAS as regras configuráveis pelo usuário
user_thresholds = {}

st.sidebar.markdown("---")
st.sidebar.header(f"2. Calibragem: {exercise_type}")

# --- CONFIGURAÇÃO DOS THRESHOLDS PELO USUÁRIO ---

if exercise_type == "Agachamento Búlgaro":
    st.sidebar.caption("Define quando conta como 'Agachou' ou 'Levantou'")
    user_thresholds['knee_min'] = st.sidebar.number_input("Ângulo Mín (Baixo)", 75)
    user_thresholds['knee_max'] = st.sidebar.number_input("Ângulo Máx (Alto)", 160)
    
    st.sidebar.caption("Segurança")
    check_torso = st.sidebar.checkbox("Alerta de Tronco", value=True)
    user_thresholds['torso_limit'] = st.sidebar.slider("Limite Tronco", 50, 90, 70) if check_torso else None

elif exercise_type == "Agachamento Padrão":
    st.sidebar.caption("Define os pontos de virada do movimento")
    user_thresholds['stand_max'] = st.sidebar.slider("Limite 'Em Pé'", 0, 40, 32)
    user_thresholds['pass_min'] = st.sidebar.slider("Limite 'Agachamento OK'", 70, 110, 80)

elif exercise_type == "Supino Máquina":
    user_thresholds['extended_min'] = st.sidebar.slider("Braço Esticado (Min)", 140, 180, 160)
    user_thresholds['flexed_max'] = st.sidebar.slider("Braço na Base (Max)", 40, 100, 80)
    
    check_safety = st.sidebar.checkbox("Alerta: Cotovelos Abertos", value=True)
    user_thresholds['safety_limit'] = st.sidebar.slider("Limite Abertura Cotovelo", 60, 90, 80) if check_safety else None

elif exercise_type == "Flexão de Braço":
    user_thresholds['pu_down'] = st.sidebar.slider("Ângulo Baixo (Descida)", 60, 100, 90)
    user_thresholds['pu_up'] = st.sidebar.slider("Ângulo Alto (Subida)", 150, 180, 165)

elif exercise_type == "Rosca Direta":
    user_thresholds['bc_flex'] = st.sidebar.slider("Contração Máxima", 30, 60, 45)
    user_thresholds['bc_ext'] = st.sidebar.slider("Extensão Completa", 140, 180, 160)

elif exercise_type == "Desenvolvimento (Ombro)":
    user_thresholds['sp_up'] = st.sidebar.slider("Braço Esticado", 150, 180, 165)
    user_thresholds['sp_down'] = st.sidebar.slider("Cotovelo na Base", 60, 100, 80)

elif exercise_type == "Afundo (Lunge)":
    user_thresholds['lg_knee'] = st.sidebar.slider("Profundidade Joelho", 70, 110, 90)
    check_torso = st.sidebar.checkbox("Alerta Tronco", value=True)
    user_thresholds['lg_torso'] = st.sidebar.slider("Inclinação Tronco", 70, 90, 80) if check_torso else None

elif exercise_type == "Levantamento Terra":
    user_thresholds['dl_hip'] = st.sidebar.slider("Extensão Final", 160, 180, 170)
    user_thresholds['dl_back'] = st.sidebar.slider("Limite Flexão (Costas)", 40, 90, 60)

elif exercise_type == "Prancha (Plank)":
    user_thresholds['pk_min'] = st.sidebar.slider("Mínimo (Cair Quadril)", 150, 175, 165)
    user_thresholds['pk_max'] = st.sidebar.slider("Máximo (Empinar)", 175, 190, 185)

elif exercise_type == "Abdominal (Crunch)":
    user_thresholds['cr_flex'] = st.sidebar.slider("Contração Máxima", 40, 100, 70)
    user_thresholds['cr_ext'] = st.sidebar.slider("Retorno (Deitado)", 110, 150, 130)

elif exercise_type == "Elevação Lateral":
    user_thresholds['lr_height'] = st.sidebar.slider("Ângulo Topo", 70, 100, 85)
    user_thresholds['lr_low'] = st.sidebar.slider("Ângulo Baixo", 10, 30, 20)

# ==========================================
# 5. Upload e Setup
# ==========================================

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("3. Carregar Vídeo", type=["mp4", "mov", "avi", "webm"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_final.webm")

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
# 6. Loop de Processamento
# ==========================================

if run_btn and video_path:
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Erro: Arquivo do modelo '{MODEL_PATH}' não encontrado.")
    else:
        # Setup MediaPipe
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
        detector = vision.PoseLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_width = 640
        scale = target_width / width_orig
        target_height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

        fourcc = cv2.VideoWriter_fourcc(*'vp80') 
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (target_width, target_height))

        progress = st.progress(0)
        status = st.empty()
        
        timestamp_ms = 0
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        
        # Pega as constantes do exercício atual
        CONSTANTS = MOVEMENT_CONSTANTS[exercise_type]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (target_width, target_height))
            h, w, _ = frame.shape
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect_for_video(mp_image, int(timestamp_ms))
            timestamp_ms += (1000.0 / fps)

            # --- Variáveis de Desenho ---
            current_state = st.session_state.last_state
            main_angle_display = 0
            alert_msg = ""
            vis_p1, vis_p2, vis_p3 = None, None, None
            label_angle = "" 

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_pose_landmarks(frame, lm, w, h)

                # --- Helpers de Pontos ---
                def get_pt(idx): return [lm[idx].x * w, lm[idx].y * h]

                # Pontos comuns (Lado Esquerdo padrão)
                sh_l, hip_l, knee_l, ank_l = get_pt(11), get_pt(23), get_pt(25), get_pt(27)
                elb_l, wr_l = get_pt(13), get_pt(15)

                # ========================================================
                # APLICAÇÃO DAS REGRAS (SEPARADAS)
                # ========================================================
                
                # Agachamento Búlgaro
                if exercise_type == "Agachamento Búlgaro":
                    # Detecta perna frontal
                    if lm[27].y > lm[28].y: s_idx, h_idx, k_idx, a_idx = 11, 23, 25, 27
                    else: s_idx, h_idx, k_idx, a_idx = 12, 24, 26, 28
                    p_sh, p_hip, p_knee, p_ank = get_pt(s_idx), get_pt(h_idx), get_pt(k_idx), get_pt(a_idx)

                    # Regra de Estado (Movimento)
                    knee_angle = calculate_angle(p_hip, p_knee, p_ank)
                    main_angle_display = knee_angle
                    vis_p1, vis_p2, vis_p3 = p_hip, p_knee, p_ank
                    label_angle = "Joelho"

                    if knee_angle > user_thresholds['knee_max']: current_state = CONSTANTS['stages']['UP']
                    elif user_thresholds['knee_min'] <= knee_angle <= user_thresholds['knee_max']: current_state = CONSTANTS['stages']['TRANSITION']
                    elif knee_angle < user_thresholds['knee_min']: current_state = CONSTANTS['stages']['DOWN']
                    
                    # Regra de Segurança (Opcional)
                    if user_thresholds.get('torso_limit'):
                        torso_angle = calculate_angle(p_sh, p_hip, p_knee)
                        if torso_angle < user_thresholds['torso_limit']:
                            alert_msg = "TRONCO INCLINADO"
                            cv2.line(frame, (int(p_sh[0]), int(p_sh[1])), (int(p_hip[0]), int(p_hip[1])), (0,0,255), 3)

                # Agachamento Padrão
                elif exercise_type == "Agachamento Padrão":
                    vertical_ref = [knee_l[0], knee_l[1] - 100]
                    femur_angle = calculate_angle(hip_l, knee_l, vertical_ref)
                    
                    main_angle_display = femur_angle
                    vis_p1, vis_p2, vis_p3 = hip_l, knee_l, vertical_ref
                    label_angle = "Coxa Vert."

                    if femur_angle <= user_thresholds['stand_max']: current_state = CONSTANTS['stages']['UP']
                    elif user_thresholds['stand_max'] < femur_angle < user_thresholds['pass_min']: current_state = CONSTANTS['stages']['TRANSITION']
                    elif femur_angle >= user_thresholds['pass_min']: current_state = CONSTANTS['stages']['DOWN']

                # Supino Máquina
                elif exercise_type == "Supino Máquina":
                    elbow_angle = calculate_angle(sh_l, elb_l, wr_l)
                    main_angle_display = elbow_angle
                    vis_p1, vis_p2, vis_p3 = sh_l, elb_l, wr_l
                    label_angle = "Cotovelo"

                    if elbow_angle >= user_thresholds['extended_min']: current_state = CONSTANTS['stages']['UP']
                    elif elbow_angle <= user_thresholds['flexed_max']: current_state = CONSTANTS['stages']['DOWN']
                    else: current_state = CONSTANTS['stages']['TRANSITION']

                    if user_thresholds.get('safety_limit'):
                        abduction_angle = calculate_angle(hip_l, sh_l, elb_l)
                        if abduction_angle > user_thresholds['safety_limit']:
                            alert_msg = "COTOVELOS MUITO ABERTOS!"
                            cv2.line(frame, (int(sh_l[0]), int(sh_l[1])), (int(elb_l[0]), int(elb_l[1])), (0, 0, 255), 3)

                # Flexão de Braço
                elif exercise_type == "Flexão de Braço":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    main_angle_display = angle_elb
                    vis_p1, vis_p2, vis_p3 = sh_l, elb_l, wr_l
                    label_angle = "Cotovelo"

                    if angle_elb < user_thresholds['pu_down']: current_state = CONSTANTS['stages']['DOWN']
                    elif angle_elb > user_thresholds['pu_up']: current_state = CONSTANTS['stages']['UP']
                    else: current_state = CONSTANTS['stages']['TRANSITION']

                # Rosca Direta
                elif exercise_type == "Rosca Direta":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    main_angle_display = angle_elb
                    vis_p1, vis_p2, vis_p3 = sh_l, elb_l, wr_l
                    label_angle = "Biceps"

                    if angle_elb < user_thresholds['bc_flex']: current_state = CONSTANTS['stages']['DOWN']
                    elif angle_elb > user_thresholds['bc_ext']: current_state = CONSTANTS['stages']['UP']
                    else: current_state = CONSTANTS['stages']['TRANSITION']

                # Desenvolvimento (Ombro)
                elif exercise_type == "Desenvolvimento (Ombro)":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    main_angle_display = angle_elb
                    vis_p1, vis_p2, vis_p3 = sh_l, elb_l, wr_l

                    if angle_elb > user_thresholds['sp_up']: current_state = CONSTANTS['stages']['UP']
                    elif angle_elb < user_thresholds['sp_down']: current_state = CONSTANTS['stages']['DOWN']
                    else: current_state = CONSTANTS['stages']['TRANSITION']

                # Afundo (Lunge)
                elif exercise_type == "Afundo (Lunge)":
                    angle_knee = calculate_angle(hip_l, knee_l, ank_l)
                    main_angle_display = angle_knee
                    vis_p1, vis_p2, vis_p3 = hip_l, knee_l, ank_l
                    label_angle = "Joelho"
                    
                    if angle_knee <= user_thresholds['lg_knee']: current_state = CONSTANTS['stages']['DOWN']
                    else: current_state = CONSTANTS['stages']['UP']
                        
                    if user_thresholds.get('lg_torso'):
                        angle_torso = calculate_angle(sh_l, hip_l, knee_l)
                        if angle_torso < user_thresholds['lg_torso']: alert_msg = "POSTURA RUIM"

                # Levantamento Terra
                elif exercise_type == "Levantamento Terra":
                    angle_hip = calculate_angle(sh_l, hip_l, knee_l)
                    main_angle_display = angle_hip
                    vis_p1, vis_p2, vis_p3 = sh_l, hip_l, knee_l
                    label_angle = "Quadril"

                    if angle_hip > user_thresholds['dl_hip']: current_state = CONSTANTS['stages']['UP']
                    elif angle_hip < user_thresholds['dl_back']: current_state = CONSTANTS['stages']['DOWN']
                    else: current_state = CONSTANTS['stages']['TRANSITION']

                # Prancha (Plank)
                elif exercise_type == "Prancha (Plank)":
                    angle_body = calculate_angle(sh_l, hip_l, ank_l)
                    main_angle_display = angle_body
                    vis_p1, vis_p2, vis_p3 = sh_l, hip_l, ank_l
                    label_angle = "Alinhamento"

                    if user_thresholds['pk_min'] <= angle_body <= user_thresholds['pk_max']: current_state = CONSTANTS['stages']['TRANSITION']
                    elif angle_body < user_thresholds['pk_min']: current_state = CONSTANTS['stages']['DOWN']
                    else: current_state = CONSTANTS['stages']['UP']

                # Abdominal (Crunch)
                elif exercise_type == "Abdominal (Crunch)":
                    angle_crunch = calculate_angle(sh_l, hip_l, knee_l)
                    main_angle_display = angle_crunch
                    vis_p1, vis_p2, vis_p3 = sh_l, hip_l, knee_l

                    if angle_crunch < user_thresholds['cr_flex']: current_state = CONSTANTS['stages']['DOWN']
                    elif angle_crunch > user_thresholds['cr_ext']: current_state = CONSTANTS['stages']['UP']
                    else: current_state = CONSTANTS['stages']['TRANSITION']

                # Elevação Lateral
                elif exercise_type == "Elevação Lateral":
                    angle_abd = calculate_angle(hip_l, sh_l, elb_l)
                    main_angle_display = angle_abd
                    vis_p1, vis_p2, vis_p3 = hip_l, sh_l, elb_l
                    label_angle = "Ombro"

                    if angle_abd >= user_thresholds['lr_height']: current_state = CONSTANTS['stages']['UP']
                    elif angle_abd < user_thresholds['lr_low']: current_state = CONSTANTS['stages']['DOWN']
                    else: current_state = CONSTANTS['stages']['TRANSITION']

                # ========================================================
                # ATUALIZAÇÃO DE UI E ESTADOS
                # ========================================================

                st.session_state.last_state = current_state

                # Cores
                # Verde para Sucesso (Agachamento OK, Braço Esticado, etc)
                # Amarelo/Laranja para Transição
                # Vermelho para Erro
                
                s_color = (255, 255, 255) # Padrão
                
                # Verifica se o estado atual é um dos estados "finais" de sucesso
                if current_state == CONSTANTS['stages']['DOWN'] or current_state == CONSTANTS['stages']['UP']:
                    # Em alguns exercícios, UP é bom, em outros DOWN é bom.
                    # Simplificação: Se não for 'transition', pinta de Verde/Azul
                    s_color = (0, 255, 0)
                else:
                    s_color = (0, 255, 255) # Transição (Amarelo)

                # Se houver alerta de segurança, sobrepõe com vermelho
                if alert_msg:
                     s_color = (0, 0, 255)

                # Desenha Linhas de Angulo
                if vis_p1:
                    draw_visual_angle(frame, vis_p1, vis_p2, vis_p3, f"{int(main_angle_display)}", s_color, label_angle)

                # Caixa de Infos
                box_h = 85 if alert_msg else 60
                cv2.rectangle(frame, (0, 0), (w, box_h), (20, 20, 20), -1)
                
                cv2.putText(frame, f"STATUS: {current_state}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_color, 2)
                
                if alert_msg:
                    cv2.putText(frame, f"ALERTA: {alert_msg}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out.write(frame)
            frame_idx += 1
            if frames_total > 0: progress.progress(frame_idx / frames_total)
            status.text(f"Processando {frame_idx}/{frames_total}...")

        cap.release()
        out.release()
        detector.close()
        
        status.success("Análise Concluída!")
        st.video(OUTPUT_PATH, format="video/webm")

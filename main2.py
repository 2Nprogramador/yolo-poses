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
# 2. Configuração da Página
# ==========================================

st.set_page_config(page_title="Treino Completo AI", layout="wide")
st.title("🏋️ Análise de Exercícios (Originais + Novos)")

# ==========================================
# 3. Sidebar: Seleção e Regras
# ==========================================

st.sidebar.header("1. Seleção do Exercício")

# Lista unificada: Seus 3 originais + Novos
EXERCISE_OPTIONS = [
    # --- SEUS ORIGINAIS ---
    "Agachamento Búlgaro", 
    "Agachamento Padrão", 
    "Supino Máquina",
    # --- NOVOS ADICIONADOS ---
    "Flexão de Braço",
    "Rosca Direta",
    "Desenvolvimento (Ombro)",
    "Afundo (Lunge)",
    "Levantamento Terra",
    "Prancha (Plank)",
    "Abdominal (Crunch)",
    "Elevação Lateral"
]

exercise_type = st.sidebar.selectbox("Qual exercício analisar?", EXERCISE_OPTIONS)

user_rules = {}
st.sidebar.markdown("---")
st.sidebar.header(f"2. Regras: {exercise_type}")

# --- LÓGICA DE REGRAS (MANTENDO AS SUAS E ADICIONANDO AS NOVAS) ---

if exercise_type == "Agachamento Búlgaro":
    st.sidebar.info("Regras Originais do Búlgaro")
    check_knee = st.sidebar.checkbox("Analisar Profundidade", value=True)
    if check_knee:
        col1, col2 = st.sidebar.columns(2)
        min_knee = col1.number_input("Ângulo Mín (Agachou)", 75)
        max_knee = col2.number_input("Ângulo Máx (Em pé)", 160)
        user_rules['knee'] = {'active': True, 'min': min_knee, 'max': max_knee}
    else:
        user_rules['knee'] = {'active': False}

    check_torso = st.sidebar.checkbox("Alerta de Tronco", value=True)
    if check_torso:
        min_torso = st.sidebar.slider("Ângulo Mínimo Tronco", 50, 90, 70)
        user_rules['torso'] = {'active': True, 'limit': min_torso}
    else:
        user_rules['torso'] = {'active': False}

elif exercise_type == "Agachamento Padrão":
    st.sidebar.info("Regras Originais do Agachamento")
    val_stand = st.sidebar.slider("Limite 'Em Pé'", 0, 40, 32)
    val_ok = st.sidebar.slider("Limite 'Agachamento OK'", 70, 110, 80)
    user_rules['squat_limits'] = {'stand_max': val_stand, 'pass_min': val_ok}

elif exercise_type == "Supino Máquina":
    st.sidebar.info("Regras Originais do Supino Máquina")
    val_extended = st.sidebar.slider("Braço Esticado (Min)", 140, 180, 160)
    val_flexed = st.sidebar.slider("Braço na Base (Max)", 40, 100, 80)
    
    check_safety = st.sidebar.checkbox("Alerta: Cotovelos Abertos", value=True)
    safety_limit = 80
    if check_safety:
        safety_limit = st.sidebar.slider("Limite Abertura Cotovelo", 60, 90, 80)

    user_rules['bench_press'] = {
        'extended_min': val_extended, 'flexed_max': val_flexed,
        'safety_check': check_safety, 'safety_limit': safety_limit
    }

# --- REGRAS PARA OS NOVOS EXERCÍCIOS ---

elif exercise_type == "Flexão de Braço":
    st.sidebar.caption("Regra: Amplitude do Cotovelo")
    pu_down = st.sidebar.slider("Ângulo Baixo (Descida)", 60, 100, 90)
    pu_up = st.sidebar.slider("Ângulo Alto (Subida)", 150, 180, 165)
    user_rules = {'pu_down': pu_down, 'pu_up': pu_up}

elif exercise_type == "Rosca Direta":
    st.sidebar.caption("Regra: Contração do Bíceps")
    bc_flex = st.sidebar.slider("Contração Máxima", 30, 60, 45)
    bc_ext = st.sidebar.slider("Extensão Completa", 140, 180, 160)
    user_rules = {'bc_flex': bc_flex, 'bc_ext': bc_ext}

elif exercise_type == "Desenvolvimento (Ombro)":
    st.sidebar.caption("Regra: Lockout e Base")
    sp_up = st.sidebar.slider("Braço Esticado", 150, 180, 165)
    sp_down = st.sidebar.slider("Cotovelo na Base", 60, 100, 80)
    user_rules = {'sp_up': sp_up, 'sp_down': sp_down}

elif exercise_type == "Afundo (Lunge)":
    st.sidebar.caption("Regra: Joelho Traseiro")
    lg_knee = st.sidebar.slider("Profundidade Joelho", 70, 110, 90)
    lg_torso = st.sidebar.slider("Inclinação Tronco", 70, 90, 80)
    user_rules = {'lg_knee': lg_knee, 'lg_torso': lg_torso}

elif exercise_type == "Levantamento Terra":
    st.sidebar.caption("Regra: Extensão de Quadril")
    dl_hip = st.sidebar.slider("Extensão Final", 160, 180, 170)
    dl_back = st.sidebar.slider("Limite Flexão (Costas)", 40, 90, 60)
    user_rules = {'dl_hip': dl_hip, 'dl_back': dl_back}

elif exercise_type == "Prancha (Plank)":
    st.sidebar.caption("Regra: Alinhamento do Corpo")
    pk_min = st.sidebar.slider("Mínimo (Cair Quadril)", 150, 175, 165)
    pk_max = st.sidebar.slider("Máximo (Empinar)", 175, 190, 185)
    user_rules = {'pk_min': pk_min, 'pk_max': pk_max}

elif exercise_type == "Abdominal (Crunch)":
    st.sidebar.caption("Regra: Flexão de Tronco")
    cr_flex = st.sidebar.slider("Contração Máxima", 40, 100, 70)
    cr_ext = st.sidebar.slider("Retorno (Deitado)", 110, 150, 130)
    user_rules = {'cr_flex': cr_flex, 'cr_ext': cr_ext}

elif exercise_type == "Elevação Lateral":
    st.sidebar.caption("Regra: Altura dos Ombros")
    lr_height = st.sidebar.slider("Ângulo Topo", 70, 100, 85)
    lr_low = st.sidebar.slider("Ângulo Baixo", 10, 30, 20)
    user_rules = {'lr_height': lr_height, 'lr_low': lr_low}

# ==========================================
# 4. Upload e Setup
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
    # Tenta usar vídeo padrão se existir
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
        st.error(f"⚠️ Erro: Arquivo do modelo '{MODEL_PATH}' não encontrado.")
    else:
        # Setup MediaPipe Tasks
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

            # --- Variáveis de Desenho ---
            current_state = st.session_state.last_state
            main_angle_display = 0
            alert_msg = ""
            vis_p1, vis_p2, vis_p3 = None, None, None
            label_angle = "" # Para nomear o ângulo na tela

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_pose_landmarks(frame, lm, w, h)

                # --- Helpers de Pontos ---
                def get_pt(idx): return [lm[idx].x * w, lm[idx].y * h]

                # Pontos comuns (Lado Esquerdo padrão)
                sh_l, hip_l, knee_l, ank_l = get_pt(11), get_pt(23), get_pt(25), get_pt(27)
                elb_l, wr_l = get_pt(13), get_pt(15)

                # ========================================================
                # INICIO DA LÓGICA DE EXERCÍCIOS
                # ========================================================

                # --------------------------------------------------------
                # 1. SEUS EXERCÍCIOS ORIGINAIS (LÓGICA MANTIDA)
                # --------------------------------------------------------
                
                if exercise_type == "Agachamento Búlgaro":
                    # Detecta qual perna está na frente
                    if lm[27].y > lm[28].y: # Esq na frente
                        s_idx, h_idx, k_idx, a_idx = 11, 23, 25, 27
                    else:
                        s_idx, h_idx, k_idx, a_idx = 12, 24, 26, 28
                    
                    p_sh = get_pt(s_idx)
                    p_hip = get_pt(h_idx)
                    p_knee = get_pt(k_idx)
                    p_ank = get_pt(a_idx)

                    if user_rules['knee']['active']:
                        knee_angle = calculate_angle(p_hip, p_knee, p_ank)
                        main_angle_display = knee_angle
                        vis_p1, vis_p2, vis_p3 = p_hip, p_knee, p_ank
                        label_angle = "Joelho"

                        limit_max_stand = user_rules['knee']['max'] 
                        limit_min_squat = user_rules['knee']['min'] 

                        if knee_angle > limit_max_stand: current_state = "EM PE"
                        elif limit_min_squat <= knee_angle <= limit_max_stand: current_state = "DESCENDO"
                        elif knee_angle < limit_min_squat: current_state = "AGACHAMENTO OK"
                    
                    if user_rules['torso']['active']:
                        torso_angle = calculate_angle(p_sh, p_hip, p_knee)
                        if torso_angle < user_rules['torso']['limit']:
                            alert_msg = "TRONCO INCLINADO"
                            cv2.line(frame, (int(p_sh[0]), int(p_sh[1])), (int(p_hip[0]), int(p_hip[1])), (0,0,255), 3)

                elif exercise_type == "Agachamento Padrão":
                    # Usa ponto vertical virtual
                    vertical_ref = [knee_l[0], knee_l[1] - 100]
                    femur_angle = calculate_angle(hip_l, knee_l, vertical_ref)
                    
                    main_angle_display = femur_angle
                    vis_p1, vis_p2, vis_p3 = hip_l, knee_l, vertical_ref
                    label_angle = "Coxa Vert."

                    lim_stand = user_rules['squat_limits']['stand_max']
                    lim_pass = user_rules['squat_limits']['pass_min']

                    if femur_angle <= lim_stand: current_state = "EM PE"
                    elif lim_stand < femur_angle < lim_pass: current_state = "DESCENDO"
                    elif femur_angle >= lim_pass: current_state = "AGACHAMENTO OK"

                elif exercise_type == "Supino Máquina":
                    elbow_angle = calculate_angle(sh_l, elb_l, wr_l)
                    main_angle_display = elbow_angle
                    vis_p1, vis_p2, vis_p3 = sh_l, elb_l, wr_l
                    label_angle = "Cotovelo"

                    limit_ext = user_rules['bench_press']['extended_min'] 
                    limit_flex = user_rules['bench_press']['flexed_max']  

                    if elbow_angle >= limit_ext: current_state = "BRACO ESTICADO"
                    elif elbow_angle <= limit_flex: current_state = "NA BASE"
                    else: current_state = "EMPURRANDO"

                    if user_rules['bench_press']['safety_check']:
                        abduction_angle = calculate_angle(hip_l, sh_l, elb_l)
                        if abduction_angle > user_rules['bench_press']['safety_limit']:
                            alert_msg = "COTOVELOS MUITO ABERTOS!"
                            cv2.line(frame, (int(sh_l[0]), int(sh_l[1])), (int(elb_l[0]), int(elb_l[1])), (0, 0, 255), 3)

                # --------------------------------------------------------
                # 2. NOVOS EXERCÍCIOS ADICIONADOS
                # --------------------------------------------------------

                elif exercise_type == "Flexão de Braço":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    main_angle_display = angle_elb
                    vis_p1, vis_p2, vis_p3 = sh_l, elb_l, wr_l
                    label_angle = "Cotovelo"

                    if angle_elb < user_rules['pu_down']: current_state = "EMBAIXO (OK)"
                    elif angle_elb > user_rules['pu_up']: current_state = "EM CIMA (OK)"
                    else: current_state = "MOVIMENTO"

                elif exercise_type == "Rosca Direta":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    main_angle_display = angle_elb
                    vis_p1, vis_p2, vis_p3 = sh_l, elb_l, wr_l
                    label_angle = "Biceps"

                    if angle_elb < user_rules['bc_flex']: current_state = "CONTRAIDO"
                    elif angle_elb > user_rules['bc_ext']: current_state = "ESTICADO"
                    else: current_state = "EM ACAO"

                elif exercise_type == "Desenvolvimento (Ombro)":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    main_angle_display = angle_elb
                    vis_p1, vis_p2, vis_p3 = sh_l, elb_l, wr_l

                    if angle_elb > user_rules['sp_up']: current_state = "TOPO (LOCKOUT)"
                    elif angle_elb < user_rules['sp_down']: current_state = "BASE"
                    else: current_state = "MOVIMENTO"

                elif exercise_type == "Afundo (Lunge)":
                    angle_knee = calculate_angle(hip_l, knee_l, ank_l)
                    main_angle_display = angle_knee
                    vis_p1, vis_p2, vis_p3 = hip_l, knee_l, ank_l
                    label_angle = "Joelho"

                    angle_torso = calculate_angle(sh_l, hip_l, knee_l)
                    
                    if angle_knee <= user_rules['lg_knee']: current_state = "BOM AFUNDO"
                    else: current_state = "DESCENDO"
                        
                    if angle_torso < user_rules['lg_torso']: alert_msg = "POSTURA RUIM"

                elif exercise_type == "Levantamento Terra":
                    angle_hip = calculate_angle(sh_l, hip_l, knee_l)
                    main_angle_display = angle_hip
                    vis_p1, vis_p2, vis_p3 = sh_l, hip_l, knee_l
                    label_angle = "Quadril"

                    if angle_hip > user_rules['dl_hip']: current_state = "TOPO (ERETO)"
                    elif angle_hip < user_rules['dl_back']: current_state = "POSICAO INICIAL"
                    else: current_state = "LEVANTANDO"

                elif exercise_type == "Prancha (Plank)":
                    angle_body = calculate_angle(sh_l, hip_l, ank_l)
                    main_angle_display = angle_body
                    vis_p1, vis_p2, vis_p3 = sh_l, hip_l, ank_l
                    label_angle = "Alinhamento"

                    if user_rules['pk_min'] <= angle_body <= user_rules['pk_max']: current_state = "PERFEITO"
                    elif angle_body < user_rules['pk_min']: current_state = "QUADRIL CAINDO"
                    else: current_state = "QUADRIL ALTO"

                elif exercise_type == "Abdominal (Crunch)":
                    angle_crunch = calculate_angle(sh_l, hip_l, knee_l)
                    main_angle_display = angle_crunch
                    vis_p1, vis_p2, vis_p3 = sh_l, hip_l, knee_l

                    if angle_crunch < user_rules['cr_flex']: current_state = "CONTRAIDO"
                    elif angle_crunch > user_rules['cr_ext']: current_state = "DEITADO"
                    else: current_state = "MOVIMENTO"

                elif exercise_type == "Elevação Lateral":
                    angle_abd = calculate_angle(hip_l, sh_l, elb_l)
                    main_angle_display = angle_abd
                    vis_p1, vis_p2, vis_p3 = hip_l, sh_l, elb_l
                    label_angle = "Ombro"

                    if angle_abd >= user_rules['lr_height']: current_state = "ALTURA CORRETA"
                    elif angle_abd < user_rules['lr_low']: current_state = "DESCANSO"
                    else: current_state = "SUBINDO"

                # --------------------------------------------------------
                # FIM DA LÓGICA / ATUALIZAÇÃO DE UI
                # --------------------------------------------------------

                st.session_state.last_state = current_state

                # Cores baseadas no estado
                color_map = {
                    "AGACHAMENTO OK": (0, 255, 0), "EM PE": (0, 255, 255),
                    "BRACO ESTICADO": (0, 255, 0), "NA BASE": (0, 255, 255),
                    "EMBAIXO (OK)": (0, 255, 0), "CONTRAIDO": (0, 255, 0),
                    "TOPO (LOCKOUT)": (0, 255, 0), "PERFEITO": (0, 255, 0),
                    "ALTURA CORRETA": (0, 255, 0), "BOM AFUNDO": (0, 255, 0)
                }
                # Pega a cor ou usa Branco se não achar
                s_color = color_map.get(current_state, (255, 255, 255))
                # Se for um estado de erro (vermelho)
                if "CAINDO" in current_state or "RUIM" in current_state: s_color = (0, 0, 255)

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

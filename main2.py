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
    # Conexões principais para visualização simplificada
    connections = [
        (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
    ]
    for s, e in connections:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    
    # Desenhar pontos
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
    
    text_display = f"{label}: {angle_text}" if label else angle_text
    cv2.putText(frame, text_display, (p2[0] + 15, p2[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ==========================================
# 2. Configuração da Página
# ==========================================

st.set_page_config(page_title="AI Gym Tracker - 10 Exercícios", layout="wide")
st.title("🏋️ AI Fitness Tracker - 10 Exercícios")

st.markdown("""
<style>
    .main-header {font-size: 20px; font-weight: bold; color: #333;}
</style>
<div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <strong>Instruções:</strong> Selecione o exercício, ajuste as regras (thresholds) e carregue seu vídeo.
    O sistema analisará automaticamente as 2 regras principais para cada movimento.
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. Sidebar: Seleção e Regras
# ==========================================

st.sidebar.header("1. Configuração do Exercício")

EXERCISES = [
    "1. Agachamento (Squat)",
    "2. Flexão de Braço (Push-up)",
    "3. Rosca Direta (Bicep Curl)",
    "4. Desenvolvimento (Shoulder Press)",
    "5. Afundo (Lunge)",
    "6. Levantamento Terra (Deadlift)",
    "7. Prancha (Plank)",
    "8. Supino (Bench Press)",
    "9. Abdominal (Crunch)",
    "10. Elevação Lateral"
]

exercise_type = st.sidebar.selectbox("Selecione o movimento:", EXERCISES)
user_rules = {}

st.sidebar.markdown("---")
st.sidebar.header("2. Regras e Limites")

# --- BLOCO GIGANTE DE REGRAS ---

if exercise_type == "1. Agachamento (Squat)":
    st.sidebar.caption("Regra 1: Profundidade (Quadril vs Joelho)")
    sq_depth = st.sidebar.slider("Ângulo Máx Agachamento", 60, 110, 85)
    st.sidebar.caption("Regra 2: Inclinação do Tronco")
    sq_torso = st.sidebar.slider("Ângulo Mín Tronco (Segurança)", 40, 80, 50)
    user_rules = {'sq_depth': sq_depth, 'sq_torso': sq_torso}

elif exercise_type == "2. Flexão de Braço (Push-up)":
    st.sidebar.caption("Regra 1: Amplitude Descida (Cotovelos)")
    pu_down = st.sidebar.slider("Ângulo Cotovelo Baixo", 60, 100, 90)
    st.sidebar.caption("Regra 2: Extensão Subida")
    pu_up = st.sidebar.slider("Ângulo Cotovelo Alto", 150, 180, 165)
    user_rules = {'pu_down': pu_down, 'pu_up': pu_up}

elif exercise_type == "3. Rosca Direta (Bicep Curl)":
    st.sidebar.caption("Regra 1: Contração Máxima")
    bc_flex = st.sidebar.slider("Ângulo Mínimo (Cima)", 30, 60, 45)
    st.sidebar.caption("Regra 2: Extensão Completa")
    bc_ext = st.sidebar.slider("Ângulo Máximo (Baixo)", 140, 180, 160)
    user_rules = {'bc_flex': bc_flex, 'bc_ext': bc_ext}

elif exercise_type == "4. Desenvolvimento (Shoulder Press)":
    st.sidebar.caption("Regra 1: Lockout (Braço Esticado)")
    sp_up = st.sidebar.slider("Extensão no Topo", 150, 180, 165)
    st.sidebar.caption("Regra 2: Amplitude Baixa")
    sp_down = st.sidebar.slider("Cotovelo na Base", 60, 100, 80)
    user_rules = {'sp_up': sp_up, 'sp_down': sp_down}

elif exercise_type == "5. Afundo (Lunge)":
    st.sidebar.caption("Regra 1: Joelho Traseiro")
    lg_knee = st.sidebar.slider("Profundidade Joelho", 70, 110, 90)
    st.sidebar.caption("Regra 2: Verticalidade Tronco")
    lg_torso = st.sidebar.slider("Inclinação Tronco", 70, 90, 80)
    user_rules = {'lg_knee': lg_knee, 'lg_torso': lg_torso}

elif exercise_type == "6. Levantamento Terra (Deadlift)":
    st.sidebar.caption("Regra 1: Extensão de Quadril (Topo)")
    dl_hip = st.sidebar.slider("Extensão Quadril", 160, 180, 170)
    st.sidebar.caption("Regra 2: Costas Retas (Aprox)")
    # Usaremos ângulo ombro-quadril-joelho para estimar postura segura
    dl_back = st.sidebar.slider("Limite Flexão Quadril (Baixo)", 40, 90, 60)
    user_rules = {'dl_hip': dl_hip, 'dl_back': dl_back}

elif exercise_type == "7. Prancha (Plank)":
    st.sidebar.caption("Regra 1 & 2: Alinhamento (Ombro-Quadril-Tornozelo)")
    pk_min = st.sidebar.slider("Mínimo (Cair Quadril)", 150, 175, 165)
    pk_max = st.sidebar.slider("Máximo (Empinar)", 175, 190, 185) # Logic check needed
    user_rules = {'pk_min': pk_min, 'pk_max': pk_max}

elif exercise_type == "8. Supino (Bench Press)":
    st.sidebar.caption("Regra 1: Toque no Peito")
    bp_chest = st.sidebar.slider("Ângulo Cotovelo Baixo", 45, 90, 75)
    st.sidebar.caption("Regra 2: Segurança (Cotovelo Aberto)")
    bp_safety = st.sidebar.slider("Ângulo Abdução Ombro", 70, 95, 85)
    user_rules = {'bp_chest': bp_chest, 'bp_safety': bp_safety}

elif exercise_type == "9. Abdominal (Crunch)":
    st.sidebar.caption("Regra 1: Contração")
    cr_flex = st.sidebar.slider("Ângulo Flexão Tronco", 40, 100, 70)
    st.sidebar.caption("Regra 2: Retorno")
    cr_ext = st.sidebar.slider("Ângulo Retorno", 110, 150, 130)
    user_rules = {'cr_flex': cr_flex, 'cr_ext': cr_ext}

elif exercise_type == "10. Elevação Lateral":
    st.sidebar.caption("Regra 1: Altura do Ombro")
    lr_height = st.sidebar.slider("Ângulo Abdução (Topo)", 70, 100, 85)
    st.sidebar.caption("Regra 2: Retorno")
    lr_low = st.sidebar.slider("Ângulo Baixo", 10, 30, 20)
    user_rules = {'lr_height': lr_height, 'lr_low': lr_low}

# ==========================================
# 4. Upload e Execução
# ==========================================

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("3. Carregar Vídeo", type=["mp4", "mov", "avi", "webm"])
run_btn = st.sidebar.button("▶️ INICIAR ANÁLISE")

# Placeholder para vídeo padrão se necessário
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_analise.webm")

video_path = None
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

# ==========================================
# 5. Loop Principal
# ==========================================

if run_btn and video_path:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Erro: Modelo {MODEL_PATH} não encontrado. Coloque o arquivo .task na pasta.")
    else:
        # Configurar MediaPipe
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

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        timestamp_ms = 0
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.resize(frame, (target_width, target_height))
            h, w, _ = frame.shape
            
            # MediaPipe precisa de RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect_for_video(mp_image, int(timestamp_ms))
            timestamp_ms += (1000.0 / fps)

            # --- LÓGICA DE DESENHO E ANÁLISE ---
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_pose_landmarks(frame, landmarks, w, h)

                # Definindo pontos principais (Assumindo lado ESQUERDO como padrão para simplificar,
                # em produção idealmente detecta o lado visível)
                # Índices MP: 11=OmbroEsq, 23=QuadrilEsq, 25=JoelhoEsq, 27=TornEsq, 13=CotoveloEsq, 15=PunhoEsq
                
                # Atalhos para coordenadas em pixels
                def get_pt(idx): return [landmarks[idx].x * w, landmarks[idx].y * h]
                
                sh_l, hip_l, knee_l, ank_l = get_pt(11), get_pt(23), get_pt(25), get_pt(27)
                elb_l, wr_l = get_pt(13), get_pt(15)
                ear_l = get_pt(7)

                # Variáveis para feedback
                status_msg = "AGUARDANDO"
                color_status = (200, 200, 200)
                alert_msg = ""
                
                # --- APLICAÇÃO DAS REGRAS POR EXERCÍCIO ---

                if exercise_type == "1. Agachamento (Squat)":
                    # Regra 1: Profundidade (Ângulo do Joelho)
                    angle_knee = calculate_angle(hip_l, knee_l, ank_l)
                    # Regra 2: Tronco (Ombro-Quadril-Joelho)
                    angle_torso = calculate_angle(sh_l, hip_l, knee_l)
                    
                    draw_visual_angle(frame, hip_l, knee_l, ank_l, f"{int(angle_knee)}", label="Joelho")
                    
                    if angle_knee < user_rules['sq_depth']:
                        status_msg = "PROFUNDIDADE OK!"
                        color_status = (0, 255, 0)
                    elif angle_knee < 140:
                        status_msg = "DESCENDO"
                        color_status = (0, 255, 255)
                    else:
                        status_msg = "EM PE"
                        
                    if angle_torso < user_rules['sq_torso']:
                        alert_msg = "TRONCO MUITO INCLINADO"

                elif exercise_type == "2. Flexão de Braço (Push-up)":
                    # Regra 1 e 2: Cotovelos
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    draw_visual_angle(frame, sh_l, elb_l, wr_l, f"{int(angle_elb)}", label="Cotovelo")

                    if angle_elb < user_rules['pu_down']:
                        status_msg = "EMBAIXO (OK)"
                        color_status = (0, 255, 0)
                    elif angle_elb > user_rules['pu_up']:
                        status_msg = "EM CIMA (OK)"
                        color_status = (0, 255, 0)
                    else:
                        status_msg = "MOVIMENTO"
                        color_status = (0, 255, 255)

                elif exercise_type == "3. Rosca Direta (Bicep Curl)":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    draw_visual_angle(frame, sh_l, elb_l, wr_l, f"{int(angle_elb)}")

                    if angle_elb < user_rules['bc_flex']:
                        status_msg = "CONTRAIDO"
                        color_status = (0, 255, 0)
                    elif angle_elb > user_rules['bc_ext']:
                        status_msg = "ESTICADO"
                        color_status = (0, 255, 255)
                    else:
                        status_msg = "CONCENTRICO/EXCEN."
                        color_status = (255, 165, 0)

                elif exercise_type == "4. Desenvolvimento (Shoulder Press)":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    draw_visual_angle(frame, sh_l, elb_l, wr_l, f"{int(angle_elb)}")
                    
                    if angle_elb > user_rules['sp_up']:
                        status_msg = "LOCKOUT (TOPO)"
                        color_status = (0, 255, 0)
                    elif angle_elb < user_rules['sp_down']:
                        status_msg = "BASE"
                        color_status = (0, 255, 255)
                    else:
                        status_msg = "EMPURRANDO"

                elif exercise_type == "5. Afundo (Lunge)":
                    angle_knee = calculate_angle(hip_l, knee_l, ank_l) # Joelho perna visível
                    angle_torso = calculate_angle(sh_l, hip_l, knee_l)
                    
                    draw_visual_angle(frame, hip_l, knee_l, ank_l, f"{int(angle_knee)}", label="Joelho")

                    if angle_knee <= user_rules['lg_knee']:
                        status_msg = "BOM AFUNDO"
                        color_status = (0, 255, 0)
                    else:
                        status_msg = "DESCENDO"
                        
                    if angle_torso < user_rules['lg_torso']:
                        alert_msg = "POSTURA RUIM (INCLINADO)"

                elif exercise_type == "6. Levantamento Terra (Deadlift)":
                    angle_hip = calculate_angle(sh_l, hip_l, knee_l)
                    draw_visual_angle(frame, sh_l, hip_l, knee_l, f"{int(angle_hip)}", label="Quadril")

                    if angle_hip > user_rules['dl_hip']:
                        status_msg = "BLOQUEADO (TOPO)"
                        color_status = (0, 255, 0)
                    elif angle_hip < user_rules['dl_back']:
                        status_msg = "POSICAO INICIAL"
                        color_status = (255, 165, 0)
                    else:
                        status_msg = "LEVANTANDO"

                elif exercise_type == "7. Prancha (Plank)":
                    angle_body = calculate_angle(sh_l, hip_l, ank_l)
                    draw_visual_angle(frame, sh_l, hip_l, ank_l, f"{int(angle_body)}", label="Corpo")

                    if user_rules['pk_min'] <= angle_body <= user_rules['pk_max']:
                        status_msg = "PRANCHA PERFEITA"
                        color_status = (0, 255, 0)
                    elif angle_body < user_rules['pk_min']:
                        status_msg = "QUADRIL CAINDO"
                        color_status = (0, 0, 255)
                    else:
                        status_msg = "QUADRIL ALTO"
                        color_status = (255, 165, 0)

                elif exercise_type == "8. Supino (Bench Press)":
                    angle_elb = calculate_angle(sh_l, elb_l, wr_l)
                    angle_abduction = calculate_angle(hip_l, sh_l, elb_l)
                    
                    draw_visual_angle(frame, sh_l, elb_l, wr_l, f"{int(angle_elb)}")

                    if angle_elb < user_rules['bp_chest']:
                        status_msg = "NO PEITO"
                        color_status = (0, 255, 0)
                    else:
                        status_msg = "MOVIMENTO"

                    if angle_abduction > user_rules['bp_safety']:
                        alert_msg = "COTOVELO MUITO ABERTO!"
                        cv2.line(frame, (int(sh_l[0]), int(sh_l[1])), (int(elb_l[0]), int(elb_l[1])), (0,0,255), 4)

                elif exercise_type == "9. Abdominal (Crunch)":
                    # Medir aproximação ombro-quadril em relação ao joelho
                    angle_crunch = calculate_angle(sh_l, hip_l, knee_l)
                    draw_visual_angle(frame, sh_l, hip_l, knee_l, f"{int(angle_crunch)}")

                    if angle_crunch < user_rules['cr_flex']:
                        status_msg = "CONTRAIDO MAX"
                        color_status = (0, 255, 0)
                    elif angle_crunch > user_rules['cr_ext']:
                        status_msg = "DEITADO"
                        color_status = (200, 200, 200)

                elif exercise_type == "10. Elevação Lateral":
                    # Abdução do ombro: Ângulo Quadril-Ombro-Cotovelo
                    angle_abd = calculate_angle(hip_l, sh_l, elb_l)
                    draw_visual_angle(frame, hip_l, sh_l, elb_l, f"{int(angle_abd)}", label="Ombro")

                    if angle_abd >= user_rules['lr_height']:
                        status_msg = "ALTURA CORRETA"
                        color_status = (0, 255, 0)
                    elif angle_abd < user_rules['lr_low']:
                        status_msg = "DESCANSO"
                    else:
                        status_msg = "SUBINDO"

                # --- DESENHO DA UI (Caixas e Textos) ---
                
                # Caixa de Fundo para textos
                box_height = 90 if alert_msg else 60
                cv2.rectangle(frame, (0, 0), (w, box_height), (20, 20, 20), -1)
                
                # Texto Estado Principal
                cv2.putText(frame, f"STATUS: {status_msg}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_status, 2)
                
                # Texto Alerta (se houver)
                if alert_msg:
                    cv2.putText(frame, f"ALERTA: {alert_msg}", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Marca d'água exercício
                cv2.putText(frame, exercise_type.split('.')[1], (10, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            out.write(frame)
            frame_idx += 1
            if frames_total > 0: progress_bar.progress(frame_idx / frames_total)
            status_text.text(f"Processando frame {frame_idx}/{frames_total}...")

        cap.release()
        out.release()
        detector.close()

        status_text.success("Processamento concluído!")
        st.video(OUTPUT_PATH, format="video/webm")

    # Botão para baixar resultado
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "rb") as file:
            btn = st.download_button(
                label="⬇️ Baixar Vídeo Analisado",
                data=file,
                file_name="treino_analisado.webm",
                mime="video/webm"
            )

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

st.set_page_config(page_title="Visão Computacional - Análise de Exercícios", layout="wide")
st.markdown("""
    <h1>Análise de Exercícios<br>com Visão Computacional</h1>
    """, unsafe_allow_html=True)
st.markdown("""
**Instruções:**
1. Escolha o exercício na barra lateral.
2. Faça o upload do vídeo.
3. Clique em **Processar Vídeo** para ver a análise com métricas visuais.
""")

# ==========================================
# 2. Funções Auxiliares e Matemáticas
# ==========================================

def get_thresholds_beginner():
    return {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (70, 95), 'TOO_LOW': 95}

def get_thresholds_pro():
    return {'NORMAL': (0, 32), 'TRANS': (35, 65), 'PASS': (80, 95), 'TOO_LOW': 95}

def calculate_angle(a, b, c):
    """Calcula o ângulo entre três pontos (a, b, c). b é o vértice."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_vertical_angle(hip, knee):
    """Calcula o ângulo do fêmur em relação à vertical (para Agachamento Padrão)."""
    vertical_point = [knee[0], knee[1] - 100]
    return calculate_angle(hip, knee, vertical_point)

def draw_pose_landmarks(frame, landmarks, w, h):
    """Desenha o esqueleto completo (pontos e conexões)."""
    connections = [
        (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
    ]
    for s, e in connections:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Linhas Verdes
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1) # Pontos Vermelhos

def draw_visual_angle(frame, p1, p2, p3, angle_text, color=(255, 255, 255)):
    """
    Desenha as linhas brancas representando o ângulo que está sendo medido
    e coloca o valor numérico próximo à articulação.
    """
    # Converter coordenadas para inteiros
    p1 = (int(p1[0]), int(p1[1])) # Ponto A
    p2 = (int(p2[0]), int(p2[1])) # Vértice (Ex: Joelho)
    p3 = (int(p3[0]), int(p3[1])) # Ponto C
    
    # Desenhar linhas conectando os pontos (Visual do Ângulo)
    cv2.line(frame, p1, p2, (255, 255, 255), 3) # Linha branca mais grossa
    cv2.line(frame, p2, p3, (255, 255, 255), 3) 
    
    # Desenhar destaque no vértice
    cv2.circle(frame, p2, 6, color, -1)
    cv2.circle(frame, p2, 8, (255, 255, 255), 2) 
    
    # Escrever o valor do ângulo próximo ao vértice
    cv2.putText(frame, angle_text, (p2[0] + 15, p2[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

# ==========================================
# 3. Interface e Configuração
# ==========================================

st.sidebar.header("Configurações")

# Seleção do Exercício
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
        st.sidebar.info("Usando vídeo padrão.")

run_analysis = st.sidebar.button("⚙️ Processar Vídeo")

if "last_state" not in st.session_state:
    st.session_state.last_state = "EM PE"

# ==========================================
# 4. Loop Principal de Processamento
# ==========================================

if run_analysis and video_source_path:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Erro: Modelo MediaPipe não encontrado em {MODEL_PATH}.")
        st.stop()

    # Configuração do Modelo
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # Leitura do Vídeo
    cap = cv2.VideoCapture(video_source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    if width_orig == 0:
        st.error("Erro ao ler o vídeo. Arquivo corrompido.")
        st.stop()

    # Redimensionamento
    target_width = 640
    scale = target_width / width_orig
    target_height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale)

    # Gravador
    fourcc = cv2.VideoWriter_fourcc(*'vp80') 
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

        # 1. Preparar Frame
        frame = cv2.resize(frame, (target_width, target_height))
        h, w, _ = frame.shape
        
        # Converter BGR para RGB para o MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 2. Detecção
        result = detector.detect_for_video(mp_image, int(timestamp_ms))
        timestamp_ms += (1000.0 / fps)

        # 3. Variáveis de Lógica e Exibição
        display_angle = 0
        label_metric = ""  # Nome do que está sendo medido (Ex: "Inclin. Coxa")
        current_state = st.session_state.last_state
        warning_msg = ""   # Alertas extras
        
        # Pontos visuais para desenhar o ângulo geométrico (A, B, C)
        vis_p1, vis_p2, vis_p3 = None, None, None

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            
            # Desenha esqueleto básico
            draw_pose_landmarks(frame, lm, w, h)

            # ---------------------------------------------------------
            # LÓGICA 1: AGACHAMENTO PADRÃO
            # ---------------------------------------------------------
            if exercise_type == "Agachamento Padrão":
                # Referência: Lado Esquerdo (Padrão)
                hip = [lm[23].x * w, lm[23].y * h]
                knee = [lm[25].x * w, lm[25].y * h]
                
                # Criar ponto virtual vertical acima do joelho para referência visual
                vertical_ref = [knee[0], knee[1] - 100]

                display_angle = calculate_vertical_angle(hip, knee)
                label_metric = "Inclin. Coxa" # Nome intuitivo

                # Definir pontos para desenhar o ângulo (Quadril -> Joelho -> Vertical)
                vis_p1, vis_p2, vis_p3 = hip, knee, vertical_ref

                # Regras de Estado
                if display_angle <= limits['NORMAL'][1]: current_state = "EM PE"
                elif limits['TRANS'][0] <= display_angle <= limits['TRANS'][1]: current_state = "DESCENDO"
                elif limits['PASS'][0] <= display_angle <= limits['PASS'][1]: current_state = "AGACHAMENTO OK"
                elif display_angle > limits['TOO_LOW']: current_state = "MUITO BAIXO"

            # ---------------------------------------------------------
            # LÓGICA 2: AGACHAMENTO BÚLGARO
            # ---------------------------------------------------------
            elif exercise_type == "Agachamento Búlgaro":
                # Detecção automática da perna da frente (Menor Y = Mais alto na tela? Não. Maior Y = Mais baixo na tela)
                # MediaPipe: Y=0 é topo, Y=1 é base. Perna no chão tem Y maior.
                left_ankle_y = lm[27].y
                right_ankle_y = lm[28].y
                
                if left_ankle_y > right_ankle_y: # Esquerda está mais baixa (chão/frente)
                    idx_s, idx_h, idx_k, idx_a = 11, 23, 25, 27
                else: # Direita está mais baixa (chão/frente)
                    idx_s, idx_h, idx_k, idx_a = 12, 24, 26, 28

                shoulder = [lm[idx_s].x * w, lm[idx_s].y * h]
                hip = [lm[idx_h].x * w, lm[idx_h].y * h]
                knee = [lm[idx_k].x * w, lm[idx_k].y * h]
                ankle = [lm[idx_a].x * w, lm[idx_a].y * h]

                # Cálculos
                knee_angle = calculate_angle(hip, knee, ankle)
                torso_angle = calculate_angle(shoulder, hip, knee)
                
                display_angle = knee_angle 
                label_metric = "Flexao Joelho" # Nome intuitivo

                # Definir pontos para desenhar o ângulo (Quadril -> Joelho -> Tornozelo)
                vis_p1, vis_p2, vis_p3 = hip, knee, ankle

                # Regras de Estado
                if knee_angle > 160: current_state = "EM PE"
                elif 100 < knee_angle <= 160: current_state = "DESCENDO"
                elif 75 <= knee_angle <= 100: current_state = "AGACHAMENTO OK"
                elif knee_angle < 75: current_state = "MUITO BAIXO"
                
                # Regra de Alerta Extra (Tronco)
                if torso_angle < 70:
                    warning_msg = "ALERTA: TRONCO CAINDO!"

            # Atualizar Sessão
            st.session_state.last_state = current_state

            # ---------------------------------------------------------
            # RENDERIZAÇÃO VISUAL FINAL
            # ---------------------------------------------------------
            
            # Mapa de Cores
            color_map = {
                "EM PE": (0, 255, 255),          # Amarelo Cyan
                "DESCENDO": (255, 165, 0),       # Laranja
                "AGACHAMENTO OK": (0, 255, 0),   # Verde
                "MUITO BAIXO": (0, 0, 255)       # Vermelho
            }
            state_color = color_map.get(current_state, (255, 255, 255))
            alert_color = (0, 0, 255)

            # Textos
            texto_estado = f"{current_state}"
            texto_metric = f"{label_metric}: {int(display_angle)}o"

            # 1. Desenhar Geometria do Ângulo no Corpo (Linhas Brancas)
            if vis_p1 and vis_p2 and vis_p3:
                # Desenha linhas brancas e o valor numérico flutuante
                draw_visual_angle(frame, vis_p1, vis_p2, vis_p3, f"{int(display_angle)}", state_color)

            # 2. Barra de Status (Topo)
            # Aumenta altura se houver alerta de tronco
            bar_height = 90 if warning_msg else 60
            cv2.rectangle(frame, (0, 0), (w, bar_height), (0, 0, 0), -1)

            # Lado Esquerdo: Estado Atual
            cv2.putText(frame, texto_estado, (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

            # Lado Direito: Métrica Específica
            (tw, th), _ = cv2.getTextSize(texto_metric, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, texto_metric, (w - tw - 20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Alerta Extra (Abaixo da linha principal)
            if warning_msg:
                 cv2.putText(frame, warning_msg, (10, 80), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

        # Escrever frame no vídeo final
        out.write(frame)

        # Atualizar UI Streamlit
        frame_count += 1
        if total_frames > 0:
            prog = min(frame_count / total_frames, 1.0)
            progress_bar.progress(prog)
            status_text.text(f"Processando frame {frame_count}/{total_frames}...")

    # Finalização
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

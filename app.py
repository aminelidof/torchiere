import streamlit as st
import cv2
import time
import pandas as pd
import threading
from datetime import datetime
import csv
import os
import numpy as np
from collections import deque
import logging

# ---------- Configuration ----------
APP_TITLE = "🔥 Surveillance de Combustion de Torchère — Diagnostic Amélioré avec Auto-ROI"
ALERTS_LOG_FILE = "alerts.log"
DEFAULT_CSV = "torchere_log.csv"
ROLLING_WINDOW = 600 # frames kept in memory for plotting
LOG_INTERVAL = 1.0 # seconds between CSV flushes

# Seuil de Ratio de Pixel : Pourcentage de pixels d'une couleur nécessaire pour la "détecter"
COLOR_RATIO_THRESH = 0.001 # 0.1% de la zone de flamme
# Facteur de dominance : Le rouge doit être au moins X fois plus présent que le jaune pour être classé "Rougeâtre"
# Si RED_DOMINANCE_FACTOR est 0.5, le rouge doit être > 50% du jaune.
RED_DOMINANCE_FACTOR = 0.5 

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("torchere")

# ---------- Diagnostic dictionary (MISES À JOUR) ----------
DIAGNOSTIC_INFO = {
    "Normale / Bonne (Bleue)": { # NOUVEAU
        "Signification": "Combustion optimale (Bleue). Flamme visible, pas de fumée noire significative.",
        "Composant": "Flamme propre",
        "Color": (255, 100, 0),  # BGR (Bleu)
    },
    "Normale / Bonne": { # FLAMME GÉNÉRIQUE
        "Signification": "Combustion idéale. Flamme visible, pas de fumée noire significative.",
        "Composant": "Flamme propre (Jaune par défaut)",
        "Color": (0, 255, 0),  # BGR (Vert)
    },
    "Attention (Jaune/Orange)": { # NOUVEAU
        "Signification": "Combustion incomplète (Jaune/Orange). Génération de suie, vérifier l'atomisation.",
        "Composant": "Flamme riche en carbone",
        "Color": (0, 165, 255), # BGR (Orange)
    },
    "Danger (Rougeâtre)": { # NOUVEAU
        "Signification": "Combustion instable ou présence d'impuretés solides (Rougeâtre).",
        "Composant": "Impuretés / Instabilité",
        "Color": (0, 0, 255),  # BGR (Rouge)
    },
    "Attention (Degradee)": {
        "Signification": "Présence de fumée grise; vérifier source (vapeur ou vapeur mélangée).",
        "Composant": "Fumée grise / vapeur",
        "Color": (0, 165, 255),
    },
    "Danger (Noire)": {
        "Signification": "Fumée noire (combustion très incomplète). Intervention requise.",
        "Composant": "Suie / Carbone",
        "Color": (0, 0, 255),
    },
    "NoFlame_BlackSmoke": {
        "Signification": "Fumée noire sans flamme détectée — possible incendie/événement.",
        "Composant": "Suie / Anomalie",
        "Color": (80, 0, 0),
    },
    "NotTorchere": {
        "Signification": "Probablement une cheminée industrielle (pas une torchère) — classification: non-flare.",
        "Composant": "Cheminée/Source non-flare",
        "Color": (128, 128, 128),
    },
    "Unknown": {
        "Signification": "Image non catégorisée / conditions d'éclairage faibles.",
        "Composant": "Incertitude",
        "Color": (255, 255, 0),
    },
}

# ---------- Lightweight control object cached for the app ----------
class ControlFlags:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.is_running = False
        self.lock = threading.Lock()

@st.cache_resource
def get_control_flags():
    return ControlFlags()

CF = get_control_flags()

# ---------- Utilities ----------
def init_csv(file_path: str):
    if not os.path.exists(file_path):
        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "TimestampUTC",
                    "MeanV_ROI",
                    "MeanV_Sky",
                    "Opacity",
                    "MeanS_ROI",
                    "FlameRatio_Total",
                    "FlameRatio_Blue",    # NOUVEAU
                    "FlameRatio_Yellow", # NOUVEAU
                    "FlameRatio_Red",    # NOUVEAU
                    "BlackSmokeRatio",
                    "GreySmokeRatio",
                    "Diagnosis"
                ])
            logger.info(f"CSV initialisé: {file_path}")
        except Exception as e:
            logger.exception(f"Impossible d'initialiser {file_path}: {e}")

init_csv(DEFAULT_CSV)

def create_cap(source):
    # Accept numeric string indices as integers
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
    except Exception:
        pass
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def trigger_alert(status, opacity, mean_s, signification, composant):
    timestamp = datetime.utcnow().isoformat() + "Z"
    log_message = (
        f"[{timestamp}] [ALERTE {status}] Opacity={opacity:.1f} | Saturation={mean_s:.1f} | Composant={composant}"
    )
    try:
        with open(ALERTS_LOG_FILE, "a") as f:
            f.write(log_message + "\n")
        logger.warning(log_message)
    except Exception as e:
        logger.exception(f"Erreur d'écriture du log d'alerte: {e}")

# ---------- AUTO-ROI DETECTION FUNCTIONS (Nouveauté) ----------
# Fonctions auto_detect_flare_roi et auto_detect_sky_roi restent inchangées...

def auto_detect_flare_roi(frame, hsv_lower, hsv_upper, scale_w=300, scale_h=400):
    """
    Détecte le centre de la source lumineuse (flamme) et propose une ROI centrée autour.
    Retourne (x, y, w, h) ou None si aucune flamme n'est trouvée.
    """
    if frame is None or frame.size == 0:
        return None
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    
    # Trouver les contours et la plus grande zone
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Choisir le plus grand contour (la flamme principale)
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 50: # Seuil minimum de détection de la flamme
        return None

    # Centre de masse
    M = cv2.moments(largest_contour)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Définir le ROI centré en X, et commençant un peu au-dessus du centre de masse en Y
        w_f, h_f = frame.shape[1], frame.shape[0]
        
        # Début de la ROI (coin supérieur gauche)
        rx = max(0, cX - scale_w // 2)
        # Position Y : On remonte pour inclure la fumée au-dessus de la flamme
        ry = max(0, cY - int(scale_h * 0.75)) 
        
        # Assurez-vous que la ROI reste dans l'image
        rx = min(rx, w_f - scale_w) if w_f > scale_w else 0
        ry = min(ry, h_f - scale_h) if h_f > scale_h else 0
        
        return (rx, ry, scale_w, scale_h)
    return None

def auto_detect_sky_roi(frame, flare_roi, sky_w=100, sky_h=50):
    """
    Détecte une petite zone de ciel en haut de l'image, loin de la torchère.
    Le coin supérieur gauche est privilégié.
    """
    h_f, w_f = frame.shape[:2]
    
    # Coordonnées par défaut (coin supérieur gauche)
    rsx, rsy = 10, 10
    
    # Si la ROI de la flamme est détectée, essayons d'éviter de la chevaucher.
    if flare_roi is not None:
        frx, fry, frw, frh = flare_roi
        
        # Si la flamme est à gauche (e.g., rx < w/2), placer le ciel à droite
        if frx < w_f // 2:
            rsx = w_f - sky_w - 10 # Coin supérieur droit
        # Sinon, laisser dans le coin supérieur gauche par défaut
        
        # Vérification simple du chevauchement avec le ROI de la flamme (seulement le coin)
        # On s'assure que le coin du ciel est loin du haut de la flamme
        if rsy + sky_h > fry and rsx < frx + frw and rsx + sky_w > frx:
             # Si chevauchement, on essaie de changer de coin, ou on remonte
             if frx < w_f // 2: # Flamme à gauche, on va à droite
                 rsx = w_f - sky_w - 10
             else: # Flamme à droite, on va à gauche
                 rsx = 10

    # S'assurer que les coordonnées ne dépassent pas
    rsx = min(rsx, w_f - sky_w - 5)
    rsy = min(rsy, h_f - sky_h - 5)
    
    return (rsx, rsy, sky_w, sky_h)

# ---------- Image analysis helpers: flame & smoke detection ----------

def detect_flame_ratio(bgr_roi, flame_hsv_lower, flame_hsv_upper):
    """(Ancienne fonction conservée pour le ratio total basé sur les seuils UI)"""
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    lower = np.array(flame_hsv_lower, dtype=np.uint8)
    upper = np.array(flame_hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    area = cv2.countNonZero(mask)
    return float(area) / (mask.size) if mask.size > 0 else 0.0

def classify_flame_color_ratios(bgr_roi):
    """
    NOUVEAU: Calcule le ratio de pixels correspondant à Bleu, Jaune/Orange, et Rougeâtre.
    Retourne: flame_ratio_total, blue_ratio, yellow_ratio, red_ratio
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0 
        
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.size // 3
    
    # --- 1. Seuils pour le BLEU (Combustion optimale) ---
    # H: 100-130, S/V élevés
    lower_blue = np.array([100, 100, 100], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_ratio = cv2.countNonZero(mask_blue) / total_pixels

    # --- 2. Seuils pour JAUNE/ORANGE (Combustion incomplète) ---
    # H: 5-40, S/V très élevés
    lower_yellow = np.array([5, 100, 150], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = cv2.countNonZero(mask_yellow) / total_pixels

    # --- 3. Seuils pour ROUGEÂTRE (Impuretés/Température) ---
    # H: 0-5 ou 170-179, S/V moyens à élevés
    lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
    upper_red1 = np.array([5, 255, 255], dtype=np.uint8)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 80, 80], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    red_ratio = cv2.countNonZero(mask_red) / total_pixels
    
    # Ratio total de la flamme (pour la métrique de présence globale)
    flame_mask_total = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_yellow, mask_red))
    flame_ratio_total = cv2.countNonZero(flame_mask_total) / total_pixels
    
    return flame_ratio_total, blue_ratio, yellow_ratio, red_ratio

def detect_black_smoke_ratio(bgr_roi, v_thresh=80, s_thresh=100):
    """Detect dark, desaturated pixels typical of black smoke."""
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    s = hsv[:,:,1]
    mask = (v < v_thresh) & (s < s_thresh)
    area = np.count_nonzero(mask)
    return float(area) / mask.size

def detect_grey_smoke_ratio(bgr_roi, v_low=80, v_high=220, s_thresh=80):
    """Detect medium brightness, low saturation pixels typical of grey smoke/vapor."""
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    s = hsv[:,:,1]
    mask = (v >= v_low) & (v <= v_high) & (s < s_thresh)
    area = np.count_nonzero(mask)
    return float(area) / mask.size

def compute_opacity_from_sky(roi_bgr, sky_bgr):
    """Estimate opacity similar to original logic: relative brightness drop vs sky."""
    try:
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        sky_hsv = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2HSV)
        mean_v = float(np.mean(roi_hsv[:,:,2]))
        mean_v_sky = float(np.mean(sky_hsv[:,:,2]))
        mean_s = float(np.mean(roi_hsv[:,:,1]))
        
        if mean_v_sky > 5.0:
            # Opacité: la différence de luminosité, normalisée par la luminosité du ciel
            opacity_normalized = 1.0 - (mean_v / mean_v_sky)
            opacity = max(0.0, min(100.0, opacity_normalized * 100.0))
        else:
            opacity = 0.0 # Ciel trop sombre (nuit ou faible luminosité)
        
        return opacity, mean_v, mean_v_sky, mean_s
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

# Simple visual heuristic to detect if ROI likely belongs to a tall stack (cheminée)
def structure_hint_roi(frame, roi):
    # use simple edge / vertical projection: tall thin dark vertical structure suggests stack/torchere
    rx, ry, rw, rh = roi
    h, w = frame.shape[:2]
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx+rw), min(h, ry+rh)
    if x1 >= x2 or y1 >= y2:
        return False
    roi_img = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    # Sobel vertical edges
    sob = cv2.Sobel(roi_img, cv2.CV_64F, 1, 0, ksize=3)
    mag = np.abs(sob)
    # If many vertical edges and ratio of height/width large => likely stack
    vert_edge_ratio = np.mean(mag > np.percentile(mag, 90))
    aspect = (y2-y1) / max((x2-x1), 1)
    return vert_edge_ratio > 0.02 and aspect > 1.2


# ---------- Combined diagnosis logic (CORRIGÉE) ----------
def diagnose_from_metrics(flame_ratio, blue_ratio, yellow_ratio, red_ratio, black_ratio, grey_ratio, opacity, has_structure_hint=False,
                             flame_thresh=0.002, black_thresh=0.03, grey_thresh=0.03, color_thresh=COLOR_RATIO_THRESH):
    """
    Return classification string and severity from combined indicators, incl. flame color.
    """
    # Heuristics:
    is_flame = flame_ratio >= flame_thresh
    is_black = black_ratio >= black_thresh
    is_grey = grey_ratio >= grey_thresh
    
    # Détection des couleurs (si le ratio de la couleur dépasse un petit seuil)
    is_blue = blue_ratio >= color_thresh
    is_yellow = yellow_ratio >= color_thresh
    is_red = red_ratio >= color_thresh
    
    # --- 1. Cas d'Anomalie Sévère (Fumée Noire) ---
    if is_black:
        if is_flame:
            return "Danger (Noire)" # Fumée noire + Flamme (incomplète)
        else:
            return "NoFlame_BlackSmoke" # Anomalie majeure
    
    # --- 2. Cas de Fumée Grise (Vapeur ou Combustion Dégradée) ---
    if is_grey and is_flame:
        # Si on a une flamme, la fumée grise est toujours une attention.
        return "Attention (Degradee)"
    
    if is_grey and not is_flame:
        # Si pas de flamme, on utilise le hint de structure pour classifier.
        return "NotTorchere" if has_structure_hint else "Unknown" # On assume que c'est une cheminée industrielle

    # --- 3. Diagnostic basé sur la Couleur de la Flamme (CORRECTION DE LA LOGIQUE DE DOMINANCE) ---
    if is_flame:
        
        # Priorité A: Rouge (Danger) - Si le rouge est présent ET est significativement plus dominant que le jaune
        # Cela empêche une petite teinte rouge d'une grande flamme jaune de surclasser le diagnostic.
        if is_red and red_ratio > yellow_ratio * RED_DOMINANCE_FACTOR:
            return "Danger (Rougeâtre)"
        
        # Priorité B: Jaune/Orange (Attention) - Si le jaune est présent (et dominant ou si le rouge est minoritaire)
        if is_yellow:
            return "Attention (Jaune/Orange)"
        
        # Priorité C: Bleu (Normale)
        if is_blue:
            return "Normale / Bonne (Bleue)"
            
        # Si flamme détectée mais couleur non classifiée (ex: trop blanche/pale)
        return "Normale / Bonne" # Fallback, couleur neutre/générique

    # --- 4. Fallback (Si pas de flamme et pas de fumée critique) ---
    if has_structure_hint:
        return "NotTorchere" # Structure détectée, pas d'activité flare
        
    if opacity > 50:
        return "Attention (Degradee)" # Opacité élevée sans fumée classifiée explicitement
        
    return "Unknown"


# ---------- Background analysis thread (MISES À JOUR) ----------
def analysis_loop(source, roi, sky_roi, thresholds_ui, csv_path, state_ref, is_uploaded):
    logger.info("Démarrage du thread d'analyse")
    cap = create_cap(source)
    
    # Retry logic for live camera
    attempts = 0
    while isinstance(source, int) and not cap.isOpened() and attempts < 3:
        attempts += 1
        time.sleep(0.5)
        cap = create_cap(source)
        
    if not cap.isOpened():
        logger.error("Impossible d'ouvrir la source vidéo")
        CF.is_running = False
        return

    last_csv_flush = time.time()
    last_status = "Unknown"
    
    # Initial ROIs (used as fallback/default)
    rx, ry, rw, rh = roi
    rsx, rsy, rsw, rsh = sky_roi

    opacity_deque: deque = state_ref.get("opacity_deque")
    if opacity_deque is None:
        opacity_deque = deque(maxlen=ROLLING_WINDOW)
        state_ref["opacity_deque"] = opacity_deque

    init_csv(csv_path)
    
    try:
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            while not CF.stop_event.is_set():
                ret, frame = cap.read()
                tnow = time.time()
                
                if not ret or frame is None:
                    logger.info("Fin de flux/lecture échouée")
                    break

                # Downscale if too large
                if max(frame.shape[:2]) > 1024:
                    scale = 1024 / max(frame.shape[:2])
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                h_f, w_f = frame.shape[:2]

                # --- AUTO-ROI LOGIC (DYNAMIC UPDATE) ---
                if state_ref.get('roi_mode') == 'Automatique':
                    # Pour l'auto-détection, on utilise le seuil jaune/orange par défaut comme indice
                    yellow_hsv_lower = [5, 100, 150]
                    yellow_hsv_upper = [40, 255, 255]
                    new_flare_roi = auto_detect_flare_roi(
                        frame, 
                        yellow_hsv_lower,
                        yellow_hsv_upper,
                        scale_w=300, scale_h=400
                    )
                    
                    if new_flare_roi:
                        rx, ry, rw, rh = new_flare_roi
                        
                    # Detect Sky ROI (normalisation area)
                    new_sky_roi = auto_detect_sky_roi(frame, new_flare_roi)
                    if new_sky_roi:
                        rsx, rsy, rsw, rsh = new_sky_roi
                
                # Update shared state with current (manual or auto) coordinates
                with CF.lock:
                    state_ref['x_roi'], state_ref['y_roi'], state_ref['w_roi'], state_ref['h_roi'] = rx, ry, rw, rh
                    state_ref['rsx'], state_ref['rsy'], state_ref['rsw'], state_ref['rsh'] = rsx, rsy, rsw, rsh
                # --- END AUTO-ROI LOGIC ---

                # Clip ROI coordinates
                x1, y1 = max(0, rx), max(0, ry)
                x2, y2 = min(w_f, rx + rw), min(h_f, ry + rh)
                sx1, sy1 = max(0, rsx), max(0, rsy)
                sx2, sy2 = min(w_f, rsx + rsw), min(h_f, rsy + rsh)

                if x1 >= x2 or y1 >= y2 or sx1 >= sx2 or sy1 >= sy2:
                    # invalid roi
                    opacity, mean_v, mean_v_sky, mean_s = 0.0, 0.0, 0.0, 0.0
                    flame_ratio = black_ratio = grey_ratio = 0.0
                    blue_ratio = yellow_ratio = red_ratio = 0.0 # NOUVEAU
                    diagnosis = "Unknown"
                else:
                    roi_bgr = frame[y1:y2, x1:x2].copy()
                    sky_bgr = frame[sy1:sy2, sx1:sx2].copy()

                    # detection metrics
                    # NOTE: Nous utilisons flame_ratio_total de la fonction de classification
                    flame_ratio, blue_ratio, yellow_ratio, red_ratio = classify_flame_color_ratios(roi_bgr)
                    
                    black_ratio = detect_black_smoke_ratio(roi_bgr, v_thresh=thresholds_ui['black_v'], s_thresh=thresholds_ui['black_s'])
                    grey_ratio = detect_grey_smoke_ratio(roi_bgr, v_low=thresholds_ui['grey_v_low'], v_high=thresholds_ui['grey_v_high'], s_thresh=thresholds_ui['grey_s'])

                    opacity, mean_v, mean_v_sky, mean_s = compute_opacity_from_sky(roi_bgr, sky_bgr)

                    has_structure = structure_hint_roi(frame, (rx, ry, rw, rh))

                    # Utilisation des ratios de couleur dans le diagnostic
                    diagnosis = diagnose_from_metrics(
                        flame_ratio, blue_ratio, yellow_ratio, red_ratio, black_ratio, grey_ratio, opacity, has_structure,
                        flame_thresh=thresholds_ui['flame_area_frac'],
                        black_thresh=thresholds_ui['black_frac'],
                        grey_thresh=thresholds_ui['grey_frac']
                    )

                # alert logic
                if diagnosis in ["Danger (Noire)", "NoFlame_BlackSmoke", "Danger (Rougeâtre)"]: # Ajout de l'alerte rouge
                    if last_status not in ["Danger (Noire)", "NoFlame_BlackSmoke", "Danger (Rougeâtre)"]:
                        trigger_alert(diagnosis, opacity, mean_s, DIAGNOSTIC_INFO[diagnosis]["Signification"], DIAGNOSTIC_INFO[diagnosis]["Composant"])
                    last_status = diagnosis
                else:
                    last_status = diagnosis

                # draw overlays and texts
                try:
                    info = DIAGNOSTIC_INFO.get(diagnosis, DIAGNOSTIC_INFO["Unknown"])
                    color = tuple(int(c) for c in info["Color"])
                    # Draw Flame/Smoke ROI (Dynamically updated coordinates)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    # Draw Sky ROI (Dynamically updated coordinates)
                    cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (255, 0, 0), 2)
                    cv2.putText(frame, f"{diagnosis}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.putText(frame, f"F:{flame_ratio:.3f} B:{black_ratio:.3f} G:{grey_ratio:.3f} O:{opacity:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                except Exception:
                    pass

                # update shared state
                with CF.lock:
                    state_ref['last_frame_rgb'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    state_ref['current_status'] = diagnosis
                    state_ref['opacity_live'] = opacity
                    state_ref['mean_s_live'] = mean_s
                    state_ref['flame_ratio'] = flame_ratio
                    state_ref['blue_ratio'] = blue_ratio  # NOUVEAU
                    state_ref['yellow_ratio'] = yellow_ratio # NOUVEAU
                    state_ref['red_ratio'] = red_ratio    # NOUVEAU
                    state_ref['black_ratio'] = black_ratio
                    state_ref['grey_ratio'] = grey_ratio

                opacity_deque.append({'time': datetime.utcnow(), 'opacity': opacity})

                # periodically write CSV
                if tnow - last_csv_flush >= LOG_INTERVAL:
                    try:
                        writer.writerow([
                            datetime.utcnow().isoformat() + 'Z',
                            f"{mean_v:.2f}",
                            f"{mean_v_sky:.2f}",
                            f"{opacity:.2f}",
                            f"{mean_s:.2f}",
                            f"{flame_ratio:.6f}",
                            f"{blue_ratio:.6f}",  # NOUVEAU
                            f"{yellow_ratio:.6f}", # NOUVEAU
                            f"{red_ratio:.6f}",  # NOUVEAU
                            f"{black_ratio:.6f}",
                            f"{grey_ratio:.6f}",
                            diagnosis
                        ])
                        csvfile.flush()
                        last_csv_flush = tnow
                    except Exception:
                        logger.exception("Erreur écriture CSV")

                time.sleep(0.03) # approx 30 FPS max

    except Exception:
        logger.exception("Erreur fatale dans le thread d'analyse")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        CF.is_running = False
        logger.info("Thread d'analyse terminé")

# ---------- Control helpers ----------
def start_analysis(source, uploaded_file, state_ref, thresholds_ui):
    # stop any running
    if CF.is_running:
        stop_analysis()

    final_source = source
    is_uploaded = False
    temp_path = None
    csv_path = state_ref.get('csv_file', DEFAULT_CSV)

    if state_ref.get('analysis_mode') == 'Upload' and uploaded_file is not None:
        is_uploaded = True
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        final_source = temp_path
        csv_path = f"log_{uploaded_file.name}.csv"
        # MISE À JOUR CSV COLUMNS
        init_csv(csv_path)

    # initialize shared structures
    state_ref['opacity_deque'] = deque(maxlen=ROLLING_WINDOW)
    state_ref['last_frame_rgb'] = None
    state_ref['current_status'] = 'Unknown'
    state_ref['opacity_live'] = 0.0
    state_ref['mean_s_live'] = 0.0
    state_ref['flame_ratio'] = 0.0
    state_ref['blue_ratio'] = 0.0  # NOUVEAU
    state_ref['yellow_ratio'] = 0.0 # NOUVEAU
    state_ref['red_ratio'] = 0.0    # NOUVEAU
    state_ref['black_ratio'] = 0.0
    state_ref['grey_ratio'] = 0.0
    state_ref['csv_file'] = csv_path

    # build args (current UI values are used as default/manual setting)
    roi = (state_ref.get('x_roi', 200), state_ref.get('y_roi', 100), state_ref.get('w_roi', 300), state_ref.get('h_roi', 400))
    sky_roi = (state_ref.get('rsx', 10), state_ref.get('rsy', 10), state_ref.get('rsw', 100), state_ref.get('rsh', 50))

    CF.stop_event.clear()
    CF.is_running = True
    CF.thread = threading.Thread(
        target=analysis_loop,
        args=(final_source, roi, sky_roi, thresholds_ui, csv_path, state_ref, is_uploaded),
        daemon=True,
    )
    CF.thread.start()
    logger.info("Analyse démarrée")

def stop_analysis(wait: float = 2.0):
    if not CF.is_running:
        return
    CF.stop_event.set()
    if CF.thread and CF.thread.is_alive():
        CF.thread.join(timeout=wait)
    CF.is_running = False
    logger.info("Analyse arrêtée")

# ---------- Streamlit UI (MISES À JOUR) ----------
st.set_page_config(layout="wide", page_title=APP_TITLE)
st.title(APP_TITLE)

# session state container
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'analysis_mode': 'Live',
        'roi_mode': 'Manuel',
        'csv_file': DEFAULT_CSV,
        'x_roi': 200, 'y_roi': 100, 'w_roi': 300, 'h_roi': 400,
        'rsx': 10, 'rsy': 10, 'rsw': 100, 'rsh': 50,
        # thresholds (UI defaults)
        'flame_hsv_lower': [5, 100, 150], # YELLOW/ORANGE FLAME default
        'flame_hsv_upper': [35, 255, 255],
        'black_v': 80, 'black_s': 100,
        'grey_v_low': 80, 'grey_v_high': 220, 'grey_s': 80,
        'flame_area_frac': 0.002, 'black_frac': 0.03, 'grey_frac': 0.03,
        # NOUVEAUX RATIOS DE COULEUR
        'blue_ratio': 0.0, 'yellow_ratio': 0.0, 'red_ratio': 0.0,
    }

app = st.session_state.app_state

# Layout
col_cfg, col_vid, col_logs = st.columns([1, 3, 1])

# ... [La colonne col_cfg (configuration) reste inchangée] ...
with col_cfg:
    st.header("⚙️ Configuration")
    mode = st.radio("Mode Source", ["Live", "Upload"], index=0, key='mode_radio')
    app['analysis_mode'] = 'Upload' if mode == 'Upload' else 'Live'

    if app['analysis_mode'] == 'Live':
        source_input = st.text_input("Source (index/path)", value='0')
        uploaded = None
    else:
        source_input = st.text_input("Source (disabled)", value='N/A', disabled=True)
        uploaded = st.file_uploader("Fichier image/vidéo", type=['jpg','jpeg','png','mp4','avi','mov'])
        
    st.markdown("---")
    
    # NOUVELLE OPTION POUR L'AUTO-ROI
    roi_mode = st.radio("Mode ROI", ["Manuel", "Automatique"], index=0, key='roi_mode_radio', help="Manuel: Coordonnées fixées. Automatique: Détection dynamique de la flamme et du ciel.")
    app['roi_mode'] = roi_mode
    
    is_manual_roi = (app['roi_mode'] == 'Manuel')

    st.subheader("ROI Fumée")
    app['x_roi'] = st.number_input("X", value=int(app['x_roi']), disabled=not is_manual_roi)
    app['y_roi'] = st.number_input("Y", value=int(app['y_roi']), disabled=not is_manual_roi)
    app['w_roi'] = st.number_input("W", value=int(app['w_roi']), disabled=not is_manual_roi)
    app['h_roi'] = st.number_input("H", value=int(app['h_roi']), disabled=not is_manual_roi)

    st.subheader("ROI Ciel (normalisation)")
    app['rsx'] = st.number_input("rsx", value=int(app['rsx']), disabled=not is_manual_roi)
    app['rsy'] = st.number_input("rsy", value=int(app['rsy']), disabled=not is_manual_roi)
    app['rsw'] = st.number_input("rsw", value=int(app['rsw']), disabled=not is_manual_roi)
    app['rsh'] = st.number_input("rsh", value=int(app['rsh']), disabled=not is_manual_roi)

    st.markdown("---")
    st.subheader("Seuils : Flamme & Fumée")
    col1, col2 = st.columns(2)
    with col1:
        app['flame_area_frac'] = st.number_input("Seuil surf. flamme", value=float(app['flame_area_frac']), min_value=0.0, max_value=0.1, step=0.001)
        app['black_frac'] = st.number_input("Seuil fumée noire", value=float(app['black_frac']), min_value=0.0, max_value=0.5, step=0.005)
        app['grey_frac'] = st.number_input("Seuil fumée grise", value=float(app['grey_frac']), min_value=0.0, max_value=0.5, step=0.005)
    with col2:
        app['black_v'] = st.number_input("black: V threshold", value=int(app['black_v']), min_value=0, max_value=255)
        app['black_s'] = st.number_input("black: S threshold", value=int(app['black_s']), min_value=0, max_value=255)
        app['grey_v_low'] = st.number_input("grey: V low", value=int(app['grey_v_low']), min_value=0, max_value=255)
        app['grey_v_high'] = st.number_input("grey: V high", value=int(app['grey_v_high']), min_value=0, max_value=255)
        app['grey_s'] = st.number_input("grey: S threshold", value=int(app['grey_s']), min_value=0, max_value=255)

    st.subheader("HSV Flamme")
    # NOTE: Les seuils ici sont pour la détection générale (flame_ratio).
    # La classification par couleur (Bleu, Jaune/Orange, Rouge) utilise des seuils fixes internes pour la cohérence.
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        lower_h = st.number_input("Flame H lower (Gen)", value=int(app['flame_hsv_lower'][0]), min_value=0, max_value=179)
        lower_s = st.number_input("Flame S lower (Gen)", value=int(app['flame_hsv_lower'][1]), min_value=0, max_value=255)
        lower_v = st.number_input("Flame V lower (Gen)", value=int(app['flame_hsv_lower'][2]), min_value=0, max_value=255)
    with fcol2:
        upper_h = st.number_input("Flame H upper (Gen)", value=int(app['flame_hsv_upper'][0]), min_value=0, max_value=179)
        upper_s = st.number_input("Flame S upper (Gen)", value=int(app['flame_hsv_upper'][1]), min_value=0, max_value=255)
        upper_v = st.number_input("Flame V upper (Gen)", value=int(app['flame_hsv_upper'][2]), min_value=0, max_value=255)
    app['flame_hsv_lower'] = [lower_h, lower_s, lower_v]
    app['flame_hsv_upper'] = [upper_h, upper_s, upper_v]


    st.markdown("---")
    if st.button("▶️ Démarrer", disabled=CF.is_running):
        # convert source to int if digit
        try:
            src = int(source_input) if source_input.isdigit() else source_input
        except Exception:
            src = source_input
        thresholds_ui = {
            'flame_hsv_lower': app['flame_hsv_lower'],
            'flame_hsv_upper': app['flame_hsv_upper'],
            'black_v': app['black_v'],
            'black_s': app['black_s'],
            'grey_v_low': app['grey_v_low'],
            'grey_v_high': app['grey_v_high'],
            'grey_s': app['grey_s'],
            'flame_area_frac': app['flame_area_frac'],
            'black_frac': app['black_frac'],
            'grey_frac': app['grey_frac'],
        }
        start_analysis(src, uploaded if app['analysis_mode'] == 'Upload' else None, app, thresholds_ui)

    if st.button("⏹️ Arrêter", disabled=not CF.is_running):
        stop_analysis()

    if CF.is_running:
        st.info("Analyse en cours...")
    else:
        st.warning("Analyse arrêtée.")

# Vidéo / Frame
with col_vid:
    st.header("🎥 Flux Vidéo & Diagnostic")
    if app.get('last_frame_rgb') is not None:
        st.image(app['last_frame_rgb'], caption="Analyse en Temps Réel", use_column_width=True)
    else:
        st.image("Gaz-torche.jpg", caption="Image par Défaut (Torchère/Cheminée)", use_column_width=True)
        # 

    st.markdown("---")
    st.subheader("Statut Actuel")
    status = app.get('current_status', 'Inconnu')
    info = DIAGNOSTIC_INFO.get(status, DIAGNOSTIC_INFO["Unknown"])
    color_hex = f'rgb({info["Color"][2]}, {info["Color"][1]}, {info["Color"][0]})' # BGR to RGB for HTML
    
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {color_hex}; color: black;">
        <strong>Statut:</strong> {status}<br>
        <strong>Signification:</strong> {info["Signification"]}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Métriques Clés")
    st.write(f"**Opacité :** {app.get('opacity_live', 0.0):.1f} %")
    st.write(f"**Ratio Flamme Totale :** {app.get('flame_ratio', 0.0):.4f}")
    st.write(f"**Ratio Fumée Noire :** {app.get('black_ratio', 0.0):.4f}")
    st.write(f"**Ratio Fumée Grise :** {app.get('grey_ratio', 0.0):.4f}")

    st.markdown("---")
    st.subheader("Ratios de Couleur (Avancé)")
    st.write(f"**Ratio Bleu (Optimal) :** {app.get('blue_ratio', 0.0):.4f}")
    st.write(f"**Ratio Jaune/Orange (Incomplet) :** {app.get('yellow_ratio', 0.0):.4f}")
    st.write(f"**Ratio Rougeâtre (Instable/Impuretés) :** {app.get('red_ratio', 0.0):.4f}")


# Logs et Historique
with col_logs:
    st.header("📊 Historique")
    
    # Opacity Plot
    opacity_df = pd.DataFrame(list(app.get('opacity_deque', deque())), columns=['time', 'opacity'])
    if not opacity_df.empty:
        opacity_df.set_index('time', inplace=True)
        st.line_chart(opacity_df['opacity'], height=200)
    else:
        st.warning("Aucune donnée d'opacité à afficher.")

    st.markdown("---")

    # CSV Log Download
    csv_file = app.get('csv_file', DEFAULT_CSV)
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            st.download_button(
                label="Télécharger le Log CSV",
                data=f.read(),
                file_name=csv_file,
                mime="text/csv"
            )
        st.subheader("Dernières Entrées CSV")
        try:
            log_df = pd.read_csv(csv_file).tail(10)
            st.dataframe(log_df, use_container_width=True)
        except pd.errors.EmptyDataError:
            st.info("Le fichier CSV est vide.")
        except Exception as e:
            st.error(f"Erreur de lecture CSV: {e}")

    # Alerts Log
    st.subheader("Journal des Alertes")
    if os.path.exists(ALERTS_LOG_FILE):
        try:
            with open(ALERTS_LOG_FILE, "r") as f:
                alerts_text = f.read()
                st.text_area("Alertes", alerts_text, height=200)
        except Exception:
            st.info("Aucune alerte enregistrée.")

# Relancer le script pour rafraîchir la UI Streamlit
# Ce block est nécessaire pour que Streamlit puisse rafraîchir l'UI
if CF.is_running:
    time.sleep(1) # Ajoute un court délai pour laisser le thread d'analyse mettre à jour les données

    st.rerun()

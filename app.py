
def to_mono_lr(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve mono, left, right (float64). Acepta (n,) o (n,2)."""
    if audio is None or len(audio) == 0:
        z = np.zeros(1, dtype=np.float64)
        return z, z, z
    if audio.ndim == 1:
        left = audio.astype(np.float64)
        right = audio.astype(np.float64)
    else:
        left = audio[:, 0].astype(np.float64)
        right = audio[:, 1].astype(np.float64) if audio.shape[1] > 1 else audio[:, 0].astype(np.float64)
    mono = 0.5 * (left + right)
    return mono, left, right
# ============================================================
# ANALIZADOR (UI) - Analizador de Mastering con informe PDF
# Estilo informe PDF: WAVE MUSIC STUDIO
# Logo PDF (arriba izquierda): C:\PYTHON314\LOGO.PNG
# Logo WEB:                    LOGO.PNG
# Firma: Gerard Fortuny
# ============================================================

import io
import os
import math
import time
import json  # ✅ FIX: necesario para incrustar datos en el HTML del visualizador
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Audio I/O (con fallback)
import soundfile as sf
import librosa
import pyloudnorm as pyln

# UI + plots
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import welch

# Cover art
from mutagen import File as MutagenFile

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "LOGO.PNG")          # PDF (WAVE MUSIC STUDIO)
WEB_LOGO_PATH = os.path.join(BASE_DIR, "LOGO.PNG")  # WEB ✅ tu logo real
ENGINEER = "Gerard Fortuny"
STUDIO = "WAVE MUSIC STUDIO"

TARGETS = {
    "streaming_lufs_range": (-14.0, -9.0),
    "club_lufs_range": (-10.0, -7.0),
    "true_peak_max_dbtp": -1.0,
}

# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(page_title="ANALIZADOR", page_icon="🎛️", layout="wide")

# ✅ Contraste: tipografía clara + fondo pro
st.markdown("""
<style>
:root { color-scheme: dark; }

/* Fondo pro (oscuro) */
html, body, [data-testid="stAppViewContainer"] {
  background: radial-gradient(1200px 900px at 20% 10%, #0b2a4a 0%, #070b18 45%, #050713 100%) !important;
}

/* Texto general (alta legibilidad) */
html, body, [data-testid="stAppViewContainer"] * {
  color: #eaf2ff !important;
}

/* Header/sidebar */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
  background: rgba(10, 18, 35, 0.65) !important;
  border-right: 1px solid rgba(34,50,79,0.9);
}

.block-container { padding-top: 1.0rem; }

/* Cards */
.card{
  background: rgba(10, 18, 35, 0.78);
  border: 1px solid rgba(34,50,79,0.95);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.small{color:#b8c8eb !important;font-size:0.9rem;}
.badge{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid #2d416b;color:#d7e4ff !important;margin-right:6px;}
hr{border:0;height:1px;background:#22324f;margin:12px 0;}

/* Cabecera */
.brand-row{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom: 6px;}
.brand-logo{
  width: 44px; height: 44px; border-radius: 12px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(34,50,79,0.95);
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;
}
.brand-title{font-size: 28px;font-weight: 800;letter-spacing: 0.5px;margin: 0;}
.brand-sub{color:#b8c8eb !important;margin-top:-6px;}

/* Visualizador */
.eq-wrap{display:flex;gap:14px;align-items:stretch;margin-top: 10px;}
.eq-canvas{
  width: 100%;
  border-radius: 14px;
  border: 1px solid rgba(34,50,79,0.95);
  background: rgba(0,0,0,0.18);
}
.eq-side{ min-width: 260px; 
  min-width: 210px;
  border-radius: 14px;
  border: 1px solid rgba(34,50,79,0.95);
  background: rgba(0,0,0,0.18);
  padding: 12px;
}
.kpi-big{font-size: 30px;font-weight: 900;margin: 0;}
.kpi-label{color:#b8c8eb !important;font-size: 12px;margin-top: -4px;}
.kpi-note{color:#d7e4ff !important;font-size: 12px;margin-top: 10px;line-height: 1.35;}

/* ✅ Botones en verde (Browse / Analizar / Imprimir) */
.stButton > button, div.stDownloadButton > button {
  background: linear-gradient(180deg, rgba(34,197,94,1) 0%, rgba(22,163,74,1) 100%) !important;
  border: 1px solid rgba(16,120,50,1) !important;
  color: #f0fff4 !important;
  font-weight: 800 !important;
  border-radius: 12px !important;
}
.stButton > button:hover, div.stDownloadButton > button:hover {
  filter: brightness(1.08) !important;
}

/* ✅ Browse (file uploader) */
[data-testid="stFileUploader"] button {
  background: linear-gradient(180deg, rgba(34,197,94,1) 0%, rgba(22,163,74,1) 100%) !important;
  border: 1px solid rgba(16,120,50,1) !important;
  color: #f0fff4 !important;
  font-weight: 800 !important;
  border-radius: 12px !important;
}

/* ✅ Inputs blancos con texto negro */
input, textarea, select, [data-baseweb="input"] input {
  color: #0b1220 !important;
}
[data-baseweb="input"] {
  background-color: #ffffff !important;
  border-radius: 12px !important;
}
[data-baseweb="input"] * {
  color: #0b1220 !important;
}
div[data-baseweb="select"] > div {
  background-color: #ffffff !important;
  color: #0b1220 !important;
  border-radius: 12px !important;
}
div[data-baseweb="select"] * {
  color: #0b1220 !important;
}

/* Slider labels */
[data-testid="stSlider"] * {
  color: #eaf2ff !important;
}


/* Logo grande a la derecha */
.brand-right{  width: 200px; height: 200px; border-radius: 18px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(34,50,79,0.95);
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;
}
.brand-left{display:flex;align-items:center;gap:14px;}

</style>
""", unsafe_allow_html=True)

# ✅ Gráficas: estilo dark + colores visibles
plt.style.use("dark_background")
plt.rcParams.update({
    "axes.facecolor": "#0a1223",
    "figure.facecolor": "#0a1223",
    "axes.edgecolor": "#7ea0d6",
    "axes.labelcolor": "#eaf2ff",
    "xtick.color": "#eaf2ff",
    "ytick.color": "#eaf2ff",
    "text.color": "#eaf2ff",
    "grid.color": "#2a3b63",
    "grid.linestyle": ":",
})
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
    "#7dd3fc", "#a78bfa", "#34d399", "#fbbf24", "#fb7185", "#60a5fa"
])

# =========================
# DATA MODELS
# =========================
@dataclass
class KPIs:
    duration_s: float
    sr: int
    channels: int
    lufs_i: float
    lra: float
    true_peak_dbfs_approx: float
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    stereo_corr: float
    clipping_pct: float
    clipping_max_run: int
    dc_offset: float


@dataclass
class AnalysisResult:
    kpis: KPIs
    bands_db: Dict[str, float]
    issues: List[str]
    fixes: List[str]
    figs: Dict[str, plt.Figure]
    loudness_series: Optional[Tuple[np.ndarray, np.ndarray]]
    advanced: Dict[str, float]
  # (t, LUFS_short)


# =========================
# DSP HELPERS
# =========================
def _db(x: float, eps: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(abs(x), eps)))

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))

def crest_factor_db(x: np.ndarray) -> float:
    r = _rms(x)
    p = _peak(x)
    if r < 1e-12:
        return float("nan")
    return float(20.0 * np.log10(p / r))

def stereo_correlation(left: np.ndarray, right: np.ndarray) -> float:
    n = min(left.size, right.size)
    l = left[:n] - np.mean(left[:n])
    r = right[:n] - np.mean(right[:n])
    denom = np.std(l) * np.std(r)
    if denom < 1e-12:
        return 0.0
    return float(np.mean(l * r) / denom)

def detect_clipping(x: np.ndarray, thresh: float = 0.999) -> Tuple[float, int]:
    near = np.abs(x) >= thresh
    pct = float(100.0 * np.mean(near))
    max_run = 0
    run = 0
    for v in near:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return pct, int(max_run)

def band_energy_db(freq: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (freq >= f_lo) & (freq < f_hi)
    if not np.any(mask):
        return -200.0
    e = np.trapz(psd[mask], freq[mask])
    return float(10.0 * np.log10(max(e, 1e-20)))


# =========================
# COVER EXTRACTION
# =========================
def extract_cover_bytes(file_bytes: bytes, filename: str) -> Optional[bytes]:
    try:
        bio = io.BytesIO(file_bytes)
        audio = MutagenFile(bio, filename=filename)
        if audio is None:
            return None

        if getattr(audio, "tags", None):
            for k, v in audio.tags.items():
                if str(k).startswith("APIC") and hasattr(v, "data"):
                    return v.data

        if hasattr(audio, "pictures") and audio.pictures:
            return audio.pictures[0].data

        if getattr(audio, "tags", None) and "covr" in audio.tags:
            covr = audio.tags["covr"]
            if covr and len(covr) > 0:
                return bytes(covr[0])
    except Exception:
        return None

    return None


# =========================
# AUDIO LOADER (robusto)
# =========================
def load_audio_any(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, int]:
    bio = io.BytesIO(file_bytes)
    try:
        y, sr = sf.read(bio, always_2d=True)
        return y.astype(np.float64), int(sr)
    except Exception:
        bio2 = io.BytesIO(file_bytes)
        y_mono, sr = librosa.load(bio2, sr=None, mono=True)
        y_mono = y_mono.astype(np.float64)
        y = np.stack([y_mono, y_mono], axis=1)
        return y, int(sr)


# =========================
# PLOTS
# =========================
def fig_waveform(y: np.ndarray, sr: int, title: str, max_points: int = 200000) -> plt.Figure:
    n = len(y)
    if n > max_points:
        step = int(np.ceil(n / max_points))
        yv = y[::step]
        t = np.linspace(0, n / sr, num=len(yv), endpoint=False)
    else:
        yv = y
        t = np.linspace(0, n / sr, num=n, endpoint=False)

    fig = plt.figure()
    plt.plot(t, yv, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    return fig

def fig_spectrum(y: np.ndarray, sr: int, title: str) -> Tuple[plt.Figure, np.ndarray, np.ndarray]:
    f, pxx = welch(y, fs=sr, nperseg=min(8192, len(y)))
    fig = plt.figure()
    plt.semilogx(f, 10*np.log10(np.maximum(pxx, 1e-20)), linewidth=1.3)
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("dB")
    plt.grid(True, which="both")
    plt.xlim(20, min(20000, sr/2))
    return fig, f, pxx

def fig_spectrogram(y: np.ndarray, sr: int, title: str) -> plt.Figure:
    n_fft = 2048
    hop = 512
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    fig = plt.figure()
    plt.imshow(
        S_db, origin="lower", aspect="auto",
        extent=[0, len(y)/sr, 0, sr/2]
    )
    plt.ylim(0, 20000)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.colorbar(label="dB")
    return fig

def fig_band_bars(bands: Dict[str, float], title: str) -> plt.Figure:
    fig = plt.figure()
    names = list(bands.keys())
    vals = list(bands.values())
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=25, ha="right")
    plt.title(title)
    plt.ylabel("dB (aprox.)")
    plt.grid(True, axis="y")
    return fig

def fig_corr_over_time(left: np.ndarray, right: np.ndarray, sr: int, title: str) -> plt.Figure:
    win_s = 1.0
    hop_s = 0.25
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    n = min(len(left), len(right))
    left = left[:n]
    right = right[:n]
    vals = []
    ts = []
    for i in range(0, n - win, hop):
        c = stereo_correlation(left[i:i+win], right[i:i+win])
        vals.append(c)
        ts.append(i / sr)
    fig = plt.figure()
    plt.plot(ts, vals, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Correlación")
    plt.ylim(-1.0, 1.0)
    plt.grid(True)
    return fig

def fig_lufs_short_time(t: np.ndarray, lufs: np.ndarray, title: str) -> plt.Figure:
    fig = plt.figure()
    plt.plot(t, lufs, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("LUFS (short-term aprox.)")
    plt.grid(True)
    return fig

def fig_ms_spectrum(left: np.ndarray, right: np.ndarray, sr: int, title: str) -> plt.Figure:
    mid = 0.5*(left+right)
    side = 0.5*(left-right)
    f_m, p_m = welch(mid, fs=sr, nperseg=min(8192, len(mid)))
    f_s, p_s = welch(side, fs=sr, nperseg=min(8192, len(side)))
    fig = plt.figure()
    plt.semilogx(f_m, 10*np.log10(np.maximum(p_m, 1e-20)), label="Mid", linewidth=1.2)
    plt.semilogx(f_s, 10*np.log10(np.maximum(p_s, 1e-20)), label="Side", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("dB")
    plt.grid(True, which="both")
    plt.xlim(20, min(20000, sr/2))
    plt.legend()
    return fig


# =========================
# LOUDNESS SERIES
# =========================
def compute_short_term_lufs(meter: pyln.Meter, y: np.ndarray, sr: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    win_s = 3.0
    hop_s = 0.5
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    if len(y) < win:
        return None

    ts = []
    vals = []
    for i in range(0, len(y) - win, hop):
        seg = y[i:i+win]
        try:
            v = float(meter.integrated_loudness(seg))
        except Exception:
            v = float("nan")
        ts.append(i / sr)
        vals.append(v)

    return np.array(ts, dtype=np.float64), np.array(vals, dtype=np.float64)


# =========================
# ANALYSIS
# =========================


def estimate_bpm_key(mono: np.ndarray, sr: int) -> Tuple[Optional[float], Optional[str]]:
    """
    Estimación robusta de BPM y tonalidad (key) aproximada.
    - BPM: beat_track + fallback con tempogram.
    - Key: chroma_cqt con fallback a chroma_stft.
    """
    bpm: Optional[float] = None
    key: Optional[str] = None

    y = mono.astype(np.float32)
    # BPM (1) beat_track
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.isfinite(tempo) and tempo > 0:
            bpm = float(tempo)
    except Exception:
        bpm = None

    # BPM fallback (2) tempogram peak
    if bpm is None or bpm < 40 or bpm > 220:
        try:
            oenv = librosa.onset.onset_strength(y=y, sr=sr)
            tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
            tmean = tg.mean(axis=1)
            bpms = librosa.tempo_frequencies(len(tmean), sr=sr)
            bpm2 = float(bpms[int(np.argmax(tmean))])
            if 40 <= bpm2 <= 220:
                bpm = bpm2
        except Exception:
            pass

    # Key detection (Krumhansl-Schmuckler)
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    maj = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88], dtype=np.float64)
    minp = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17], dtype=np.float64)
    maj = maj / np.linalg.norm(maj)
    minp = minp / np.linalg.norm(minp)

    chroma = None
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=1024)
        except Exception:
            chroma = None

    if chroma is not None:
        try:
            cm = np.mean(chroma, axis=1)
            cm = cm / (np.linalg.norm(cm) + 1e-12)
            best_i = 0
            best_score = -1e9
            best_minor = False
            for i in range(12):
                smaj = float(np.dot(cm, np.roll(maj, i)))
                smin = float(np.dot(cm, np.roll(minp, i)))
                if smaj > best_score:
                    best_score = smaj
                    best_i = i
                    best_minor = False
                if smin > best_score:
                    best_score = smin
                    best_i = i
                    best_minor = True
            key = f"{names[best_i]} {'minor' if best_minor else 'major'}"
        except Exception:
            key = None

    return bpm, key
def analyze_audio(mono: np.ndarray, left: np.ndarray, right: np.ndarray, sr: int) -> AnalysisResult:
    duration = len(mono) / sr if sr > 0 else 0.0
    dc = float(np.mean(mono)) if len(mono) else 0.0

    meter = pyln.Meter(sr)
    try:
        lufs_i = float(meter.integrated_loudness(mono))
    except Exception:
        lufs_i = float("nan")

    try:
        lra = float(pyln.loudness_range(mono, meter))
    except Exception:
        lra = float("nan")

    # True peak aprox (oversampling x4)
    try:
        mono_os = librosa.resample(mono.astype(np.float32), orig_sr=sr_full, target_sr=sr_full*4, res_type="kaiser_best")
        true_peak = _db(float(np.max(np.abs(mono_os))))
    except Exception:
        true_peak = _db(_peak(mono))

    peak_dbfs = _db(_peak(mono))
    rms_dbfs = _db(_rms(mono))
    crest_db = crest_factor_db(mono)

    clip_pct, clip_run = detect_clipping(mono)
    corr = stereo_correlation(left, right)

    # Espectro y bandas
    spec_fig, f, pxx = fig_spectrum(mono, sr, "Espectro medio (Welch PSD)")
    bands = {
        "Sub 20–60": band_energy_db(f, pxx, 20, 60),
        "Bass 60–120": band_energy_db(f, pxx, 60, 120),
        "Low-Mids 120–500": band_energy_db(f, pxx, 120, 500),
        "Mids 500–2k": band_energy_db(f, pxx, 500, 2000),
        "High-Mids 2–6k": band_energy_db(f, pxx, 2000, 6000),
        "High 6–16k": band_energy_db(f, pxx, 6000, 16000),
    }
    band_vals = np.array(list(bands.values()), dtype=np.float64)
    band_mean = float(np.mean(band_vals)) if band_vals.size else -200.0

    # Métricas avanzadas (robustas)
    advanced: Dict[str, float] = {}
    try:
        sc = float(librosa.feature.spectral_centroid(y=mono.astype(np.float32), sr=sr).mean())
    except Exception:
        sc = float("nan")
    try:
        ro = float(librosa.feature.spectral_rolloff(y=mono.astype(np.float32), sr=sr, roll_percent=0.95).mean())
    except Exception:
        ro = float("nan")
    try:
        sfla = float(librosa.feature.spectral_flatness(y=mono.astype(np.float32)).mean())
    except Exception:
        sfla = float("nan")

    # M/S width
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = _rms(mid)
    side_rms = _rms(side)
    mid_db = _db(mid_rms)
    side_db = _db(side_rms)
    width_db = float(side_db - mid_db)

    # DR aprox por percentiles de RMS ventana 0.4s
    dr = float("nan")
    try:
        win = int(0.4 * sr)
        hop = int(0.2 * sr)
        if len(mono) >= win and win > 0:
            vals = []
            for i in range(0, len(mono) - win, hop):
                vals.append(_db(_rms(mono[i:i+win])))
            arr = np.array(vals, dtype=np.float64)
            if arr.size >= 10:
                dr = float(np.nanpercentile(arr, 95) - np.nanpercentile(arr, 10))
    except Exception:
        dr = float("nan")

    # Noise floor aprox (percentil 5 RMS ventana 0.2s)
    noise_floor = float("nan")
    try:
        win = int(0.2 * sr)
        hop = int(0.1 * sr)
        if len(mono) >= win and win > 0:
            vals = []
            for i in range(0, len(mono) - win, hop):
                vals.append(_db(_rms(mono[i:i+win])))
            arr = np.array(vals, dtype=np.float64)
            if arr.size:
                noise_floor = float(np.nanpercentile(arr, 5))
    except Exception:
        noise_floor = float("nan")

    # Densidad de transitorios (onsets/min)
    trans_per_min = float("nan")
    try:
        oenv = librosa.onset.onset_strength(y=mono.astype(np.float32), sr=sr, hop_length=512)
        onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=512)
        minutes = max((len(mono)/sr)/60.0, 1e-6)
        trans_per_min = float(len(onsets) / minutes)
    except Exception:
        trans_per_min = float("nan")

    # Guardamos advanced
    advanced["Spectral Centroid (Hz)"] = sc
    advanced["Spectral Rolloff 95% (Hz)"] = ro
    advanced["Spectral Flatness"] = sfla
    advanced["Width (Side-Mid, dB)"] = width_db
    advanced["DR aprox (dB)"] = dr
    advanced["Noise floor aprox (dBFS)"] = noise_floor
    advanced["Transitorios/min"] = trans_per_min

    issues: List[str] = []
    fixes: List[str] = []

    # Diagnóstico por bandas
    if bands["Sub 20–60"] > band_mean + 3:
        issues.append("Exceso de subgrave (20–60 Hz): riesgo de bombeo y pérdida de headroom.")
        fixes.append("HPF suave 20–30 Hz + EQ dinámica 30–60 Hz para controlar picos sin adelgazar el master.")
    if bands["Low-Mids 120–500"] > band_mean + 3:
        issues.append("Acumulación en 120–500 Hz: posible ‘bola’ y pérdida de claridad.")
        fixes.append("Recorte/EQ dinámica 180–300 Hz y control 200–450 Hz según se dispare la energía.")
    if bands["High-Mids 2–6k"] > band_mean + 3:
        issues.append("Exceso en 2–6 kHz: posible aspereza/fatiga.")
        fixes.append("EQ dinámica 2.5–4.5 kHz; revisar saturación/clipper y transitorios.")
    if bands["High 6–16k"] < band_mean - 4:
        issues.append("Falta de aire/brillo (6–16 kHz): el master puede percibirse apagado.")
        fixes.append("Shelf suave +0.5 a +2 dB desde 10–12 kHz o excitación armónica controlada (vigilando sibilancia).")

    # Loudness / picos
    if not math.isnan(lufs_i):
        if lufs_i > TARGETS["club_lufs_range"][1]:
            issues.append(f"Loudness muy alto (~{lufs_i:.1f} LUFS): riesgo de distorsión y pérdida de dinámica.")
            fixes.append("Reducir 1–2 dB de limitación; clipper suave antes del limitador; ceiling seguro; validar con oversampling.")
        elif lufs_i < TARGETS["streaming_lufs_range"][0]:
            issues.append(f"Loudness bajo (~{lufs_i:.1f} LUFS): puede sentirse flojo frente a referencias.")
            fixes.append("Aumentar ganancia hacia limitador controlando picos con clipper/limitador en oversampling.")

    if true_peak > TARGETS["true_peak_max_dbtp"]:
        issues.append(f"True Peak alto (~{true_peak:.1f} dBFS aprox.): riesgo de inter-sample peaks.")
        fixes.append("Ceiling del limitador a -1.0 dBTP (o -1.2 dBTP) y revisar cadena con oversampling.")

    if clip_pct > 0.05 or clip_run >= 3:
        issues.append(f"Posible clipping: {clip_pct:.4f}% de muestras cerca de 0 dBFS (racha máx {clip_run}).")
        fixes.append("Bajar entrada al clipper/limitador; oversampling; comprobar si el clipping es estético o un error.")

    # Estéreo / fase
    if corr < 0.0:
        issues.append(f"Correlación estéreo negativa ({corr:.2f}): riesgo de cancelación en mono.")
        fixes.append("Mantener <120 Hz en mono; reducir Haas/ensanchadores; revisar fase en midrange.")
    elif corr < 0.2:
        issues.append(f"Correlación estéreo baja ({corr:.2f}): posible inestabilidad en mono.")
        fixes.append("Reducir width en 200 Hz–2 kHz si hay desfase; asegurar sub/bajo centrados.")

    # Extra pro: dinámica / ruido / brillo
    if not math.isnan(dr) and dr < 6.0:
        issues.append(f"Dinámica baja (DR aprox. {dr:.1f} dB): master muy apretado.")
        fixes.append("Revisar compresión/limitación: 1–2 dB menos en limitador y recuperar transitorios (sin perder pegada).")
    if not math.isnan(noise_floor) and noise_floor > -45.0:
        issues.append(f"Ruido de fondo elevado (noise floor aprox. {noise_floor:.1f} dBFS).")
        fixes.append("Revisar emulaciones analógicas/ruido de plugins; limpiar con denoise leve si procede.")
    if not math.isnan(sc) and sc > 3200:
        issues.append(f"Balance tonal brillante (centroid {sc:.0f} Hz): posible dureza.")
        fixes.append("Suavizar 2–6 kHz con EQ dinámica y controlar saturación/clipper.")
    if not math.isnan(sc) and sc < 1100:
        issues.append(f"Balance tonal oscuro (centroid {sc:.0f} Hz): falta de definición.")
        fixes.append("Aportar aire con shelf 10–12 kHz y revisar acumulación low-mids.")

    if abs(dc) > 1e-3:
        issues.append(f"DC offset apreciable ({dc:.6f}): reduce headroom y afecta a dinámica.")
        fixes.append("Aplicar eliminación de DC / HPF muy bajo antes de dinámica final.")

    figs: Dict[str, plt.Figure] = {}
    figs["Waveform"] = fig_waveform(mono, sr, "Waveform (Mono)")
    figs["Espectrograma"] = fig_spectrogram(mono, sr, "Espectrograma (dB)")
    figs["Energía por bandas"] = fig_band_bars(bands, "Energía por bandas (aprox.)")
    figs["Espectro medio"] = spec_fig
    figs["Correlación (tiempo)"] = fig_corr_over_time(left, right, sr, "Correlación estéreo a lo largo del tiempo")

    st_series = compute_short_term_lufs(meter, mono, sr)
    if st_series is not None:
        t_l, l_l = st_series
        figs["LUFS (short-term)"] = fig_lufs_short_time(t_l, l_l, "LUFS short-term (aprox.)")

    # Figura M/S spectrum (ligera)
    try:
        figs["M/S Espectro"] = fig_ms_spectrum(left, right, sr, "Espectro Mid/Side")
    except Exception:
        pass

    kpis = KPIs(
        duration_s=duration,
        sr=sr,
        channels=2,
        lufs_i=lufs_i,
        lra=lra,
        true_peak_dbfs_approx=true_peak,
        peak_dbfs=peak_dbfs,
        rms_dbfs=rms_dbfs,
        crest_db=crest_db,
        stereo_corr=corr,
        clipping_pct=clip_pct,
        clipping_max_run=clip_run,
        dc_offset=dc
    )

    return AnalysisResult(
        kpis=kpis,
        bands_db=bands,
        issues=issues,
        fixes=fixes,
        figs=figs,
        loudness_series=st_series,
        advanced=advanced
    )


# =========================
# PDF RENDERER
# =========================
def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    bio.seek(0)
    return bio.read()

def render_pdf_wave_music_studio(
    track_title: str,
    client_name: str,
    result: AnalysisResult,
    cover_bytes: Optional[bytes],
    logo_path: str = LOGO_PATH,
    engineer_name: str = ENGINEER,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    def draw_header():
        if os.path.exists(logo_path):
            try:
                c.drawImage(ImageReader(logo_path), 2*cm, h - 3.2*cm, width=3.1*cm, height=3.1*cm, mask='auto')
            except Exception:
                pass

        c.setFont("Helvetica-Bold", 16)
        c.drawRightString(w - 2*cm, h - 2.3*cm, STUDIO)
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#6f87b5"))
        c.drawRightString(w - 2*cm, h - 2.85*cm, "Mastering • Mixing • Audio Analysis")
        c.setFillColor(colors.black)

        c.setStrokeColor(colors.HexColor("#22324f"))
        c.setLineWidth(1)
        c.line(2*cm, h - 3.55*cm, w - 2*cm, h - 3.55*cm)

    def draw_footer():
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(colors.HexColor("#6f87b5"))
        c.drawString(2*cm, 1.6*cm, f"Firmado: {engineer_name}")
        c.drawRightString(w - 2*cm, 1.6*cm, time.strftime("%d/%m/%Y %H:%M"))
        c.setFillColor(colors.black)

    # PAGE 1
    draw_header()
    c.setFont("Helvetica-Bold", 22)
    c.drawString(2*cm, h - 5.2*cm, "INFORME DE MASTERING")

    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#23314f"))
    c.drawString(2*cm, h - 6.2*cm, "Análisis técnico y plan de corrección")
    c.setFillColor(colors.black)

    if cover_bytes:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(cover_bytes)
                tmp_path = tmp.name
            # ✅ Portada arriba a la derecha, sin pisar el título
            c.drawImage(tmp_path, w - 7.2*cm, h - 8.8*cm, width=5.2*cm, height=5.2*cm,
                        preserveAspectRatio=True, mask='auto')
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        except Exception:
            pass
        except Exception:
            pass

    box_x, box_y = 2*cm, h - 12.0*cm
    box_w, box_h = (w - 4*cm) - (6.0*cm if cover_bytes else 0), 3.3*cm
    c.setStrokeColor(colors.HexColor("#22324f"))
    c.setFillColor(colors.HexColor("#f3f6ff"))
    c.rect(box_x, box_y, box_w, box_h, fill=1, stroke=1)
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(box_x + 0.6*cm, box_y + 2.1*cm, "TEMA:")
    c.setFont("Helvetica", 12)
    c.drawString(box_x + 3.0*cm, box_y + 2.1*cm, track_title)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(box_x + 0.6*cm, box_y + 1.1*cm, "CLIENTE:")
    c.setFont("Helvetica", 12)
    c.drawString(box_x + 3.0*cm, box_y + 1.1*cm, client_name if client_name else "—")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, 4.2*cm, "Ingeniero:")
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, 3.5*cm, engineer_name)

    draw_footer()
    c.showPage()

    # PAGE 2
    draw_header()
    k = result.kpis

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, h - 4.5*cm, "Informe técnico completo")

    y = h - 5.5*cm
    line = 0.62*cm

    def row(lbl, val):
        nonlocal y
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2*cm, y, lbl)
        c.setFont("Helvetica", 10)
        c.drawString(7.2*cm, y, val)
        y -= line

    row("Duración / SR:", f"{k.duration_s:.2f} s · {k.sr} Hz")
    row("LUFS integrado:", "n/a" if math.isnan(k.lufs_i) else f"{k.lufs_i:.1f} LUFS")
    row("LRA:", "n/a" if math.isnan(k.lra) else f"{k.lra:.1f} LU")
    row("True Peak (aprox):", f"{k.true_peak_dbfs_approx:.1f} dBFS")
    row("Peak / RMS:", f"{k.peak_dbfs:.1f} dBFS · {k.rms_dbfs:.1f} dBFS")
    row("Crest Factor:", f"{k.crest_db:.1f} dB")
    row("Correlación estéreo:", f"{k.stereo_corr:.2f}")
    row("Clipping:", f"{k.clipping_pct:.4f}% (racha {k.clipping_max_run})")
    row("DC offset:", f"{k.dc_offset:.6f}")
    # Métricas avanzadas
    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Métricas avanzadas")
    y -= 0.75*cm
    c.setFont("Helvetica", 10)
    adv = result.advanced if hasattr(result, "advanced") else {}
    def _fmt(v, nd=1):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "n/a"
            return f"{float(v):.{nd}f}"
        except Exception:
            return "n/a"

    lines = [
        ("DR aprox (dB)", _fmt(adv.get("DR aprox (dB)"), 1)),
        ("Width Side-Mid (dB)", _fmt(adv.get("Width (Side-Mid, dB)"), 1)),
        ("Spectral Centroid (Hz)", _fmt(adv.get("Spectral Centroid (Hz)"), 0)),
        ("Spectral Rolloff 95% (Hz)", _fmt(adv.get("Spectral Rolloff 95% (Hz)"), 0)),
        ("Spectral Flatness", _fmt(adv.get("Spectral Flatness"), 2)),
        ("Noise floor aprox (dBFS)", _fmt(adv.get("Noise floor aprox (dBFS)"), 1)),
        ("Transitorios/min", _fmt(adv.get("Transitorios/min"), 0)),
    ]
    for lbl, val in lines:
        c.drawString(2*cm, y, f"• {lbl}: {val}")
        y -= line


    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Diagnóstico (puntos a corregir)")
    y -= 0.75*cm
    c.setFont("Helvetica", 10)
    if not result.issues:
        c.drawString(2*cm, y, "• Sin incidencias críticas detectadas.")
        y -= line
    else:
        for it in result.issues[:12]:
            c.drawString(2*cm, y, f"• {it}")
            y -= line

    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Plan de corrección (acciones recomendadas)")
    y -= 0.75*cm
    c.setFont("Helvetica", 10)
    if not result.fixes:
        c.drawString(2*cm, y, "• Sin acciones urgentes.")
        y -= line
    else:
        for it in result.fixes[:12]:
            c.drawString(2*cm, y, f"• {it}")
            y -= line

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, 2.9*cm, "Firma:")
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, 2.3*cm, engineer_name)

    draw_footer()
    c.showPage()

    # PLOTS
    fig_images = {name: fig_to_png_bytes(fig) for name, fig in result.figs.items()}

    def add_plot_page(title_text: str, img_bytes: bytes):
        draw_header()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, h - 4.5*cm, title_text)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        margin_x = 2*cm
        margin_y = 2*cm
        max_w = w - 2*margin_x
        max_h = h - 7.0*cm
        c.drawImage(tmp_path, margin_x, margin_y, width=max_w, height=max_h, preserveAspectRatio=True, anchor='c')

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        draw_footer()
        c.showPage()

    order = ["Waveform", "LUFS (short-term)", "Espectro medio", "Energía por bandas", "Espectrograma", "Correlación (tiempo)"]
    for key in order:
        if key in fig_images:
            add_plot_page(key, fig_images[key])

    c.save()
    buf.seek(0)
    return buf.read()


# =========================
# HTML PLAYER + EQ VISUALIZER (WebAudio)
# =========================
def render_player_with_eq(audio_bytes: bytes, mime: str, lufs_t: Optional[np.ndarray], lufs_v: Optional[np.ndarray], lufs_max: Optional[float], bpm: Optional[float], key_str: Optional[str], peak_dbfs: Optional[float], rms_dbfs: Optional[float], tp_dbfs: Optional[float]) -> None:
    import base64
    import streamlit.components.v1 as components

    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    lufs_t_list = lufs_t.tolist() if lufs_t is not None else []
    lufs_v_list = lufs_v.tolist() if lufs_v is not None else []

    payload = {"lufs_t": lufs_t_list, "lufs_v": lufs_v_list, "lufs_max": (None if lufs_max is None else float(lufs_max)), "bpm": (None if bpm is None else float(bpm)), "key": (None if key_str is None else str(key_str)), "peak_dbfs": (None if peak_dbfs is None else float(peak_dbfs)), "rms_dbfs": (None if rms_dbfs is None else float(rms_dbfs)), "tp_dbfs": (None if tp_dbfs is None else float(tp_dbfs))}

    html = f"""
<style>
  body{{margin:0;padding:0;background:transparent;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;}}
  .eq-wrap{{display:flex;gap:14px;align-items:stretch;margin-top:10px;}}
  .eq-canvas{{width:100%;border-radius:14px;border:1px solid rgba(34,50,79,0.95);background:rgba(0,0,0,0.10);margin-top:10px;}}
  .eq-side{{min-width:260px;border-radius:14px;border:1px solid rgba(34,50,79,0.95);background:rgba(0,0,0,0.18);padding:12px;}}
  .kpi-big{{font-size:30px;font-weight:900;margin:0;color:#eaf2ff;}}
  .kpi-label{{color:#b8c8eb;font-size:12px;margin-top:-4px;}}
  .kpi-note{{color:#d7e4ff;font-size:12px;margin-top:10px;line-height:1.35;}}
  hr{{border:0;border-top:1px solid rgba(34,50,79,0.75);margin:10px 0;}}
</style>

<div class="eq-wrap">

  <div style="flex: 1;">
    <audio id="player" controls style="width: 100%; border-radius: 12px;"></audio>
    <canvas id="viz" class="eq-canvas" height="190"></canvas>
  </div>
  <div class="eq-side">
    <div class="kpi-label">LUFS (short-term)</div>
    <div id="lufsVal" class="kpi-big">—</div>
    <div class="kpi-label">aprox. (ventana 3s)</div>
    <hr/>
    <div class="kpi-label">LUFS (MAX)</div>
    <div id="lufsMax" class="kpi-big">—</div>
    <div class="kpi-label">máximo alcanzado</div>
    <hr/>
    <div class="kpi-label">BPM</div>
    <div id="bpmVal" class="kpi-big">—</div>
    <div class="kpi-label">Tempo estimado</div>
    <hr/>
    <div class="kpi-label">TONALIDAD</div>
    <div id="keyVal" class="kpi-big">—</div>
    <div class="kpi-label">Key estimada</div>
    <div class="kpi-note">
      Visualizador en tiempo real (frecuencia).<br/>
      LUFS sincronizado con el reproductor.
    </div>
  </div>
</div>

<script>
(() => {{
  const audio = document.getElementById('player');
  audio.src = "data:{mime};base64,{b64}";

  const data = {json.dumps(payload)};
  const lufsT = data.lufs_t;
  const lufsV = data.lufs_v;

  const lufsMaxVal = data.lufs_max;
  const lufsMaxEl = document.getElementById('lufsMax');
  lufsMaxEl.textContent = (lufsMaxVal === null || lufsMaxVal === undefined || !isFinite(lufsMaxVal)) ? '—' : Number(lufsMaxVal).toFixed(1);


  const canvas = document.getElementById('viz');
  const ctx = canvas.getContext('2d');

  let audioCtx = null;
  let analyser = null;
  let source = null;
  let freqData = null;

  function nearestLUFS(t) {{
    if (!lufsT || lufsT.length === 0) return null;
    let lo = 0, hi = lufsT.length - 1;
    while (lo < hi) {{
      const mid = Math.floor((lo + hi) / 2);
      if (lufsT[mid] < t) lo = mid + 1;
      else hi = mid;
    }}
    const i = lo;
    const j = Math.max(0, i - 1);
    const pick = (Math.abs(lufsT[i] - t) < Math.abs(lufsT[j] - t)) ? i : j;
    const v = lufsV[pick];
    if (v === null || v === undefined || !isFinite(v)) return null;
    return v;
  }}

  function ensureAudioGraph() {{
    if (audioCtx) return;
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.85;
    freqData = new Uint8Array(analyser.frequencyBinCount);

    source = audioCtx.createMediaElementSource(audio);
    source.connect(analyser);
    analyser.connect(audioCtx.destination);
  }}

  function resizeCanvas() {{
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}

  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const lufsEl = document.getElementById('lufsVal');

  function draw() {{
    requestAnimationFrame(draw);

    const t = audio.currentTime || 0;
    const v = nearestLUFS(t);
    lufsEl.textContent = (v === null) ? '—' : v.toFixed(1);

    if (!analyser) {{
      ctx.clearRect(0,0,canvas.width,canvas.height);
      return;
    }}

    analyser.getByteFrequencyData(freqData);

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);

    // fondo sutil
    ctx.fillStyle = 'rgba(0,0,0,0.10)';
    ctx.fillRect(0,0,width,height);

    const bars = 80;
    const step = Math.floor(freqData.length / bars);
    const barW = width / bars;

    for (let i=0; i<bars; i++) {{
      const idx = i*step;
      const val = freqData[idx] / 255.0;
      const barH = Math.max(2, val * height);

      const hue = 195 + (i / bars) * 140;  // azul -> magenta
      ctx.fillStyle = `hsla(${{hue}}, 95%, 60%, 0.95)`;

      const x = i * barW;
      const y = height - barH;
      ctx.fillRect(x, y, barW*0.78, barH);

      ctx.fillStyle = `hsla(${{hue}}, 95%, 70%, 0.25)`;
      ctx.fillRect(x, y-3, barW*0.78, 3);
    }}
  }}

  audio.addEventListener('play', async () => {{
    ensureAudioGraph();
    try {{
      if (audioCtx.state === 'suspended') await audioCtx.resume();
    }} catch(e) {{}}
  }});

  draw();
}})();
</script>
"""
    components.html(html, height=520)


# =========================
# HEADER (logo + título)
# =========================
def show_header():
    bw_logo = None
    if os.path.exists(WEB_LOGO_PATH):
        try:
            with open(WEB_LOGO_PATH, "rb") as f:
                bw_logo = f.read()
        except Exception:
            bw_logo = None

    if bw_logo:
        import base64
        b64 = base64.b64encode(bw_logo).decode("utf-8")
        st.markdown(f"""
<div class="brand-row">
  <div class="brand-left">
    <div>
      <div class="brand-title">ANALIZADOR</div>
      <div class="brand-sub">Mastering Analysis · {STUDIO}</div>
    </div>
  </div>
  <div class="brand-right">
    <img src="data:image/png;base64,{b64}" style="width:100%;height:100%;object-fit:contain;padding:6px"/>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="brand-row">
  <div class="brand-left">
    <div>
      <div class="brand-title">ANALIZADOR</div>
      <div class="brand-sub">Mastering Analysis · WAVE MUSIC STUDIO</div>
    </div>
  </div>
  <div class="brand-right">🎛️</div>
</div>
""", unsafe_allow_html=True)

        st.warning(f"No encuentro el logo en: {WEB_LOGO_PATH}")


# =========================
# APP
# =========================
show_header()
client_name = ""

# =========================
# MANUAL / AYUDA
# =========================
with st.expander("📘 MANUAL DEL ANALIZADOR (qué significa cada métrica)"):
    st.markdown("""
**LUFS (I)**: loudness integrado (volumen percibido total).  
**LUFS (short-term)**: loudness por ventana (~3s), útil para ver secciones “apretadas”.  
**LUFS (MAX)**: máximo de LUFS short-term alcanzado (control rápido).  
**True Peak**: pico inter-muestra estimado (mejor mantener ≤ -1.0 dBTP).  
**Peak / RMS / Crest**: pico / nivel medio / diferencia pico–medio (más crest = más transitorio).  
**LRA**: rango de loudness (macro-dinámica).  
**Correlación estéreo**: estabilidad de fase (-1 a +1). Negativo = cancelación en mono.  

### Métricas avanzadas
**Spectral Centroid**: “centro de brillo” (bajo = oscuro, alto = brillante).  
**Spectral Rolloff 95%**: extensión de agudos (dónde cae la energía).  
**Spectral Flatness**: planitud/ruidosidad (hiss/distorsión suele subirla).  
**Width (Side-Mid)**: cuánto SIDE domina sobre MID (ancho excesivo puede dar problemas en mono).  
**DR aprox**: dinámica aproximada por percentiles de RMS (orientativo).  
**Noise floor aprox**: estimación del ruido de fondo (colas/reverb/hiss).  
**Transitorios/min**: densidad de golpes (si es muy baja puede estar “aplastado”).  

> Nota: el análisis es técnico y orientativo. La decisión final depende del estilo (club/streaming/vinilo) y de la referencia.
""")

st.write("Sube **una sola pista** (máx. 80 MB) → escucha la previsualización → pulsa **ANALIZAR** → genera y descarga tu **informe PDF**.")

# Estado de pista cargada (para evitar reemplazos accidentales y ocultar el uploader)
if "track_bytes" not in st.session_state:
    st.session_state.track_bytes = None
    st.session_state.track_name = None
    st.session_state.track_type = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
    st.session_state.pdf_name = None

left, right = st.columns([1.05, 1.95], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Cargar pista")

    if st.session_state.track_bytes is None:
        uploaded = st.file_uploader(
            "Audio (WAV/FLAC/MP3/M4A/OGG/AIFF) · una sola pista · máximo 80 MB",
            type=["wav", "flac", "mp3", "m4a", "ogg", "aiff"],
            accept_multiple_files=False,
            key="main_uploader",
            help="Sube una sola pista por análisis. Tamaño máximo recomendado: 80 MB."
        )
        if uploaded is not None:
            audio_bytes_tmp = uploaded.getvalue()
            if len(audio_bytes_tmp) > 80 * 1024 * 1024:
                st.error("El archivo supera el límite de 80 MB. Sube una pista más ligera o exportada a menor peso.")
                st.stop()
            st.session_state.track_bytes = audio_bytes_tmp
            st.session_state.track_name = uploaded.name
            st.session_state.track_type = uploaded.type or "audio/wav"
            st.session_state.pdf_bytes = None
            st.session_state.pdf_name = None
            st.rerun()
    else:
        st.success(f"Pista cargada: {st.session_state.track_name}")
        st.caption("Para cargar otra pista, primero pulsa en ‘Quitar pista y empezar de nuevo’. Así evitamos reemplazos accidentales.")
        if st.button("🗑️ Quitar pista y empezar de nuevo", use_container_width=True):
            st.session_state.track_bytes = None
            st.session_state.track_name = None
            st.session_state.track_type = None
            st.session_state.pdf_bytes = None
            st.session_state.pdf_name = None
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**Tramo a analizar (opcional)**")
    st.markdown('<div class="small">Puedes analizar toda la pista o seleccionar un tramo concreto. Si lo dejas completo, el análisis usará el archivo entero.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Variable de compatibilidad para el resto de la interfaz
uploaded = st.session_state.track_bytes is not None

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎧 Reproductor, portada y EQ")

    analyze_btn = False
    cover_bytes: Optional[bytes] = None
    start_s = 0.0
    end_s = 0.0

    if uploaded:
        audio_bytes = st.session_state.track_bytes
        track_name = st.session_state.track_name
        track_type = st.session_state.track_type or "audio/wav"

        embedded_cover = extract_cover_bytes(audio_bytes, track_name)
        cover_bytes = embedded_cover

        c1, c2 = st.columns([1, 2.2], gap="large")
        with c1:
            st.markdown("**Portada**")
            if cover_bytes:
                st.image(cover_bytes, use_container_width=True)
            else:
                st.info("No se detecta portada embebida en el archivo.")

        # Precompute LUFS series for synced display (quick)
        y_full, sr_full = load_audio_any(audio_bytes, track_name)
        l_full = y_full[:, 0]
        r_full = y_full[:, 1] if y_full.shape[1] > 1 else y_full[:, 0]
        mono_full = 0.5 * (l_full + r_full)
        meter_full = pyln.Meter(sr_full)
        st_series = compute_short_term_lufs(meter_full, mono_full, sr_full)
        lufs_t = st_series[0] if st_series is not None else None
        lufs_v = st_series[1] if st_series is not None else None

        # ✅ LUFS máximo (short-term) ignorando NaN
        lufs_max = None
        if lufs_v is not None and len(lufs_v) > 0:
            vv = np.array(lufs_v, dtype=np.float64)
            if np.isfinite(vv).any():
                lufs_max = float(np.nanmax(vv))

        # ✅ BPM / Tonalidad (estimación)
        mono, left, right = to_mono_lr(y_full)
        bpm_est, key_est = estimate_bpm_key(mono, sr_full)

        with c2:
            # ✅ Medidores rápidos (para VU meters)
            peak_dbfs = _db(_peak(mono))
            rms_dbfs = _db(_rms(mono))
            try:
                mono_os = librosa.resample(mono.astype(np.float32), orig_sr=sr_full, target_sr=sr_full*4, res_type="kaiser_best")
                tp_dbfs = _db(float(np.max(np.abs(mono_os))))
            except Exception:
                tp_dbfs = peak_dbfs

            render_player_with_eq(audio_bytes, track_type, lufs_t, lufs_v, lufs_max, bpm_est, key_est, peak_dbfs, rms_dbfs, tp_dbfs)

        dur = len(mono_full) / sr_full
        start_s, end_s = st.slider(
            "Selecciona tramo a analizar (segundos)",
            min_value=0.0,
            max_value=float(dur),
            value=(0.0, float(dur)),
            step=0.1
        )

        st.markdown("**Waveform (al cargar)**")
        st.pyplot(fig_waveform(mono_full, sr_full, "Waveform (Mono) - Vista previa"), use_container_width=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        colA, colB = st.columns([1, 1], gap="large")
        with colA:
            analyze_btn = st.button("🔎 ANALIZAR", use_container_width=True, type="primary")
        with colB:
            st.markdown('<span class="badge">EQ LIVE</span><span class="badge">PDF</span><span class="badge">LUFS</span>', unsafe_allow_html=True)
    else:
        st.info("Sube una sola pista para ver la previsualización, la forma de onda y activar el análisis técnico.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ANALYSIS + REPORT
# =========================
if uploaded and analyze_btn:
    audio_bytes = st.session_state.track_bytes
    track_name = st.session_state.track_name
    y, sr = load_audio_any(audio_bytes, track_name)
    l = y[:, 0]
    r = y[:, 1] if y.shape[1] > 1 else y[:, 0]
    mono = 0.5 * (l + r)

    ss = float(start_s)
    ee = float(end_s)
    i0 = max(0, int(ss * sr))
    i1 = min(len(mono), int(ee * sr))
    if i1 <= i0 + sr:
        i0 = 0
        i1 = len(mono)

    mono_seg = mono[i0:i1]
    l_seg = l[i0:i1]
    r_seg = r[i0:i1]

    st.markdown("## 📊 Resultados del análisis")
    with st.spinner("Analizando master (KPIs + espectro + loudness + estéreo + diagnóstico)..."):
        result = analyze_audio(mono_seg, l_seg, r_seg, sr)


    k = result.kpis

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("KPIs (Master)")
    row1 = st.columns(6)
    row1[0].metric("Duración", f"{k.duration_s:.2f} s")
    row1[1].metric("SR", f"{k.sr} Hz")
    row1[2].metric("LUFS (I)", "n/a" if math.isnan(k.lufs_i) else f"{k.lufs_i:.1f}")
    row1[3].metric("LRA", "n/a" if math.isnan(k.lra) else f"{k.lra:.1f}")
    row1[4].metric("True Peak*", f"{k.true_peak_dbfs_approx:.1f} dBFS")
    row1[5].metric("Stereo Corr", f"{k.stereo_corr:.2f}")

    row2 = st.columns(6)
    row2[0].metric("Peak", f"{k.peak_dbfs:.1f} dBFS")
    row2[1].metric("RMS", f"{k.rms_dbfs:.1f} dBFS")
    row2[2].metric("Crest", f"{k.crest_db:.1f} dB")
    row2[3].metric("Clipping", f"{k.clipping_pct:.4f}%")
    row2[4].metric("Clipping run", f"{k.clipping_max_run}")
    row2[5].metric("DC offset", f"{k.dc_offset:.6f}")

    st.caption("*True Peak aproximado (oversampling x4).")

    st.markdown("### Métricas avanzadas")
    adv = result.advanced
    r3 = st.columns(4)
    r3[0].metric("DR aprox", "n/a" if math.isnan(adv.get("DR aprox (dB)", float("nan"))) else f"{adv.get('DR aprox (dB)'):.1f} dB")
    r3[1].metric("Width (dB)", "n/a" if math.isnan(adv.get("Width (Side-Mid, dB)", float("nan"))) else f"{adv.get('Width (Side-Mid, dB)'):.1f}")
    r3[2].metric("Noise floor", "n/a" if math.isnan(adv.get("Noise floor aprox (dBFS)", float("nan"))) else f"{adv.get('Noise floor aprox (dBFS)'):.1f} dBFS")
    r3[3].metric("Trans/min", "n/a" if math.isnan(adv.get("Transitorios/min", float("nan"))) else f"{adv.get('Transitorios/min'):.0f}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Diagnóstico y plan de corrección")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ❌ Qué falla")
        if result.issues:
            for it in result.issues:
                st.write(f"- {it}")
        else:
            st.write("- Sin incidencias críticas detectadas.")
    with c2:
        st.markdown("### ✅ Qué haría yo en el master")
        if result.fixes:
            for it in result.fixes:
                st.write(f"- {it}")
        else:
            st.write("- Sin acciones urgentes.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Gráficas (Master)")
    g1, g2 = st.columns(2)
    with g1:
        if "LUFS (short-term)" in result.figs:
            st.pyplot(result.figs["LUFS (short-term)"], use_container_width=True)
        st.pyplot(result.figs["Espectro medio"], use_container_width=True)
        cols_ms_bands = st.columns(2)
        with cols_ms_bands[0]:
            if "M/S Espectro" in result.figs:
                st.pyplot(result.figs["M/S Espectro"], use_container_width=True)
            else:
                st.info("M/S Espectro no disponible para este archivo.")
        with cols_ms_bands[1]:
            if "Energía por bandas" in result.figs:
                st.pyplot(result.figs["Energía por bandas"], use_container_width=True)
            else:
                st.info("Energía por bandas no disponible.")

        if "M/S Espectro" in result.figs:
            st.pyplot(result.figs["M/S Espectro"], use_container_width=True)
    with g2:
        st.pyplot(result.figs["Espectrograma"], use_container_width=True)
        st.pyplot(result.figs["Correlación (tiempo)"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🖨️ Descargar informe (PDF)")
    st.caption("El informe PDF se genera automáticamente al terminar el análisis.")

    embedded_cover = extract_cover_bytes(audio_bytes, track_name)
    final_cover = embedded_cover

    pdf_bytes = render_pdf_wave_music_studio(
        track_title=os.path.splitext(track_name)[0],
        client_name=client_name,
        result=result,
        cover_bytes=final_cover,
        logo_path=LOGO_PATH,
        engineer_name=ENGINEER
    )

    st.session_state.pdf_bytes = pdf_bytes
    st.session_state.pdf_name = f"{os.path.splitext(track_name)[0]}_WAVE_MUSIC_STUDIO_Report.pdf"

    st.download_button(
        label="🖨️ DESCARGAR INFORME PDF",
        data=st.session_state.pdf_bytes,
        file_name=st.session_state.pdf_name,
        mime="application/pdf",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Mantener visible la descarga del PDF aunque la app se vuelva a ejecutar
if st.session_state.get("pdf_bytes") and not analyze_btn:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🖨️ Descargar informe (PDF)")
    st.caption("Tu informe ya está listo. Puedes descargarlo de nuevo desde aquí.")
    st.download_button(
        label="🖨️ DESCARGAR INFORME PDF",
        data=st.session_state.pdf_bytes,
        file_name=st.session_state.pdf_name or "WAVE_MUSIC_STUDIO_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
        key="persistent_pdf_download"
    )
    st.markdown('</div>', unsafe_allow_html=True)

#!/usr/bin/env python3
"""
Brutal Matrix Sequencer  ·  16×16 grid  ·  one playhead  ·  multi-instrument
Paint notes with different instruments. Hit SPACE to play.
"""

import math, threading, time, signal, sys
import numpy as np
import pygame
import sounddevice as sd
from scipy.signal import lfilter

# Let Ctrl-C kill the process even with background audio threads
signal.signal(signal.SIGINT, signal.SIG_DFL)

# ══════════════════════════════════════════════════════════════════════════════
#  Audio engine — lockless double-buffer voice pool
# ══════════════════════════════════════════════════════════════════════════════
SR = 44100
MAX_VOICES = 64

_vlock   = threading.Lock()
_voices  = []   # [{'buf': ndarray(N,2) float32, 'pos': int}]


def _audio_cb(outdata, frames, _t, _status):
    out = np.zeros((frames, 2), dtype=np.float32)
    with _vlock:
        live = []
        for v in _voices:
            b, p = v['buf'], v['pos']
            chunk = b[p : p + frames]
            n = len(chunk)
            if n:
                out[:n] += chunk
            if p + frames < len(b):
                v['pos'] = p + frames
                live.append(v)
        _voices.clear()
        _voices.extend(live)
    np.clip(out, -1.0, 1.0, out=out)
    outdata[:] = out


_stream = None   # started inside main() after pygame init


def start_audio():
    global _stream
    try:
        _stream = sd.OutputStream(
            samplerate=SR, channels=2, dtype='float32',
            blocksize=512, callback=_audio_cb
        )
        _stream.start()
        return True
    except Exception as e:
        print(f"Audio init failed: {e}\nRunning silent.", file=sys.stderr)
        return False


def voice_play(mono: np.ndarray, vol: float = 0.80):
    s = np.column_stack([mono, mono]).astype(np.float32) * (vol * 0.5)
    with _vlock:
        if len(_voices) < MAX_VOICES:
            _voices.append({'buf': s, 'pos': 0})


# ══════════════════════════════════════════════════════════════════════════════
#  Synthesis — one function per instrument
# ══════════════════════════════════════════════════════════════════════════════

def _ar(n, attack, release):
    e = np.ones(n, dtype=np.float32)
    a = min(int(SR * attack), n)
    r = min(int(SR * release), n)
    if a: e[:a]  = np.linspace(0, 1, a)
    if r: e[-r:] = np.linspace(1, 0, r)
    return e

def _lp(sig, cutoff):
    """Single-pole IIR lowpass via scipy lfilter."""
    a1 = math.exp(-2 * math.pi * cutoff / SR)
    return lfilter([1 - a1], [1, -a1], sig).astype(np.float32)


def synth_glass_bell(freq):
    dur, n = 2.0, int(SR * 2.0)
    t = np.linspace(0, dur, n, endpoint=False)
    # Inharmonic partials — real bells are not harmonic
    wave = (0.55 * np.sin(2*np.pi*freq*t)
          + 0.22 * np.sin(2*np.pi*freq*2.76*t)
          + 0.10 * np.sin(2*np.pi*freq*5.40*t)
          + 0.07 * np.sin(2*np.pi*freq*8.93*t))
    env = np.exp(-t * 2.4) * _ar(n, 0.003, 0)
    return (wave * env * 0.65).astype(np.float32)


def synth_c64_lead(freq):
    dur, n = 0.28, int(SR * 0.28)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = np.where((t * freq % 1.0) < 0.25, 1.0, -1.0).astype(np.float32)
    wave = _lp(wave, 3200)
    return (wave * _ar(n, 0.005, 0.05) * 0.45).astype(np.float32)


def synth_c64_bass(freq):
    # Force into bass register regardless of row
    freq = max(freq * 0.5, 55.0)
    dur, n = 0.40, int(SR * 0.40)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (2 * ((t * freq) % 1.0) - 1.0).astype(np.float32)
    wave = _lp(wave, 700)
    return (wave * _ar(n, 0.008, 0.07) * 0.55).astype(np.float32)


def synth_c64_arp(freq):
    dur, n = 0.12, int(SR * 0.12)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = np.where((t * freq % 1.0) < 0.5, 1.0, -1.0).astype(np.float32)
    wave = _lp(wave, 4000)
    return (wave * _ar(n, 0.003, 0.03) * 0.40).astype(np.float32)


def synth_c64_bell(freq):
    dur, n = 1.2, int(SR * 1.2)
    t = np.linspace(0, dur, n, endpoint=False)
    tri = 2 * np.abs(2 * ((t * freq) % 1.0) - 1.0) - 1.0
    env = np.exp(-t * 3.0) * _ar(n, 0.002, 0)
    return (tri * env * 0.50).astype(np.float32)


def synth_rhodes(freq):
    dur, n = 1.1, int(SR * 1.1)
    t = np.linspace(0, dur, n, endpoint=False)
    mi = np.exp(-t * 3.5) * 5.0
    mod = np.sin(2*np.pi * freq * t)
    car = np.sin(2*np.pi * freq * t + mi * mod)
    env = np.exp(-t * 2.0) * _ar(n, 0.004, 0)
    return (car * env * 0.55).astype(np.float32)


def synth_kalimba(freq):
    """Karplus-Strong via scipy lfilter."""
    delay = max(2, int(round(SR / freq)))
    dur = 1.6
    n = int(SR * dur)
    rng = np.random.default_rng(int(freq * 10))
    burst = np.zeros(n, dtype=np.float32)
    burst[:delay] = rng.standard_normal(delay).astype(np.float32)
    a = np.zeros(delay + 2, dtype=np.float64)
    a[0] = 1.0; a[delay] = -0.4985; a[delay + 1] = -0.4985
    wave = lfilter([1.0], a, burst).astype(np.float32)
    env = np.exp(-np.linspace(0, 3.5, n))
    return (wave * env * 0.80).astype(np.float32)


def synth_didgeridoo(freq):
    freq = min(freq, 220.0)   # keep it bassy
    dur, n = 0.65, int(SR * 0.65)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (0.50 * np.sin(2*np.pi*freq*t)
          + 0.28 * np.sin(2*np.pi*freq*2*t)
          + 0.14 * np.sin(2*np.pi*freq*3*t)
          + 0.06 * np.sin(2*np.pi*freq*5*t))
    tremolo = 0.75 + 0.25 * np.sin(2*np.pi * 8 * t)
    return (wave * tremolo * _ar(n, 0.025, 0.12) * 0.52).astype(np.float32)


def synth_drum(row):
    """Percussion: row 0=top=cymbal, row 15=bottom=kick."""
    rng = np.random.default_rng(row * 7 + 13)
    if row >= 12:         # Kick
        dur = 0.50; n = int(SR * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        base = 80 + (15 - row) * 20
        pitch_env = base * np.exp(-t * 20)
        wave = np.sin(2*np.pi * np.cumsum(pitch_env / SR))
        noise = rng.standard_normal(n).astype(np.float32) * 0.07
        env = np.exp(-t * 9)
        return ((wave * 0.93 + noise) * env * 0.85).astype(np.float32)
    elif row >= 8:        # Snare / clap
        dur = 0.22; n = int(SR * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        tone = np.sin(2*np.pi * (160 + (11-row)*25) * t).astype(np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        env = np.exp(-t * 18)
        return ((tone * 0.25 + noise * 0.75) * env * 0.70).astype(np.float32)
    elif row >= 5:        # Hihat
        dur = 0.05 + (8 - row) * 0.04; n = int(SR * dur)
        noise = rng.standard_normal(n).astype(np.float32)
        noise[1::2] *= -1   # crude highpass
        env = np.exp(-np.linspace(0, 10, n))
        return (noise * env * 0.45).astype(np.float32)
    else:                 # Cymbal / perc
        dur = 0.30 + row * 0.01; n = int(SR * dur)
        noise = rng.standard_normal(n).astype(np.float32)
        noise[1::2] *= -1
        env = np.exp(-np.linspace(0, 7, n))
        return (noise * env * 0.38).astype(np.float32)


def synth_cat(freq):
    """Formant-ish meow pitched to freq."""
    dur, n = 0.55, int(SR * 0.55)
    t = np.linspace(0, dur, n, endpoint=False)
    glide = freq * (1.5 - 0.5 * t / dur)
    ph = np.cumsum(glide / SR) * 2 * np.pi
    wave = np.sin(ph) + 0.30 * np.sin(2*ph) + 0.10 * np.sin(3*ph)
    env = np.sin(np.pi * t / dur) ** 0.6
    return (wave * env * 0.45).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Instrument registry
# ══════════════════════════════════════════════════════════════════════════════

# 16 pitches — A minor pentatonic, A2→A5 (row 0=top=highest)
_PITCHES_ASC = [110.00, 130.81, 146.83, 164.81, 196.00, 220.00,
                261.63, 293.66, 329.63, 392.00, 440.00, 523.25,
                587.33, 659.25, 783.99, 880.00]
PITCHES = list(reversed(_PITCHES_ASC))   # index 0 = highest
PITCH_NAMES = ["A5","G5","E5","D5","C5","A4","G4","E4",
               "D4","C4","A3","G3","E3","D3","C3","A2"]

INSTRUMENTS = [
    # name           color              pitched_fn          drum_mode
    ("Glass Bell",  (0,   220, 255),   synth_glass_bell,   False),
    ("C64 Lead",    (100, 140, 255),   synth_c64_lead,     False),
    ("C64 Bass",    (60,  80,  220),   synth_c64_bass,     False),
    ("C64 Arp",     (160, 100, 255),   synth_c64_arp,      False),
    ("C64 Bell",    (180, 220, 255),   synth_c64_bell,     False),
    ("C64 Drums",   (140, 140, 200),   None,               True ),
    ("Didgeridoo",  (80,  180, 80 ),   synth_didgeridoo,   False),
    ("Rhodes",      (255, 165, 55 ),   synth_rhodes,       False),
    ("Kalimba",     (255, 220, 60 ),   synth_kalimba,      False),
    ("Cat Meow",    (255, 100, 165),   synth_cat,          False),
]
N_INST = len(INSTRUMENTS)
COLS = ROWS = 16


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-generate all sound buffers  (runs in background thread)
# ══════════════════════════════════════════════════════════════════════════════

BUFFERS: dict[tuple[int,int], np.ndarray] = {}   # (inst_idx, row) → mono float32
_buffers_ready = threading.Event()


def _prebuild():
    for ii, (name, color, fn, is_drum) in enumerate(INSTRUMENTS):
        for row in range(ROWS):
            if is_drum:
                buf = synth_drum(row)
            else:
                buf = fn(PITCHES[row])
            BUFFERS[(ii, row)] = buf
    _buffers_ready.set()


# Started inside main() after audio is confirmed working

# Fallback beep while buffers are building
def _beep(freq):
    n = int(SR * 0.10)
    t = np.linspace(0, 0.10, n, endpoint=False)
    buf = (np.sin(2*np.pi*freq*t) * np.exp(-t*20)).astype(np.float32)
    voice_play(buf, vol=0.6)


# ══════════════════════════════════════════════════════════════════════════════
#  Sequencer clock
# ══════════════════════════════════════════════════════════════════════════════

class Sequencer:
    def __init__(self):
        self.bpm        = 120
        self.step       = 0
        self.playing    = False
        self.step_time  = time.perf_counter()
        self.step_dur   = self._calc_dur()
        self._lock      = threading.Lock()
        self._thread    = None
        # grid[row][col] = inst_idx or None
        self.grid: list[list[int|None]] = [[None]*COLS for _ in range(ROWS)]
        self.flash: dict[tuple[int,int], float] = {}   # (r,c) → 0..1

    def _calc_dur(self):
        return 60.0 / self.bpm / 4   # 16th notes

    def toggle_play(self):
        if self.playing:
            self.playing = False
        else:
            self.playing = True
            self.step_dur = self._calc_dur()
            self._thread  = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        while self.playing:
            t0 = time.perf_counter()
            with self._lock:
                s = self.step
                self.step_time = t0
            # Fire notes
            for row in range(ROWS):
                ii = self.grid[row][s]
                if ii is not None:
                    key = (row, s)
                    self.flash[key] = 1.0
                    if _buffers_ready.is_set():
                        voice_play(BUFFERS[(ii, row)])
                    else:
                        _beep(PITCHES[row])
            with self._lock:
                self.step = (self.step + 1) % COLS
                self.step_dur = self._calc_dur()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, self.step_dur - elapsed))

    def set_bpm(self, bpm):
        self.bpm = max(40, min(300, bpm))


# ══════════════════════════════════════════════════════════════════════════════
#  Visual constants
# ══════════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H = 1440, 900
CTRL_H       = 72          # bottom control bar height

CELL  = 50                 # grid cell size
DOT_R = 14                 # dot radius
GLOW1 = 22                 # outer glow radius
GLOW2 = 18                 # inner glow radius

# Grid position (centered)
GRID_W = COLS * CELL       # 800
GRID_H = ROWS * CELL       # 800
GX = (WIN_W - GRID_W) // 2  # 320
GY = (WIN_H - CTRL_H - GRID_H) // 2  # 14

BG   = (10,  10,  16)
GRID = (28,  28,  40)
OFF  = (38,  38,  54)
TEXT = (190, 190, 210)
DIM  = (80,  80,  100)


# ══════════════════════════════════════════════════════════════════════════════
#  Glow cache — pre-render glow surfaces per instrument color
# ══════════════════════════════════════════════════════════════════════════════

def make_glow(color, r1, r2, r3):
    """Return a Surface with concentric glowing circles."""
    sz = r1 * 2 + 2
    s = pygame.Surface((sz, sz), pygame.SRCALPHA)
    cx = cy = sz // 2
    pygame.draw.circle(s, (*color, 25),  (cx, cy), r1)
    pygame.draw.circle(s, (*color, 55),  (cx, cy), r2)
    pygame.draw.circle(s, (*color, 160), (cx, cy), r3)
    pygame.draw.circle(s, (*color, 255), (cx, cy), r3 - 2)
    return s


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("Brutal Matrix Sequencer")
    clock  = pygame.time.Clock()

    # Start audio and sound prebuild now that the event loop is ready
    start_audio()
    threading.Thread(target=_prebuild, daemon=True).start()

    font_big = pygame.font.SysFont("Helvetica Neue,Arial", 16, bold=True)
    font_sm  = pygame.font.SysFont("Helvetica Neue,Arial", 12)

    seq          = Sequencer()
    cur_inst     = 0          # selected instrument index
    paint_mode   = None       # True=paint, False=erase, None=idle
    paint_inst   = None       # instrument used for current drag
    bpm_drag     = None       # (start_mx, start_bpm)
    hover_inst   = None       # instrument palette hover

    # Build glow surfaces after display init
    glow_cache: dict[int, pygame.Surface] = {}
    flash_cache: dict[int, pygame.Surface] = {}

    def get_glow(ii):
        if ii not in glow_cache:
            c = INSTRUMENTS[ii][1]
            glow_cache[ii]  = make_glow(c, GLOW1, GLOW2, DOT_R)
            flash_cache[ii] = make_glow(c, GLOW1 + 8, GLOW2 + 6, DOT_R + 4)
        return glow_cache[ii], flash_cache[ii]

    # ── Helper: cell under mouse ──────────────────────────────────────────────
    def cell_at(mx, my):
        col = (mx - GX) // CELL
        row = (my - GY) // CELL
        if 0 <= row < ROWS and 0 <= col < COLS:
            return row, col
        return None

    # ── Instrument palette layout ─────────────────────────────────────────────
    pal_y      = WIN_H - CTRL_H // 2
    pal_r      = 14         # palette dot radius
    pal_gap    = 44         # spacing between palette dots
    pal_total  = N_INST * pal_gap
    pal_x0     = (WIN_W - pal_total) // 2 + pal_gap // 2

    def inst_palette_rects():
        return [(pal_x0 + i * pal_gap, pal_y) for i in range(N_INST)]

    # ── BPM rect ──────────────────────────────────────────────────────────────
    bpm_r = pygame.Rect(WIN_W - 200, WIN_H - CTRL_H + 12, 130, 36)

    # ── Play/Stop rect ────────────────────────────────────────────────────────
    play_r = pygame.Rect(WIN_W - 60, WIN_H - CTRL_H + 12, 48, 36)

    running = True
    prev_t  = time.perf_counter()

    while running:
        now = time.perf_counter()
        dt  = now - prev_t
        prev_t = now
        W, H = screen.get_size()

        # ── Update flash timers ───────────────────────────────────────────────
        done = [k for k, v in seq.flash.items() if v <= 0]
        for k in done:
            del seq.flash[k]
        for k in seq.flash:
            seq.flash[k] = max(0.0, seq.flash[k] - dt * 5.5)

        # ── Draw background ───────────────────────────────────────────────────
        screen.fill(BG)

        # Subtle grid lines
        for r in range(ROWS + 1):
            pygame.draw.line(screen, GRID,
                             (GX, GY + r * CELL), (GX + GRID_W, GY + r * CELL))
        for c in range(COLS + 1):
            pygame.draw.line(screen, GRID,
                             (GX + c * CELL, GY), (GX + c * CELL, GY + GRID_H))

        # ── Playhead ──────────────────────────────────────────────────────────
        with seq._lock:
            cur_step = seq.step
            step_t   = seq.step_time
            step_dur = seq.step_dur

        # Smooth playhead interpolation
        if seq.playing:
            elapsed = now - step_t
            frac    = min(1.0, elapsed / step_dur) if step_dur > 0 else 1.0
            prev_col = (cur_step - 1) % COLS
            ph_x  = GX + prev_col * CELL + frac * CELL
        else:
            ph_x = GX + cur_step * CELL

        ph_surf = pygame.Surface((CELL, GRID_H), pygame.SRCALPHA)
        ph_surf.fill((255, 255, 255, 16))
        screen.blit(ph_surf, (ph_x, GY))
        # Bright leading edge
        pygame.draw.line(screen, (255, 255, 255, 100),
                         (int(ph_x + CELL), GY), (int(ph_x + CELL), GY + GRID_H), 2)

        # ── Grid dots ─────────────────────────────────────────────────────────
        for row in range(ROWS):
            for col in range(COLS):
                cx = GX + col * CELL + CELL // 2
                cy = GY + row * CELL + CELL // 2
                ii = seq.grid[row][col]
                if ii is None:
                    pygame.draw.circle(screen, OFF, (cx, cy), DOT_R - 4)
                else:
                    f = seq.flash.get((row, col), 0.0)
                    glow_s, flash_s = get_glow(ii)
                    if f > 0.0:
                        fs = pygame.transform.scale(
                            flash_s,
                            (int(flash_s.get_width() * (1 + f * 0.4)),
                             int(flash_s.get_height() * (1 + f * 0.4)))
                        )
                        screen.blit(fs, fs.get_rect(center=(cx, cy)),
                                    special_flags=pygame.BLEND_RGBA_ADD)
                    else:
                        screen.blit(glow_s, glow_s.get_rect(center=(cx, cy)),
                                    special_flags=pygame.BLEND_RGBA_ADD)

        # ── Bottom control bar ────────────────────────────────────────────────
        ctrl_y = H - CTRL_H
        pygame.draw.rect(screen, (14, 14, 22), pygame.Rect(0, ctrl_y, W, CTRL_H))
        pygame.draw.line(screen, GRID, (0, ctrl_y), (W, ctrl_y))

        # Instrument palette
        palette_centers = inst_palette_rects()
        for i, (px, py) in enumerate(palette_centers):
            py = ctrl_y + CTRL_H // 2
            c  = INSTRUMENTS[i][1]
            name = INSTRUMENTS[i][0]
            is_cur = (i == cur_inst)
            is_hov = (i == hover_inst)
            # Outer ring for selected
            if is_cur:
                pygame.draw.circle(screen, c, (px, py), pal_r + 5, 2)
            elif is_hov:
                pygame.draw.circle(screen, (*c, 120), (px, py), pal_r + 3, 1)
            pygame.draw.circle(screen, c, (px, py), pal_r)
            if is_cur or is_hov:
                img = font_sm.render(name, True, c)
                screen.blit(img, img.get_rect(centerx=px, top=ctrl_y + 4))

        # BPM — drag left/right to change
        pygame.draw.rect(screen, (22, 22, 34), bpm_r, border_radius=6)
        pygame.draw.rect(screen, GRID, bpm_r, 1, border_radius=6)
        b_img = font_big.render(f"♩ {seq.bpm} BPM", True, TEXT)
        screen.blit(b_img, b_img.get_rect(center=bpm_r.center))

        # Play/Stop button
        pcol = (40, 160, 70) if seq.playing else (40, 90, 180)
        pygame.draw.rect(screen, pcol, play_r, border_radius=6)
        p_img = font_big.render("■" if seq.playing else "▶", True, (255, 255, 255))
        screen.blit(p_img, p_img.get_rect(center=play_r.center))

        # Keyboard hint
        hint = font_sm.render("SPACE play/stop  ·  R reset  ·  drag BPM  ·  right-click erase",
                               True, DIM)
        screen.blit(hint, (12, ctrl_y + CTRL_H - 18))

        # Buffer status
        if not _buffers_ready.is_set():
            st = font_sm.render("● loading sounds…", True, (255, 180, 0))
            screen.blit(st, (12, ctrl_y + 8))

        # ── Events ────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    seq.toggle_play()
                elif ev.key == pygame.K_r:
                    with seq._lock:
                        seq.step = 0
                elif ev.key == pygame.K_ESCAPE:
                    running = False

            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                # Hover on palette
                hover_inst = None
                for i, (px, py) in enumerate(palette_centers):
                    py = ctrl_y + CTRL_H // 2
                    if math.hypot(mx - px, my - py) <= pal_r + 6:
                        hover_inst = i

                # Drag BPM
                if bpm_drag:
                    dx = mx - bpm_drag[0]
                    seq.set_bpm(bpm_drag[1] + dx // 2)

                # Paint/erase drag
                if paint_mode is not None:
                    c = cell_at(mx, my)
                    if c:
                        row, col = c
                        if paint_mode:
                            seq.grid[row][col] = paint_inst
                        else:
                            seq.grid[row][col] = None

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos

                if play_r.collidepoint(mx, my):
                    seq.toggle_play()

                elif bpm_r.collidepoint(mx, my):
                    bpm_drag = (mx, seq.bpm)

                else:
                    # Palette click
                    clicked_pal = False
                    for i, (px, py) in enumerate(palette_centers):
                        py = ctrl_y + CTRL_H // 2
                        if math.hypot(mx - px, my - py) <= pal_r + 6:
                            cur_inst = i
                            clicked_pal = True
                            break

                    if not clicked_pal:
                        c = cell_at(mx, my)
                        if c:
                            row, col = c
                            if ev.button == 3:           # right-click = erase
                                seq.grid[row][col] = None
                                paint_mode = False
                                paint_inst = None
                            elif ev.button == 1:
                                if seq.grid[row][col] is None:
                                    seq.grid[row][col] = cur_inst
                                    paint_mode = True
                                    paint_inst = cur_inst
                                else:
                                    seq.grid[row][col] = None
                                    paint_mode = False
                                    paint_inst = None

            elif ev.type == pygame.MOUSEBUTTONUP:
                paint_mode = None
                paint_inst = None
                bpm_drag   = None

            elif ev.type == pygame.MOUSEWHEEL:
                # Scroll on BPM area to change BPM
                mx, my = pygame.mouse.get_pos()
                if bpm_r.collidepoint(mx, my):
                    seq.set_bpm(seq.bpm + ev.y * 2)

        pygame.display.flip()
        clock.tick(60)

    seq.playing = False
    if _stream:
        _stream.stop()
    pygame.quit()


if __name__ == "__main__":
    main()

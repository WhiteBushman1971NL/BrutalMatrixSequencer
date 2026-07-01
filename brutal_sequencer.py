#!/usr/bin/env python3
"""
Brutal Matrix Sequencer  ·  16×16 grid  ·  one playhead  ·  multi-instrument
Paint notes with different instruments. Hit SPACE to play.
"""

import math, threading, time, signal, sys
import numpy as np
import pygame
from scipy.signal import lfilter

# Let Ctrl-C kill the process even with background audio threads
signal.signal(signal.SIGINT, signal.SIG_DFL)

# ══════════════════════════════════════════════════════════════════════════════
#  Audio engine — pygame.mixer (no external DLL, works on Windows/Mac/Linux)
# ══════════════════════════════════════════════════════════════════════════════
SR         = 44100
MAX_VOICES = 64
SOUNDS: dict = {}   # (inst_idx, row) → pygame.mixer.Sound


def start_audio():
    try:
        pygame.mixer.pre_init(SR, -16, 2, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(MAX_VOICES)
        return True
    except Exception as e:
        print(f"Audio init failed: {e}", file=sys.stderr)
        return False


def _to_sound(mono: np.ndarray) -> pygame.mixer.Sound:
    """Convert float32 mono buffer → stereo int16 pygame Sound."""
    int16  = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([int16, int16]))
    return pygame.sndarray.make_sound(stereo)


def voice_play(sound: pygame.mixer.Sound, vol: float = 0.80):
    ch = pygame.mixer.find_channel(True)   # True = steal oldest if all busy
    if ch:
        ch.set_volume(vol)
        ch.play(sound)


# ══════════════════════════════════════════════════════════════════════════════
#  Synthesis — one function per instrument  (all return float32 mono arrays)
# ══════════════════════════════════════════════════════════════════════════════

def _ar(n, attack, release):
    e = np.ones(n, dtype=np.float32)
    a = min(int(SR * attack), n)
    r = min(int(SR * release), n)
    if a: e[:a]  = np.linspace(0, 1, a)
    if r: e[-r:] = np.linspace(1, 0, r)
    return e

def _lp(sig, cutoff):
    a1 = math.exp(-2 * math.pi * cutoff / SR)
    return lfilter([1 - a1], [1, -a1], sig).astype(np.float32)


def synth_glass_bell(freq):
    dur, n = 2.0, int(SR * 2.0)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (0.55 * np.sin(2*np.pi*freq*t)
          + 0.22 * np.sin(2*np.pi*freq*2.76*t)
          + 0.10 * np.sin(2*np.pi*freq*5.40*t)
          + 0.07 * np.sin(2*np.pi*freq*8.93*t))
    return (wave * np.exp(-t * 2.4) * _ar(n, 0.003, 0) * 0.65).astype(np.float32)


def synth_c64_lead(freq):
    dur, n = 0.28, int(SR * 0.28)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = np.where((t * freq % 1.0) < 0.25, 1.0, -1.0).astype(np.float32)
    return (_lp(wave, 3200) * _ar(n, 0.005, 0.05) * 0.45).astype(np.float32)


def synth_c64_bass(freq):
    freq = max(freq * 0.5, 55.0)
    dur, n = 0.40, int(SR * 0.40)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (2 * ((t * freq) % 1.0) - 1.0).astype(np.float32)
    return (_lp(wave, 700) * _ar(n, 0.008, 0.07) * 0.55).astype(np.float32)


def synth_c64_arp(freq):
    dur, n = 0.12, int(SR * 0.12)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = np.where((t * freq % 1.0) < 0.5, 1.0, -1.0).astype(np.float32)
    return (_lp(wave, 4000) * _ar(n, 0.003, 0.03) * 0.40).astype(np.float32)


def synth_c64_bell(freq):
    dur, n = 1.2, int(SR * 1.2)
    t = np.linspace(0, dur, n, endpoint=False)
    tri = 2 * np.abs(2 * ((t * freq) % 1.0) - 1.0) - 1.0
    return (tri * np.exp(-t * 3.0) * _ar(n, 0.002, 0) * 0.50).astype(np.float32)


def synth_rhodes(freq):
    dur, n = 1.1, int(SR * 1.1)
    t = np.linspace(0, dur, n, endpoint=False)
    mi  = np.exp(-t * 3.5) * 5.0
    mod = np.sin(2*np.pi * freq * t)
    car = np.sin(2*np.pi * freq * t + mi * mod)
    return (car * np.exp(-t * 2.0) * _ar(n, 0.004, 0) * 0.55).astype(np.float32)


def synth_kalimba(freq):
    delay = max(2, int(round(SR / freq)))
    n     = int(SR * 1.6)
    rng   = np.random.default_rng(int(freq * 10))
    burst = np.zeros(n, dtype=np.float32)
    burst[:delay] = rng.standard_normal(delay).astype(np.float32)
    a = np.zeros(delay + 2, dtype=np.float64)
    a[0] = 1.0;  a[delay] = -0.4985;  a[delay + 1] = -0.4985
    wave = lfilter([1.0], a, burst).astype(np.float32)
    return (wave * np.exp(-np.linspace(0, 3.5, n)) * 0.80).astype(np.float32)


def synth_didgeridoo(freq):
    freq = min(freq, 220.0)
    dur, n = 0.65, int(SR * 0.65)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (0.50 * np.sin(2*np.pi*freq*t)
          + 0.28 * np.sin(2*np.pi*freq*2*t)
          + 0.14 * np.sin(2*np.pi*freq*3*t)
          + 0.06 * np.sin(2*np.pi*freq*5*t))
    return (wave * (0.75 + 0.25*np.sin(2*np.pi*8*t)) * _ar(n, 0.025, 0.12) * 0.52).astype(np.float32)


def synth_drum(row):
    rng = np.random.default_rng(row * 7 + 13)
    if row >= 12:
        dur = 0.50;  n = int(SR * dur)
        t   = np.linspace(0, dur, n, endpoint=False)
        pe  = (80 + (15 - row) * 20) * np.exp(-t * 20)
        w   = np.sin(2*np.pi * np.cumsum(pe / SR))
        return ((w * 0.93 + rng.standard_normal(n).astype(np.float32) * 0.07)
                * np.exp(-t * 9) * 0.85).astype(np.float32)
    elif row >= 8:
        dur = 0.22;  n = int(SR * dur)
        t   = np.linspace(0, dur, n, endpoint=False)
        tone = np.sin(2*np.pi * (160 + (11-row)*25) * t).astype(np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        return ((tone * 0.25 + noise * 0.75) * np.exp(-t * 18) * 0.70).astype(np.float32)
    elif row >= 5:
        dur = 0.05 + (8 - row) * 0.04;  n = int(SR * dur)
        noise = rng.standard_normal(n).astype(np.float32)
        noise[1::2] *= -1
        return (noise * np.exp(-np.linspace(0, 10, n)) * 0.45).astype(np.float32)
    else:
        dur = 0.30 + row * 0.01;  n = int(SR * dur)
        noise = rng.standard_normal(n).astype(np.float32)
        noise[1::2] *= -1
        return (noise * np.exp(-np.linspace(0, 7, n)) * 0.38).astype(np.float32)


def synth_cat(freq):
    dur, n = 0.55, int(SR * 0.55)
    t = np.linspace(0, dur, n, endpoint=False)
    glide = freq * (1.5 - 0.5 * t / dur)
    ph = np.cumsum(glide / SR) * 2 * np.pi
    wave = np.sin(ph) + 0.30*np.sin(2*ph) + 0.10*np.sin(3*ph)
    return (wave * np.sin(np.pi * t / dur)**0.6 * 0.45).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Instrument registry
# ══════════════════════════════════════════════════════════════════════════════

_PITCHES_ASC = [110.00, 130.81, 146.83, 164.81, 196.00, 220.00,
                261.63, 293.66, 329.63, 392.00, 440.00, 523.25,
                587.33, 659.25, 783.99, 880.00]
PITCHES = list(reversed(_PITCHES_ASC))   # index 0 = highest pitch (top row)

INSTRUMENTS = [
    # name           color              synth_fn           drum_mode
    ("Glass Bell",  (0,   220, 255),   synth_glass_bell,  False),
    ("C64 Lead",    (100, 140, 255),   synth_c64_lead,    False),
    ("C64 Bass",    (60,  80,  220),   synth_c64_bass,    False),
    ("C64 Arp",     (160, 100, 255),   synth_c64_arp,     False),
    ("C64 Bell",    (180, 220, 255),   synth_c64_bell,    False),
    ("C64 Drums",   (140, 140, 200),   None,              True ),
    ("Didgeridoo",  (80,  180, 80 ),   synth_didgeridoo,  False),
    ("Rhodes",      (255, 165, 55 ),   synth_rhodes,      False),
    ("Kalimba",     (255, 220, 60 ),   synth_kalimba,     False),
    ("Cat Meow",    (255, 100, 165),   synth_cat,         False),
]
N_INST = len(INSTRUMENTS)
COLS = ROWS = 16


def build_sounds(screen, font):
    """Generate and convert all instrument sounds. Shows progress on screen."""
    total = N_INST * ROWS
    done  = 0
    W, H  = screen.get_size()
    for ii, (name, color, fn, is_drum) in enumerate(INSTRUMENTS):
        for row in range(ROWS):
            buf = synth_drum(row) if is_drum else fn(PITCHES[row])
            SOUNDS[(ii, row)] = _to_sound(buf)
            done += 1
            # Draw progress bar
            screen.fill((10, 10, 16))
            label = font.render(f"Loading  {name}…", True, color)
            screen.blit(label, label.get_rect(center=(W//2, H//2 - 20)))
            bw = int(W * 0.4)
            bx = (W - bw) // 2
            pygame.draw.rect(screen, (30, 30, 45), pygame.Rect(bx, H//2 + 10, bw, 12), border_radius=6)
            pygame.draw.rect(screen, color,        pygame.Rect(bx, H//2 + 10, int(bw * done/total), 12), border_radius=6)
            pygame.display.flip()
            pygame.event.pump()   # keep window responsive


# ══════════════════════════════════════════════════════════════════════════════
#  Sequencer clock
# ══════════════════════════════════════════════════════════════════════════════

class Sequencer:
    def __init__(self):
        self.bpm       = 120
        self.step      = 0
        self.playing   = False
        self.step_time = time.perf_counter()
        self.step_dur  = self._calc_dur()
        self._lock     = threading.Lock()
        self._thread   = None
        self.grid: list = [[None]*COLS for _ in range(ROWS)]
        self.flash: dict = {}

    def _calc_dur(self):
        return 60.0 / self.bpm / 4

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
            for row in range(ROWS):
                ii = self.grid[row][s]
                if ii is not None:
                    self.flash[(row, s)] = 1.0
                    snd = SOUNDS.get((ii, row))
                    if snd:
                        voice_play(snd)
            with self._lock:
                self.step     = (self.step + 1) % COLS
                self.step_dur = self._calc_dur()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, self.step_dur - elapsed))

    def set_bpm(self, bpm):
        self.bpm = max(40, min(300, bpm))


# ══════════════════════════════════════════════════════════════════════════════
#  Visual constants
# ══════════════════════════════════════════════════════════════════════════════

WIN_W, WIN_H = 1440, 900
CTRL_H = 72
CELL   = 50
DOT_R  = 14
GLOW1, GLOW2 = 22, 18

GRID_W = COLS * CELL
GRID_H = ROWS * CELL
GX = (WIN_W - GRID_W) // 2
GY = (WIN_H - CTRL_H - GRID_H) // 2

BG   = (10,  10,  16)
GRID = (28,  28,  40)
OFF  = (38,  38,  54)
TEXT = (190, 190, 210)
DIM  = (80,  80,  100)


def make_glow(color, r1, r2, r3):
    sz = r1 * 2 + 2
    s  = pygame.Surface((sz, sz), pygame.SRCALPHA)
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
    start_audio()

    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("Brutal Matrix Sequencer")
    clock  = pygame.time.Clock()

    font_big = pygame.font.SysFont("Helvetica Neue,Arial", 16, bold=True)
    font_sm  = pygame.font.SysFont("Helvetica Neue,Arial", 12)

    # Build all sounds (shows progress bar, ~0.5s on M4 Max)
    build_sounds(screen, font_big)

    seq        = Sequencer()
    cur_inst   = 0
    paint_mode = None
    paint_inst = None
    bpm_drag   = None
    hover_inst = None

    glow_cache  = {}
    flash_cache = {}

    def get_glow(ii):
        if ii not in glow_cache:
            c = INSTRUMENTS[ii][1]
            glow_cache[ii]  = make_glow(c, GLOW1, GLOW2, DOT_R)
            flash_cache[ii] = make_glow(c, GLOW1 + 8, GLOW2 + 6, DOT_R + 4)
        return glow_cache[ii], flash_cache[ii]

    def cell_at(mx, my):
        col = (mx - GX) // CELL
        row = (my - GY) // CELL
        if 0 <= row < ROWS and 0 <= col < COLS:
            return row, col
        return None

    pal_r   = 14
    pal_gap = 44
    pal_x0  = (WIN_W - N_INST * pal_gap) // 2 + pal_gap // 2

    def palette_centers(ctrl_y):
        return [(pal_x0 + i * pal_gap, ctrl_y + CTRL_H // 2) for i in range(N_INST)]

    bpm_r  = pygame.Rect(WIN_W - 200, WIN_H - CTRL_H + 12, 130, 36)
    play_r = pygame.Rect(WIN_W - 60,  WIN_H - CTRL_H + 12, 48,  36)

    running = True
    prev_t  = time.perf_counter()

    while running:
        now    = time.perf_counter()
        dt     = now - prev_t
        prev_t = now
        W, H   = screen.get_size()
        ctrl_y = H - CTRL_H

        # Flash decay
        for k in list(seq.flash):
            seq.flash[k] = max(0.0, seq.flash[k] - dt * 5.5)
            if seq.flash[k] <= 0:
                del seq.flash[k]

        screen.fill(BG)

        # Grid lines
        for r in range(ROWS + 1):
            pygame.draw.line(screen, GRID, (GX, GY + r*CELL), (GX + GRID_W, GY + r*CELL))
        for c in range(COLS + 1):
            pygame.draw.line(screen, GRID, (GX + c*CELL, GY), (GX + c*CELL, GY + GRID_H))

        # Playhead
        with seq._lock:
            cur_step = seq.step
            step_t   = seq.step_time
            step_dur = seq.step_dur

        if seq.playing:
            frac    = min(1.0, (now - step_t) / step_dur) if step_dur > 0 else 1.0
            ph_x    = GX + ((cur_step - 1) % COLS) * CELL + frac * CELL
        else:
            ph_x = GX + cur_step * CELL

        ph = pygame.Surface((CELL, GRID_H), pygame.SRCALPHA)
        ph.fill((255, 255, 255, 16))
        screen.blit(ph, (ph_x, GY))
        pygame.draw.line(screen, (255, 255, 255),
                         (int(ph_x + CELL), GY), (int(ph_x + CELL), GY + GRID_H), 2)

        # Dots
        for row in range(ROWS):
            for col in range(COLS):
                cx = GX + col*CELL + CELL//2
                cy = GY + row*CELL + CELL//2
                ii = seq.grid[row][col]
                if ii is None:
                    pygame.draw.circle(screen, OFF, (cx, cy), DOT_R - 4)
                else:
                    f = seq.flash.get((row, col), 0.0)
                    gs, fs = get_glow(ii)
                    if f > 0.0:
                        scaled = pygame.transform.scale(
                            fs, (int(fs.get_width()*(1+f*0.4)), int(fs.get_height()*(1+f*0.4))))
                        screen.blit(scaled, scaled.get_rect(center=(cx, cy)),
                                    special_flags=pygame.BLEND_RGBA_ADD)
                    else:
                        screen.blit(gs, gs.get_rect(center=(cx, cy)),
                                    special_flags=pygame.BLEND_RGBA_ADD)

        # Control bar
        pygame.draw.rect(screen, (14, 14, 22), pygame.Rect(0, ctrl_y, W, CTRL_H))
        pygame.draw.line(screen, GRID, (0, ctrl_y), (W, ctrl_y))

        pcs = palette_centers(ctrl_y)
        for i, (px, py) in enumerate(pcs):
            c    = INSTRUMENTS[i][1]
            name = INSTRUMENTS[i][0]
            if i == cur_inst:
                pygame.draw.circle(screen, c, (px, py), pal_r + 5, 2)
            elif i == hover_inst:
                pygame.draw.circle(screen, (*c, 120), (px, py), pal_r + 3, 1)
            pygame.draw.circle(screen, c, (px, py), pal_r)
            if i == cur_inst or i == hover_inst:
                img = font_sm.render(name, True, c)
                screen.blit(img, img.get_rect(centerx=px, top=ctrl_y + 4))

        pygame.draw.rect(screen, (22, 22, 34), bpm_r, border_radius=6)
        pygame.draw.rect(screen, GRID, bpm_r, 1, border_radius=6)
        screen.blit(font_big.render(f"♩ {seq.bpm} BPM", True, TEXT),
                    font_big.render(f"♩ {seq.bpm} BPM", True, TEXT).get_rect(center=bpm_r.center))

        pcol = (40, 160, 70) if seq.playing else (40, 90, 180)
        pygame.draw.rect(screen, pcol, play_r, border_radius=6)
        screen.blit(font_big.render("■" if seq.playing else "▶", True, (255,255,255)),
                    font_big.render("■" if seq.playing else "▶", True, (255,255,255)).get_rect(center=play_r.center))

        screen.blit(font_sm.render("SPACE play/stop  ·  R reset  ·  drag BPM  ·  right-click erase", True, DIM),
                    (12, ctrl_y + CTRL_H - 18))

        # Events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if   ev.key == pygame.K_SPACE:  seq.toggle_play()
                elif ev.key == pygame.K_r:
                    with seq._lock: seq.step = 0
                elif ev.key == pygame.K_ESCAPE: running = False

            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                hover_inst = None
                for i, (px, py) in enumerate(pcs):
                    if math.hypot(mx-px, my-py) <= pal_r + 6:
                        hover_inst = i
                if bpm_drag:
                    seq.set_bpm(bpm_drag[1] + (mx - bpm_drag[0]) // 2)
                if paint_mode is not None:
                    c = cell_at(mx, my)
                    if c:
                        row, col = c
                        seq.grid[row][col] = paint_inst if paint_mode else None

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                if play_r.collidepoint(mx, my):
                    seq.toggle_play()
                elif bpm_r.collidepoint(mx, my):
                    bpm_drag = (mx, seq.bpm)
                else:
                    hit_pal = False
                    for i, (px, py) in enumerate(pcs):
                        if math.hypot(mx-px, my-py) <= pal_r + 6:
                            cur_inst = i;  hit_pal = True;  break
                    if not hit_pal:
                        c = cell_at(mx, my)
                        if c:
                            row, col = c
                            if ev.button == 3:
                                seq.grid[row][col] = None
                                paint_mode = False;  paint_inst = None
                            elif ev.button == 1:
                                if seq.grid[row][col] is None:
                                    seq.grid[row][col] = cur_inst
                                    paint_mode = True;  paint_inst = cur_inst
                                else:
                                    seq.grid[row][col] = None
                                    paint_mode = False;  paint_inst = None

            elif ev.type == pygame.MOUSEBUTTONUP:
                paint_mode = None;  paint_inst = None;  bpm_drag = None

            elif ev.type == pygame.MOUSEWHEEL:
                if bpm_r.collidepoint(*pygame.mouse.get_pos()):
                    seq.set_bpm(seq.bpm + ev.y * 2)

        pygame.display.flip()
        clock.tick(60)

    seq.playing = False
    pygame.quit()


if __name__ == "__main__":
    main()

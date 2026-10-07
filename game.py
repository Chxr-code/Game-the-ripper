import pygame
import sys
import math
import random
import os

try:
    from PIL import Image as PILImage
    PILLOW_OK = True
except ImportError:
    PILLOW_OK = False
    print("[warn] Pillow not found. (pip install Pillow)")

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False
    print("[warn] numpy not found. (pip install numpy)")

pygame.init()
pygame.mixer.init()


#  Window / World

WORLD_WIDTH = 3200
WIDTH       = 800
HEIGHT      = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Soul Collector")
clock = pygame.time.Clock()

font_big = pygame.font.SysFont(None, 64)
font_mid = pygame.font.SysFont(None, 36)
font_sm  = pygame.font.SysFont(None, 26)

# ══════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════
FLOOR_Y       = 500
CEILING_Y     = 40
TARGET_CHAR_H = 70
NPC_H         = 65
SOUL_H        = 80
TIMER_SECONDS = 60

REAPER_SPEED = 4
REAPER_MIN_Y = float(CEILING_Y)
REAPER_MAX_Y = float(FLOOR_Y - TARGET_CHAR_H)

_soul_ratio     = random.uniform(0.60, 0.90)
SOUL_SPEED      = round(REAPER_SPEED * random.uniform(0.35, 0.55), 2)
SOUL_FLEE_SPEED = round(REAPER_SPEED * _soul_ratio, 2)
SOUL_DETECT     = 230
SOUL_MIN_Y      = 50.0
SOUL_MAX_Y      = float(FLOOR_Y - SOUL_H - 20)

NPC_ANIM_SPD = 8
GIF_ANIM_SPD = 6

# ══════════════════════════════════════════════════════════════
#  Asset loader
# ══════════════════════════════════════════════════════════════

def make_placeholder(w, h, color, label=""):
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill(color)
    if label:
        f = pygame.font.SysFont(None, 16)
        surf.blit(f.render(label, True, (255,255,255)), (2, h//2-8))
    return surf

def _try_path(path):
    """Try both original extension and upper/lower case variants."""
    if os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for e in (ext.lower(), ext.upper()):
        p = base + e
        if os.path.exists(p):
            return p
    return path

def _pil_to_pygame(pil_img):
    rgba = pil_img.convert("RGBA")
    return pygame.image.fromstring(rgba.tobytes(), rgba.size, "RGBA").convert_alpha()

def _crop_black_bg(pil_img, threshold=40):
    """
    Given a PIL RGBA image with a solid black background,
    detect non-black pixels as the character, crop it,
    and make black pixels transparent.
    Returns (cropped_pil_img, (left, top, right, bottom)) or (None, None).
    Works with numpy if available, falls back to pure PIL otherwise.
    """
    rgba = pil_img.convert("RGBA")

    if NUMPY_OK:
        arr  = np.array(rgba)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        char_mask = (r.astype(int) > threshold) | \
                    (g.astype(int) > threshold) | \
                    (b.astype(int) > threshold)
        rows = np.where(char_mask.any(axis=1))[0]
        cols = np.where(char_mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None, None
        top, bottom = int(rows[0]), int(rows[-1])
        left, right = int(cols[0]), int(cols[-1])
        bbox = (left, top, right+1, bottom+1)
        cropped = rgba.crop(bbox)
        arr_crop = np.array(cropped)
        rc,gc,bc = arr_crop[:,:,0], arr_crop[:,:,1], arr_crop[:,:,2]
        black = (rc.astype(int)<threshold) & (gc.astype(int)<threshold) & (bc.astype(int)<threshold)
        arr_crop[:,:,3][black] = 0
        return PILImage.fromarray(arr_crop, "RGBA"), bbox
    else:
        # Pure PIL fallback — use getbbox on a thresholded version
        # Build a mask image: white where non-black, black elsewhere
        r_ch, g_ch, b_ch, a_ch = rgba.split()
        # point() applies a per-pixel function via lookup table
        def above(v): return 255 if v > threshold else 0
        mask_r = r_ch.point(lambda v: 255 if v > threshold else 0)
        mask_g = g_ch.point(lambda v: 255 if v > threshold else 0)
        mask_b = b_ch.point(lambda v: 255 if v > threshold else 0)
        # combine: non-black if any channel > threshold
        from PIL import ImageChops
        combined = ImageChops.lighter(ImageChops.lighter(mask_r, mask_g), mask_b)
        bbox = combined.getbbox()   # (left, top, right, bottom)
        if not bbox:
            return None, None
        cropped = rgba.crop(bbox)
        # make black pixels transparent using PIL putdata
        pixels = list(cropped.getdata())
        new_pixels = []
        for (rv, gv, bv, av) in pixels:
            if rv < threshold and gv < threshold and bv < threshold:
                new_pixels.append((rv, gv, bv, 0))
            else:
                new_pixels.append((rv, gv, bv, av))
        cropped.putdata(new_pixels)
        return cropped, bbox

def _normalise_strip(frames_with_meta, target_h):
    """
    Use FRAME 0 as the size reference (standing pose).
    All frames are scaled so their HEIGHT matches frame 0's scaled height.
    This prevents lying-down frames (last frame of die gif) from
    distorting the scale of the whole strip.
    """
    if not frames_with_meta:
        return []

    # Find first valid frame to use as size reference
    ref_pil, ref_bbox = None, None
    for pil_img, bbox in frames_with_meta:
        if pil_img is not None and bbox is not None:
            ref_pil, ref_bbox = pil_img, bbox
            break
    if ref_pil is None:
        w = max(1, int(target_h * 0.55))
        return [make_placeholder(w, target_h, (80,80,200))]

    # Reference: scale frame 0 to target_h → get reference width
    ref_w, ref_h = ref_pil.size
    ref_scaled_w = max(1, int(target_h * ref_w / ref_h))
    # ref_scaled_w is the "standard" width all frames share

    result = []
    for pil_img, bbox in frames_with_meta:
        if pil_img is None or bbox is None:
            canvas = pygame.Surface((ref_scaled_w, target_h), pygame.SRCALPHA)
            result.append(canvas)
            continue

        surf = _pil_to_pygame(pil_img)
        # Scale so HEIGHT = target_h (preserving aspect — may be wider if lying down)
        fw, fh = surf.get_size()
        if fh == 0:
            result.append(pygame.Surface((ref_scaled_w, target_h), pygame.SRCALPHA))
            continue
        scaled_h = target_h
        scaled_w = max(1, int(target_h * fw / fh))

        scaled = pygame.transform.smoothscale(surf, (scaled_w, scaled_h))

        # Place on a canvas ref_scaled_w wide, bottom-anchored, centred
        canvas = pygame.Surface((ref_scaled_w, target_h), pygame.SRCALPHA)
        blit_x = (ref_scaled_w - scaled_w) // 2
        blit_y = target_h - scaled_h   # bottom-anchor
        canvas.blit(scaled, (blit_x, blit_y))
        result.append(canvas)

    return result

def load_png_strip_black(paths, target_h, ph_color=(80,80,200)):
    """
    Load PNGs that have a solid BLACK background.
    Removes background, detects exact character bounds,
    and normalises all frames to the same canvas anchored at bottom.
    """
    frames_meta = []
    for path in paths:
        try:
            p   = _try_path(path)
            pil = PILImage.open(p).convert("RGBA")
            cropped, bbox = _crop_black_bg(pil)
            frames_meta.append((cropped, bbox))
        except Exception:
            frames_meta.append((None, None))

    result = _normalise_strip(frames_meta, target_h)
    if not result:
        w = max(1, int(target_h * 0.55))
        return [make_placeholder(w, target_h, ph_color)]
    return result

def load_png(path, target_h, ph_color=(80,80,200), label=""):
    """Single transparent-BG PNG."""
    try:
        p   = _try_path(path)
        img = pygame.image.load(p).convert_alpha()
        rect = img.get_bounding_rect()
        crop = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        crop.blit(img, (0, 0), rect)
        ow, oh = crop.get_size()
        nw = max(1, int(target_h * ow / oh))
        return pygame.transform.smoothscale(crop, (nw, target_h))
    except Exception:
        w = max(1, int(target_h * 0.55))
        return make_placeholder(w, target_h, ph_color, label)

def load_gif_frames(path, target_h, ph_color=(80,80,200), label=""):
    """Load all GIF frames (transparent BG)."""
    try:
        p = _try_path(path)
        if PILLOW_OK:
            pil   = PILImage.open(p)
            surfs = []
            try:
                while True:
                    s = _pil_to_pygame(pil)
                    rect = s.get_bounding_rect()
                    if rect.width > 0 and rect.height > 0:
                        crop = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                        crop.blit(s, (0,0), rect)
                        ow, oh = crop.get_size()
                        nw = max(1, int(target_h * ow / oh))
                        surfs.append(pygame.transform.smoothscale(crop, (nw, target_h)))
                    pil.seek(pil.tell() + 1)
            except EOFError:
                pass
            if surfs:
                # normalise GIF canvas too
                max_w = max(s.get_width() for s in surfs)
                out = []
                for s in surfs:
                    c = pygame.Surface((max_w, target_h), pygame.SRCALPHA)
                    c.blit(s, ((max_w - s.get_width())//2, 0))
                    out.append(c)
                return out
        img = pygame.image.load(p).convert_alpha()
        ow, oh = img.get_size()
        nw = max(1, int(target_h * ow / oh))
        return [pygame.transform.smoothscale(img, (nw, target_h))]
    except Exception:
        w = max(1, int(target_h * 0.55))
        return [make_placeholder(w, target_h, ph_color, label)]

def load_gif_frames_black(path, target_h, ph_color=(80,80,200), label=""):
    """Load GIF frames that have a solid black background — same crop/normalise as PNG strip."""
    try:
        p = _try_path(path)
        if PILLOW_OK:
            pil = PILImage.open(p)
            frames_meta = []
            try:
                while True:
                    frame = pil.convert("RGBA")
                    cropped, bbox = _crop_black_bg(frame)
                    frames_meta.append((cropped, bbox))
                    pil.seek(pil.tell() + 1)
            except EOFError:
                pass
            if frames_meta:
                result = _normalise_strip(frames_meta, target_h)
                if result:
                    return result
        # fallback: load as-is
        img = pygame.image.load(p).convert_alpha()
        ow, oh = img.get_size()
        nw = max(1, int(target_h * ow / oh))
        return [pygame.transform.smoothscale(img, (nw, target_h))]
    except Exception:
        w = max(1, int(target_h * 0.55))
        return [make_placeholder(w, target_h, ph_color, label)]

def load_bg(path):
    try:
        img = pygame.image.load(_try_path(path)).convert()
        return pygame.transform.scale(img, (WORLD_WIDTH, HEIGHT))
    except Exception:
        surf = pygame.Surface((WORLD_WIDTH, HEIGHT))
        for y in range(HEIGHT):
            pygame.draw.line(surf,
                (int(8+12*y/HEIGHT), int(8+8*y/HEIGHT), int(28+22*y/HEIGHT)),
                (0,y),(WORLD_WIDTH,y))
        return surf

# ══════════════════════════════════════════════════════════════
#  Load sprites
# ══════════════════════════════════════════════════════════════
# โหลดเสียง — bg1 เล่นตอนปกติ, BGM_CHASE เล่นตอนจับเวลา
BGM_CHASE = None
try:
    BGM_CHASE = pygame.mixer.Sound(_try_path("sound/bg.wav"))
    BGM_CHASE.set_volume(0.1)
except Exception as e:
    print(f"[warn] Cannot load bg.wav: {e}")

SFX_WIN = None
try:
    SFX_WIN = pygame.mixer.Sound(_try_path("sound/win.wav"))
    SFX_WIN.set_volume(0.15)
except Exception as e:
    print(f"[warn] Cannot load win.wav: {e}")

SFX_LOSE = None
try:
    SFX_LOSE = pygame.mixer.Sound(_try_path("sound/lose.wav"))
    SFX_LOSE.set_volume(0.5)
except Exception as e:
    print(f"[warn] Cannot load lose.wav: {e}")

try:
    pygame.mixer.music.load(_try_path("sound/bgh.wav"))
    pygame.mixer.music.set_volume(0.1)
    pygame.mixer.music.play(-1, fade_ms=3000)
except Exception as e:
    print(f"[warn] Cannot load bg1.wav: {e}")

print("Loading sprites...")

REAPER_IDLE_SURF   = load_png("reaper_idle.png", TARGET_CHAR_H, (50,50,120), "reaper")
REAPER_COLLECT_GIF = load_gif_frames("collect.gif", TARGET_CHAR_H, (80,50,150), "collect")

# Load all NPC sprites together → all normalised to same canvas size
_npc_all = load_png_strip_black(["idle1.png", "walkl1.png", "walkr1.png"], NPC_H)
NPC_SPRITE_IDLE  = _npc_all[0]
NPC_SPRITE_WALKL = _npc_all[1]
NPC_SPRITE_WALKR = _npc_all[2]
# die.gif — scale each frame to NPC_H, no cropping
def _load_die_gif(path):
    try:
        p = _try_path(path)
        if not PILLOW_OK:
            raise RuntimeError("no pillow")
        pil = PILImage.open(p)
        frames = []
        try:
            while True:
                surf = _pil_to_pygame(pil.convert("RGBA"))
                fw, fh = surf.get_size()
                nw = max(1, int(NPC_H * fw / fh))
                die_h = int(NPC_H * 2.5)   # die animation bigger than NPC
                nw2 = max(1, int(die_h * fw / fh))
                frames.append(pygame.transform.smoothscale(surf, (nw2, die_h)))
                pil.seek(pil.tell() + 1)
        except EOFError:
            pass
        if frames:
            return frames
    except Exception:
        pass
    return [NPC_SPRITE_WALKL]

NPC_DIE_GIF = _load_die_gif("die.gif")
SOUL_APPEAR_GIF = load_gif_frames("soul_appear.gif",  SOUL_H, (100,200,255),"soul_appear")
SOUL_IDLE_SURF  = load_png("soul_idle.png", SOUL_H, (160,240,255), "soul_idle")
BG_SURF         = load_bg("bg.png")

NPC_DRAW_W = NPC_SPRITE_IDLE.get_width()

# โหลด start screen assets
try:
    COVER_SURF = pygame.image.load(_try_path("cover.png")).convert()
    COVER_SURF = pygame.transform.scale(COVER_SURF, (WIDTH, HEIGHT))
except Exception:
    COVER_SURF = None

try:
    START_BTN = pygame.image.load(_try_path("start.png")).convert_alpha()
    print(f"[ok] start.png loaded: {START_BTN.get_size()}")
except Exception as e:
    START_BTN = None
    print(f"[fail] start.png: {e}")

try:
    HTP_BTN = pygame.image.load(_try_path("htp.png")).convert_alpha()
    print(f"[ok] htp.png loaded: {HTP_BTN.get_size()}")
except Exception as e:
    HTP_BTN = None
    print(f"[fail] htp.png: {e}")

def _find_file(name):
    """หาไฟล์แบบ case-insensitive"""
    if os.path.exists(name):
        return name
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
    try:
        for f in os.listdir(base_dir):
            if f.lower() == name.lower():
                return os.path.join(base_dir, f)
    except Exception:
        pass
    return name

try:
    HTP1_SURF = pygame.image.load(_find_file("htp1.png")).convert()
    HTP1_SURF = pygame.transform.scale(HTP1_SURF, (WIDTH, HEIGHT))
    print(f"[ok] htp1 loaded")
except Exception as e:
    HTP1_SURF = None
    print(f"[warn] htp1.png: {e}")

try:
    HTPBTN1_SURF = pygame.image.load(_find_file("htpbtn1.png")).convert_alpha()
    print(f"[ok] htpbtn1 loaded: {HTPBTN1_SURF.get_size()}")
except Exception as e:
    HTPBTN1_SURF = None
    print(f"[warn] htpbtn1.png: {e}")

try:
    H1_SURF = pygame.image.load(_find_file("h1.png")).convert_alpha()
    print(f"[ok] h1 loaded: {H1_SURF.get_size()}")
except Exception as e:
    H1_SURF = None
    print(f"[warn] h1.png: {e}")

try:
    H2_SURF = pygame.image.load(_find_file("h2.png")).convert_alpha()
    print(f"[ok] h2 loaded: {H2_SURF.get_size()}")
except Exception as e:
    H2_SURF = None
    print(f"[warn] h2.png: {e}")

try:
    HTP2_SURF = pygame.image.load(_find_file("htp2.png")).convert()
    HTP2_SURF = pygame.transform.scale(HTP2_SURF, (WIDTH, HEIGHT))
    print(f"[ok] htp2 loaded")
except Exception as e:
    HTP2_SURF = None
    print(f"[warn] htp2.png: {e}")

try:
    HTPBTN2_SURF = pygame.image.load(_find_file("htpbtn2.png")).convert_alpha()
    print(f"[ok] htpbtn2 loaded: {HTPBTN2_SURF.get_size()}")
except Exception as e:
    HTPBTN2_SURF = None
    print(f"[warn] htpbtn2.png: {e}")

# Start screen animation
START_GIF = load_gif_frames("start.gif", HEIGHT, (20,10,30), "start")
start_gif_tick  = 0
start_gif_frame = 0

# ══════════════════════════════════════════════════════════════
#  Shadow
# ══════════════════════════════════════════════════════════════
def make_shadow(surf):
    fw, fh = surf.get_size()
    s    = pygame.transform.flip(surf, False, True).copy()
    dark = pygame.Surface((fw,fh), pygame.SRCALPHA)
    dark.fill((0,0,0,70))
    s.blit(dark, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
    mask = pygame.Surface((fw,fh), pygame.SRCALPHA)
    for i in range(fh):
        a = max(0, 110-int(110*i/fh))
        pygame.draw.line(mask,(255,255,255,a),(0,i),(fw,i))
    s.blit(mask,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
    return s

# ══════════════════════════════════════════════════════════════
#  NPC class
# ══════════════════════════════════════════════════════════════
class NPC:
    DIE_SHOW   = 80
    APPEAR_DUR = 90

    def __init__(self, x):
        self.x      = float(x)
        self.y      = float(FLOOR_Y - NPC_H)
        self.width  = NPC_DRAW_W
        self.height = NPC_H
        self.speed  = random.uniform(1.2, 2.5)
        self.facing = random.choice(["left","right"])
        self.anim_tick  = 0
        self.anim_frame = 0
        self.die_gif_tick  = 0
        self.die_gif_frame = 0

        self.state       = "walk"
        self.state_timer = random.randint(120, 300)
        self.die_timer   = 0
        self.appear_timer = 0
        self.soul_alpha   = 0

        self.is_target        = False
        self.soul_collectible = False
        self.soul_collected   = False

        self.soul_x  = float(x)
        self.soul_y  = float(FLOOR_Y - NPC_H - SOUL_H)
        self.soul_vx = 0.0
        self.soul_vy = 0.0

    def _pick_next_state(self):
        if random.random() < 0.3:
            self.state = "idle"
            self.state_timer = random.randint(60, 120)
        else:
            self.state = "walk"
            self.facing = random.choice(["left","right"])
            self.state_timer = random.randint(120, 240)

    def trigger_death(self):
        if self.state not in ("dying","soul","collected"):
            self.state = "dying"
            self.die_timer = self.die_gif_tick = self.die_gif_frame = 0

    @property
    def soul_rect(self):
        sw = SOUL_IDLE_SURF.get_width()
        return pygame.Rect(int(self.soul_x)-sw//2-10, int(self.soul_y)-10,
                           sw+20, SOUL_H+20)

    def _move_soul(self, rx, ry):
        sw = SOUL_IDLE_SURF.get_width()
        cx, cy = self.soul_x+sw/2, self.soul_y+SOUL_H/2
        dx, dy = cx-rx, cy-ry
        dist = math.hypot(dx, dy)
        t    = pygame.time.get_ticks() * 0.001

        # --- teleport เมื่อ reaper เข้าใกล้มาก ---
        if dist < 80 and random.random() < 0.015:
            self.soul_teleport_flash = 20   # กระพริบ 20 frame ก่อน teleport
        if hasattr(self, 'soul_teleport_flash') and self.soul_teleport_flash > 0:
            self.soul_teleport_flash -= 1
            if self.soul_teleport_flash == 0:
                self.soul_x = float(random.randint(100, WORLD_WIDTH-100))
                self.soul_y = float(random.randint(int(SOUL_MIN_Y)+20, int(SOUL_MAX_Y)-20))
                self.soul_vx = random.uniform(-SOUL_FLEE_SPEED, SOUL_FLEE_SPEED)
                self.soul_vy = random.uniform(-SOUL_FLEE_SPEED, SOUL_FLEE_SPEED)
                return

        if dist < SOUL_DETECT and dist > 1:
            # หนีตรง + ส่าย zigzag
            angle = math.atan2(dy, dx)
            zigzag = math.sin(t * 8) * 1.2          # ส่ายซ้ายขวาเร็ว
            angle += zigzag + random.uniform(-0.2, 0.2)
            boost = 1.0 + max(0, (120-dist)/120)*1.5  # ยิ่งใกล้ยิ่งเร็ว
            self.soul_vx += math.cos(angle) * SOUL_FLEE_SPEED * 0.5 * boost
            self.soul_vy += math.sin(angle) * SOUL_FLEE_SPEED * 0.5 * boost

            # ดักทิศ: ถ้า reaper วิ่งตรงมา ให้เบี่ยงฉับพลัน
            if dist < 150 and random.random() < 0.08:
                perp = angle + math.pi/2 * random.choice([-1,1])
                self.soul_vx += math.cos(perp) * SOUL_FLEE_SPEED * 1.5
                self.soul_vy += math.sin(perp) * SOUL_FLEE_SPEED * 1.5
        else:
            # ล่องลอยแบบ sine wave เมื่อไม่มีภัย
            self.soul_vx += math.sin(t*1.3 + self.soul_y*0.01) * 0.15
            self.soul_vy += math.cos(t*1.7 + self.soul_x*0.01) * 0.10
            if math.hypot(self.soul_vx, self.soul_vy) < 0.3:
                a = random.uniform(0, math.pi*2)
                self.soul_vx = math.cos(a) * SOUL_SPEED * 0.8
                self.soul_vy = math.sin(a) * SOUL_SPEED * 0.8

        self.soul_vx *= 0.84
        self.soul_vy *= 0.84
        max_spd = SOUL_FLEE_SPEED * 1.4 if dist < SOUL_DETECT else SOUL_SPEED
        spd = math.hypot(self.soul_vx, self.soul_vy)
        if spd > max_spd:
            self.soul_vx = self.soul_vx/spd*max_spd
            self.soul_vy = self.soul_vy/spd*max_spd

        self.soul_x = max(20.0, min(float(WORLD_WIDTH-sw-20), self.soul_x+self.soul_vx))
        self.soul_y = max(SOUL_MIN_Y, min(SOUL_MAX_Y, self.soul_y+self.soul_vy))
        if self.soul_x<=20 or self.soul_x>=WORLD_WIDTH-sw-20: self.soul_vx*=-0.8
        if self.soul_y<=SOUL_MIN_Y or self.soul_y>=SOUL_MAX_Y: self.soul_vy*=-0.8

    def update(self, rx=None, ry=None):
        if self.state == "walk":
            # toggle 0↔1 (walkl1 ↔ walkr1 stepping cycle)
            self.anim_tick += 1
            if self.anim_tick >= NPC_ANIM_SPD:
                self.anim_tick  = 0
                self.anim_frame = 1 - self.anim_frame

        if self.state == "idle":
            self.state_timer -= 1
            if self.state_timer <= 0: self._pick_next_state()

        elif self.state == "walk":
            dx = -self.speed if self.facing=="left" else self.speed
            self.x = max(50.0, min(float(WORLD_WIDTH-self.width-50), self.x+dx))
            if self.x<=50 or self.x>=WORLD_WIDTH-self.width-50:
                self.facing = "right" if self.facing=="left" else "left"
            self.state_timer -= 1
            if self.state_timer <= 0: self._pick_next_state()

        elif self.state == "dying":
            self.die_gif_tick += 1
            if self.die_gif_tick >= GIF_ANIM_SPD:
                self.die_gif_tick  = 0
                self.die_gif_frame = min(self.die_gif_frame+1, len(NPC_DIE_GIF)-1)
            self.die_timer += 1
            if self.die_timer >= self.DIE_SHOW:
                self.state = "soul"
                self.appear_timer = self.soul_alpha = 0
                self.soul_x = self.x + self.width/2
                self.soul_y = self.y - SOUL_H + self.height/2

        elif self.state == "soul":
            self.appear_timer += 1
            self.soul_alpha = min(255, int(255*self.appear_timer/self.APPEAR_DUR))
            if self.appear_timer > self.APPEAR_DUR:
                self.soul_collectible = True
            if self.soul_collectible and rx is not None:
                self._move_soul(rx, ry)

    def draw(self, surf, cam_x):
        sx = int(self.x) - cam_x

        if self.state == "idle":
            # show idle1, flipped to match facing direction
            base = NPC_SPRITE_IDLE
            if self.facing == "left":
                base = pygame.transform.flip(base, True, False)
            surf.blit(base, (sx, int(self.y)))

        elif self.state == "walk":
            # alternate walkl1 ↔ walkr1, flip when facing left
            base = NPC_SPRITE_WALKL if self.anim_frame == 0 else NPC_SPRITE_WALKR
            if self.facing == "left":
                base = pygame.transform.flip(base, True, False)
            surf.blit(base, (sx, int(self.y)))
        elif self.state in ("dying", "soul"):
            frame = NPC_DIE_GIF[self.die_gif_frame] if self.state == "dying" else NPC_DIE_GIF[-1]
            fh = frame.get_height()
            # bottom-anchor: feet stay at floor level
            dy = int(self.y) + NPC_H - fh
            surf.blit(frame, (sx, dy))
            # soul
            if self.appear_timer <= self.APPEAR_DUR:
                ap_fi = min(int(self.appear_timer/self.APPEAR_DUR*len(SOUL_APPEAR_GIF)),
                            len(SOUL_APPEAR_GIF)-1)
                soul_s = SOUL_APPEAR_GIF[ap_fi].copy()
            else:
                soul_s = SOUL_IDLE_SURF.copy()
            # flip soul based on movement direction (left = flip)
            if self.soul_vx < -0.1:
                soul_s = pygame.transform.flip(soul_s, True, False)
            # กระพริบก่อน teleport — สลับ alpha
            if hasattr(self, 'soul_teleport_flash') and self.soul_teleport_flash > 0:
                flash_on = (self.soul_teleport_flash // 3) % 2 == 0
                soul_s.set_alpha(0 if flash_on else self.soul_alpha)
            else:
                soul_s.set_alpha(self.soul_alpha)
            sw     = soul_s.get_width()
            draw_x = int(self.soul_x)-cam_x-sw//2
            draw_y = int(self.soul_y)
            surf.blit(soul_s, (draw_x, draw_y))
            if self.soul_collectible:
                pulse = int(60+60*math.sin(pygame.time.get_ticks()*0.005))
                glow  = pygame.Surface((90,90), pygame.SRCALPHA)
                pygame.draw.circle(glow,(140,220,255,pulse),(45,45),40)
                surf.blit(glow,(draw_x+sw//2-45, draw_y+SOUL_H//2-45))

# ══════════════════════════════════════════════════════════════
#  HUD helpers
# ══════════════════════════════════════════════════════════════
def draw_hud(timer_left, needed, got, tmr_running):
    hud = pygame.Surface((260,80), pygame.SRCALPHA)
    hud.fill((0,0,0,130))
    screen.blit(hud,(8,8))
    screen.blit(font_sm.render(f"Souls: {got} / {needed}", True,(180,240,255)),(18,16))
    if tmr_running:
        mins,secs = timer_left//60, timer_left%60
        col = (255,80,80) if timer_left<30 else (255,220,80)
        screen.blit(font_mid.render(f"Time  {mins:02d}:{secs:02d}", True,col),(18,46))
    else:
        screen.blit(font_sm.render("Waiting for a soul...",True,(200,200,200)),(18,46))

def draw_label(text, x, y, color=(255,255,180)):
    shadow = font_sm.render(text,True,(0,0,0))
    for ddx,ddy in [(-1,0),(1,0),(0,-1),(0,1)]:
        screen.blit(shadow,(x+ddx,y+ddy))
    screen.blit(font_sm.render(text,True,color),(x,y))

# ══════════════════════════════════════════════════════════════
#  Game state
# ══════════════════════════════════════════════════════════════
SOULS_NEEDED = 3
NPC_COUNT    = 8

player_x = 400.0
player_y = REAPER_MAX_Y
pw       = REAPER_IDLE_SURF.get_width()
ph       = TARGET_CHAR_H
facing   = "right"
is_moving = False

reaper_gif_tick  = 0
reaper_gif_frame = 0
camera_x         = 0

collecting     = False
collect_timer  = 0
COLLECT_FRAMES = 60

# Item system
ITEM_W, ITEM_H     = 30, 30
ITEM_DURATION      = 600          # 10 วิ (60fps)
ITEM_COLLECT_BOOST = 3            # เก็บวิญญาณเร็วขึ้น 3x
item_active   = False             # มี item อยู่บนจอ
item_x = item_y = 0.0
item_spawn_timer   = random.randint(5*60, 10*60)  # spawn ครั้งแรก 5-10วิ
item_pickup_timer  = 0            # นับ item ที่ผู้เล่นถือ (0=ไม่มี)

souls_collected = 0
timer_running   = False
time_left       = TIMER_SECONDS
last_tick_time  = 0
game_state      = "start"
confirm_exit    = False

npcs = [NPC(random.randint(200, WORLD_WIDTH-200)) for _ in range(NPC_COUNT)]

target_npc_spawn_timer = random.randint(3*60, 10*60)  # 3-10 วิแรก
target_npcs            = []   # list of active target NPCs
souls_spawned          = 0    # จำนวนที่ spawn แล้ว

fade_alpha = 255
fade_in    = True

# ══════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════
running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and game_state in ("success", "fail"):
                if SFX_WIN: SFX_WIN.stop()
                if SFX_LOSE: SFX_LOSE.stop()
                if BGM_CHASE: BGM_CHASE.stop()
                pygame.mixer.music.stop()
                game_state = "playing"
                try:
                    pygame.mixer.music.load(_try_path("sound/bg1.wav"))
                    pygame.mixer.music.set_volume(0.1)
                    pygame.mixer.music.play(-1, fade_ms=1500)
                except: pass
                player_x = 400.0
                player_y = REAPER_MAX_Y
                facing   = "right"
                souls_collected = 0
                timer_running   = False
                time_left       = TIMER_SECONDS
                last_tick_time  = 0
                collect_timer   = 0
                collecting      = False
                item_active       = False
                item_pickup_timer = 0
                item_spawn_timer  = random.randint(5*60, 10*60)
                for n in npcs:
                    n.__init__(random.randint(200, WORLD_WIDTH-200))
                target_npcs[:]   = []
                souls_spawned    = 0
                target_npc_spawn_timer = random.randint(3*60, 10*60)
                fade_alpha = 255
                fade_in    = True
            if event.key == pygame.K_m and game_state in ("success", "fail"):
                if SFX_WIN: SFX_WIN.stop()
                if SFX_LOSE: SFX_LOSE.stop()
                if BGM_CHASE: BGM_CHASE.stop()
                pygame.mixer.music.stop()
                game_state = "start"
                try:
                    pygame.mixer.music.load(_try_path("sound/bgh.wav"))
                    pygame.mixer.music.set_volume(0.1)
                    pygame.mixer.music.play(-1, fade_ms=3000)
                except: pass
                player_x = 400.0
                player_y = REAPER_MAX_Y
                facing   = "right"
                souls_collected = 0
                timer_running   = False
                time_left       = TIMER_SECONDS
                last_tick_time  = 0
                collect_timer   = 0
                collecting      = False
                item_active       = False
                item_pickup_timer = 0
                item_spawn_timer  = random.randint(15*60, 30*60)
                for n in npcs:
                    n.__init__(random.randint(200, WORLD_WIDTH-200))
                target_npcs[:]   = []
                souls_spawned    = 0
                target_npc_spawn_timer = random.randint(3*60, 10*60)
                fade_alpha = 255
                fade_in    = True
                fade_alpha = 255
                fade_in    = True
            if event.key == pygame.K_SPACE and game_state == "start":
                game_state = "playing"
                pygame.mixer.music.stop()
                try:
                    pygame.mixer.music.load(_try_path("sound/bg1.wav"))
                    pygame.mixer.music.set_volume(0.1)
                    pygame.mixer.music.play(-1, fade_ms=1500)
                except: pass
            if event.key == pygame.K_ESCAPE and game_state == "playing" and not confirm_exit:
                confirm_exit = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if confirm_exit:
                yes_rect = pygame.Rect(WIDTH//2 - 120, HEIGHT//2 + 10, 90, 40)
                no_rect  = pygame.Rect(WIDTH//2 + 30,  HEIGHT//2 + 10, 90, 40)
                if yes_rect.collidepoint(mx, my):
                    confirm_exit = False
                    # reset ทุกอย่างกลับค่าเริ่มต้น
                    game_state = "start"
                    pygame.mixer.music.stop()
                    if BGM_CHASE: BGM_CHASE.stop()
                    try:
                        pygame.mixer.music.load(_try_path("sound/bgh.wav"))
                        pygame.mixer.music.set_volume(0.1)
                        pygame.mixer.music.play(-1, fade_ms=3000)
                    except: pass
                    # reset player
                    player_x = 400.0
                    player_y = REAPER_MAX_Y
                    facing   = "right"
                    # reset souls/timer
                    souls_collected = 0
                    timer_running   = False
                    time_left       = TIMER_SECONDS
                    last_tick_time  = 0
                    collect_timer   = 0
                    collecting      = False
                    # reset item
                    item_active       = False
                    item_pickup_timer = 0
                    item_spawn_timer  = random.randint(15*60, 30*60)
                    # reset npcs
                    for n in npcs:
                        n.__init__(random.randint(200, WORLD_WIDTH-200))
                    # reset target npcs
                    target_npcs[:]   = []
                    souls_spawned    = 0
                    target_npc_spawn_timer = random.randint(3*60, 10*60)
                elif no_rect.collidepoint(mx, my):
                    confirm_exit = False
                    game_state = "playing"
                    if timer_running:
                        if BGM_CHASE: BGM_CHASE.play(-1)
                    else:
                        pygame.mixer.music.unpause()
            elif game_state == "playing" and not confirm_exit:
                h2_rect = pygame.Rect(713, 531, 28, 26)
                if h2_rect.collidepoint(mx, my):
                    confirm_exit = True
                    game_state = "paused"
                    pygame.mixer.music.pause()
                    if BGM_CHASE: BGM_CHASE.stop()
            elif game_state == "start":
                start_rect = pygame.Rect(312, 324, 209, 45)
                htp_click  = pygame.Rect(308, 404, 219, 41)
                if start_rect.collidepoint(mx, my):
                    game_state = "playing"
                    pygame.mixer.music.stop()
                    try:
                        pygame.mixer.music.load(_try_path("sound/bg1.wav"))
                        pygame.mixer.music.set_volume(0.1)
                        pygame.mixer.music.play(-1, fade_ms=1500)
                    except: pass
                elif htp_click.collidepoint(mx, my):
                    game_state = "htp"
            elif game_state == "htp":
                htpbtn1_rect = pygame.Rect(690, 47, 45, 37)
                h1_rect = pygame.Rect(63, 39, 25, 27)
                if htpbtn1_rect.collidepoint(mx, my):
                    game_state = "htp2"
                elif h1_rect.collidepoint(mx, my):
                    game_state = "start"
            elif game_state == "htp2":
                htpbtn2_rect = pygame.Rect(65, 46, 38, 39)
                if htpbtn2_rect.collidepoint(mx, my):
                    game_state = "htp"
        if event.type == pygame.KEYDOWN:
            if game_state == "htp":
                game_state = "start"
            elif game_state == "htp2":
                game_state = "htp" 

    # ── pause guard — skip update logic only, still draw below ──
    if game_state == "paused" and not confirm_exit:
        pygame.display.update()
        continue

    # spawn NPC death (spawn ทีละตัว จนครบ SOULS_NEEDED)
    if souls_spawned < SOULS_NEEDED and game_state == "playing":
        target_npc_spawn_timer -= 1
        if target_npc_spawn_timer <= 0:
            candidates = [n for n in npcs if n.state in ("walk","idle") and n not in target_npcs]
            if candidates:
                t = random.choice(candidates)
                t.is_target = True
                t.trigger_death()
                target_npcs.append(t)
                souls_spawned += 1
                target_npc_spawn_timer = random.randint(5*60, 15*60)  # spawn ถัดไปใน 5-15 วิ

    # start timer เมื่อมีวิญญาณตัวแรกออกมา
    any_collectible = any(n.soul_collectible and not n.soul_collected for n in target_npcs)
    if any_collectible and not timer_running:
        timer_running  = True
        last_tick_time = pygame.time.get_ticks()
        pygame.mixer.music.stop()
        if BGM_CHASE:
            BGM_CHASE.play(-1)

    # countdown
    if timer_running and game_state == "playing":
        now = pygame.time.get_ticks()
        if now - last_tick_time >= 1000:
            time_left -= 1
            last_tick_time = now
        if time_left <= 0:
            game_state = "fail"
            pygame.mixer.music.stop()
            if BGM_CHASE: BGM_CHASE.stop()
            if SFX_LOSE: SFX_LOSE.play()

    # item spawn / pickup / expire
    if timer_running and game_state == "playing":
        # spawn
        if not item_active and item_pickup_timer == 0:
            item_spawn_timer -= 1
            if item_spawn_timer <= 0:
                item_active = True
                item_x = float(random.randint(60, WORLD_WIDTH-60))
                item_y = float(FLOOR_Y - ITEM_H - 10)
                item_spawn_timer = random.randint(10*60, 20*60)  # spawn ถัดไป 20-40 วิ

        # pickup
        if item_active:
            item_rect = pygame.Rect(int(item_x)-ITEM_W//2, int(item_y)-ITEM_H//2, ITEM_W, ITEM_H)
            if player_rect.colliderect(item_rect):
                item_active = False
                item_pickup_timer = ITEM_DURATION

        # expire
        if item_pickup_timer > 0:
            item_pickup_timer -= 1

    # input
    keys   = pygame.key.get_pressed()
    e_held = keys[pygame.K_e]

    if game_state == "playing":
        ml = keys[pygame.K_LEFT]
        mr = keys[pygame.K_RIGHT]
        mu = keys[pygame.K_UP]
        md = keys[pygame.K_DOWN]
        is_moving = ml or mr or mu or md
        _spd = REAPER_SPEED * (1.6 if item_pickup_timer > 0 else 1.0)
        if ml: player_x -= _spd; facing = "left"
        if mr: player_x += _spd; facing = "right"
        if mu: player_y -= _spd
        if md: player_y += _spd
        player_x = max(0.0, min(float(WORLD_WIDTH-pw), player_x))
        player_y = max(REAPER_MIN_Y, min(REAPER_MAX_Y, player_y))

    camera_x = max(0, min(int(player_x)-WIDTH//2, WORLD_WIDTH-WIDTH))
    pw = REAPER_IDLE_SURF.get_width()
    player_rect = pygame.Rect(int(player_x), int(player_y), pw, ph)

    # reaper collect GIF
    collecting = e_held
    if collecting:
        reaper_gif_tick += 1
        if reaper_gif_tick >= GIF_ANIM_SPD:
            reaper_gif_tick  = 0
            reaper_gif_frame = (reaper_gif_frame+1) % len(REAPER_COLLECT_GIF)
    else:
        reaper_gif_tick = reaper_gif_frame = 0

    # collect progress — หา soul ที่ผู้เล่นอยู่ใกล้
    collecting_npc = None
    if game_state == "playing":
        for tn in target_npcs:
            if tn.soul_collectible and not tn.soul_collected:
                if player_rect.colliderect(tn.soul_rect):
                    collecting_npc = tn
                    break

    if game_state == "playing" and collecting_npc:
        if e_held:
            speed = ITEM_COLLECT_BOOST if item_pickup_timer > 0 else 1
            collect_timer += speed
            if collect_timer >= COLLECT_FRAMES:
                collecting_npc.soul_collected = True
                collecting_npc.state = "collected"
                souls_collected += 1
                collecting = False; collect_timer = 0
                if souls_collected >= SOULS_NEEDED:
                    game_state = "success"
                    pygame.mixer.music.stop()
                    if BGM_CHASE: BGM_CHASE.stop()
                    if SFX_WIN: SFX_WIN.play()
        else:
            collect_timer = 0
    elif not collecting_npc:
        collect_timer = 0

    # update
    rx_c = player_x + pw/2
    ry_c = player_y + ph/2
    for npc in npcs:
        npc.update(rx_c, ry_c)

    # ── draw ──────────────────────────────────────────────────
    if game_state == "htp":
        if HTP1_SURF:
            screen.blit(HTP1_SURF, (0, 0))
        else:
            screen.fill((10, 8, 28))
        if HTPBTN1_SURF:
            screen.blit(HTPBTN1_SURF, (0, 0))
        if H1_SURF:
            screen.blit(H1_SURF, (0, 0))
        pygame.display.update()
        continue

    if game_state == "htp2":
        if HTP2_SURF:
            screen.blit(HTP2_SURF, (0, 0))
        else:
            screen.fill((10, 8, 28))
        if HTPBTN2_SURF:
            screen.blit(HTPBTN2_SURF, (0, 0))
        pygame.display.update()
        continue

    if game_state == "start":
        mx, my = pygame.mouse.get_pos()
        start_rect = pygame.Rect(312, 324, 209, 45)
        htp_rect   = pygame.Rect(308, 404, 219, 41)
        # วาด cover ก่อน แล้ววาง start.png ทับ
        if COVER_SURF:
            screen.blit(COVER_SURF, (0, 0))
        else:
            screen.fill((10, 8, 28))
        if START_BTN:
            screen.blit(START_BTN, (0, 0))
        if HTP_BTN:
            screen.blit(HTP_BTN, (0, 0))
        pygame.display.update()
        continue

    screen.blit(BG_SURF, (-camera_x, 0))
    pygame.draw.line(screen,(60,60,80),(0,FLOOR_Y),(WIDTH,FLOOR_Y),1)

    for npc in npcs:
        if npc.state != "collected":
            npc.draw(screen, camera_x)

    # draw item on ground
    if item_active:
        ix = int(item_x) - camera_x
        iy = int(item_y)
        pulse = int(180 + 75*math.sin(pygame.time.get_ticks()*0.006))
        item_surf = pygame.Surface((ITEM_W, ITEM_H), pygame.SRCALPHA)
        pygame.draw.circle(item_surf, (255, 220, 80, pulse), (ITEM_W//2, ITEM_H//2), ITEM_W//2)
        pygame.draw.circle(item_surf, (255, 255, 200, 220), (ITEM_W//2, ITEM_H//2), ITEM_W//2-4, 2)
        screen.blit(item_surf, (ix - ITEM_W//2, iy - ITEM_H//2))

    # draw item HUD (timer bar) when holding item
    if item_pickup_timer > 0:
        bar_w = 100
        bar_x = WIDTH - bar_w - 16
        bar_y = 16
        ratio = item_pickup_timer / ITEM_DURATION
        pygame.draw.rect(screen, (40,30,10), (bar_x-2, bar_y-2, bar_w+4, 14))
        pygame.draw.rect(screen, (255,220,80), (bar_x, bar_y, int(bar_w*ratio), 10))
        draw_label("Speed UP!", bar_x - 72, bar_y - 2, (255, 230, 80))

    cur_frame = REAPER_COLLECT_GIF[reaper_gif_frame] if collecting else REAPER_IDLE_SURF
    if facing == "left":
        cur_frame = pygame.transform.flip(cur_frame, True, False)

    ground_dist  = REAPER_MAX_Y - player_y
    total_range  = max(REAPER_MAX_Y-REAPER_MIN_Y, 1)
    shadow_alpha = max(0, int(160*(1.0-ground_dist/total_range)))
    if shadow_alpha > 10:
        sh  = make_shadow(cur_frame)
        sh.set_alpha(shadow_alpha)
        sdx = int(player_x)-camera_x+pw//2-sh.get_width()//2
        screen.blit(sh,(sdx,FLOOR_Y),(0,0,sh.get_width(),min(sh.get_height(),22)))

    screen.blit(cur_frame,(int(player_x)-camera_x, int(player_y)))

    # progress bar
    if (collecting and collecting_npc and
            collecting_npc.soul_collectible and not collecting_npc.soul_collected):
        bar_w = 140
        bar_x = WIDTH//2-bar_w//2
        bar_y = int(player_y)-36
        pygame.draw.rect(screen,(30,30,50),(bar_x-2,bar_y-2,bar_w+4,16))
        pygame.draw.rect(screen,(100,220,255),(bar_x,bar_y,int(bar_w*collect_timer/COLLECT_FRAMES),12))
        draw_label("Collecting...", bar_x, bar_y-20)

    # [E] prompt
    if collecting_npc and not collecting_npc.soul_collected:
        sw = SOUL_IDLE_SURF.get_width()
        draw_label("[E] Hold to collect",
                   int(collecting_npc.soul_x)-camera_x-sw//2,
                   int(collecting_npc.soul_y)-28, (255,255,150))

    draw_hud(time_left, SOULS_NEEDED, souls_collected, timer_running)

    if game_state == "success":
        ov = pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA)
        ov.fill((0,0,0,160)); screen.blit(ov,(0,0))
        for f,txt,col,y in [
            (font_big,"Mission Complete!",(150,240,255),HEIGHT//2-70),
            (font_mid,"Soul delivered.",  (200,255,200),HEIGHT//2),
            (font_sm, "[R] Play again",   (180,180,180),HEIGHT//2+50),
            (font_sm, "[M] Menu",           (180,180,180),HEIGHT//2+85)]:
            t=f.render(txt,True,col); screen.blit(t,(WIDTH//2-t.get_width()//2,y))

    elif game_state == "fail":
        ov = pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA)
        ov.fill((40,0,0,170)); screen.blit(ov,(0,0))
        for f,txt,col,y in [
            (font_big,"Time's Up!",        (255,80,80),  HEIGHT//2-70),
            (font_mid,"The soul escaped.", (255,160,100),HEIGHT//2),
            (font_sm, "[R] Play again",    (180,180,180),HEIGHT//2+50),
            (font_sm, "[M] Menu",            (180,180,180),HEIGHT//2+85)]:
            t=f.render(txt,True,col); screen.blit(t,(WIDTH//2-t.get_width()//2,y))

    # draw H2 button on gameplay
    if game_state in ("playing", "paused") and H2_SURF:
        h2_rect = pygame.Rect(713, 531, 28, 26)   # debug — ปรับพิกัดทีหลัง
        screen.blit(H2_SURF, (0, 0))

    # confirm exit popup
    if confirm_exit:
        ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        screen.blit(ov, (0, 0))
        # box
        box = pygame.Rect(WIDTH//2 - 160, HEIGHT//2 - 60, 320, 130)
        pygame.draw.rect(screen, (30, 30, 50), box, border_radius=12)
        pygame.draw.rect(screen, (150, 150, 200), box, 2, border_radius=12)
        txt = font_mid.render("Do you want to exit?", True, (255, 255, 255))
        screen.blit(txt, (WIDTH//2 - txt.get_width()//2, HEIGHT//2 - 45))
        # Yes
        yes_rect = pygame.Rect(WIDTH//2 - 120, HEIGHT//2 + 10, 90, 40)
        pygame.draw.rect(screen, (180, 60, 60), yes_rect, border_radius=8)
        yt = font_mid.render("Yes", True, (255, 255, 255))
        screen.blit(yt, (yes_rect.centerx - yt.get_width()//2, yes_rect.centery - yt.get_height()//2))
        # No
        no_rect = pygame.Rect(WIDTH//2 + 30, HEIGHT//2 + 10, 90, 40)
        pygame.draw.rect(screen, (60, 140, 60), no_rect, border_radius=8)
        nt = font_mid.render("No", True, (255, 255, 255))
        screen.blit(nt, (no_rect.centerx - nt.get_width()//2, no_rect.centery - nt.get_height()//2))

    # fade in
    if fade_in:
        fade_alpha = max(0, fade_alpha-5)
        if fade_alpha<=0: fade_in=False
    if fade_alpha>0:
        fs=pygame.Surface((WIDTH,HEIGHT)); fs.fill((0,0,0))
        fs.set_alpha(fade_alpha); screen.blit(fs,(0,0))

    pygame.display.update()

pygame.quit()
sys.exit()
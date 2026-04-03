"""
Пошук дрона на відео: оптичний потік + стерео диспарність.

Логіка виявлення:
  1. Оптичний потік Фарнебека → маска рухомих пікселів
  2. StereoBM → глибина в метрах для кожного пікселя
  3. Перетин масок: об'єкт рухається І знаходиться в 10–500 м
  4. grow_count: лічильник кадрів поспіль, в яких об'єкт наближається
  5. Якщо grow_count >= 5 → статус TARGET LOCKED, малюємо прицел
"""

import cv2
import numpy as np
import glob
import os

# ─────────────────────────────────────────────
# Налаштування шляхів
# ─────────────────────────────────────────────
CALIB_FILE  = "2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt"
LEFT_DIR    = "2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data"
RIGHT_DIR   = "2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_03/data"
OUTPUT_FILE = "drone_finder_result.mp4"

# ─────────────────────────────────────────────
# Параметри детектора
# ─────────────────────────────────────────────
MIN_DEPTH_M       = 10.0    # ближче — занадто близько для дрона
MAX_DEPTH_M       = 500.0   # далі — не видно
FLOW_THRESHOLD    = 2.5     # мінімальна норма оптичного потоку (пікселів/кадр)
MIN_BLOB_AREA     = 200     # мінімальна площа кандидата (px²)
LOCK_FRAMES       = 5       # grow_count поріг для TARGET LOCKED


# ─────────────────────────────────────────────
# Крок 1: Калібрування
# ─────────────────────────────────────────────
# Зчитуємо P_rect_02 і P_rect_03 з calib_cam_to_cam.txt.
# Базова лінія (baseline) у метрах обчислюється з різниці tx-компонент:
#   P[0,3] = -f * Tx  →  baseline = (tx_L - tx_R) / f

def load_calibration(path):
    """Повертає (f, baseline, cx, cy) з файлу калібрування KITTI."""
    params = {}
    with open(path) as fh:
        for line in fh:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    params[key.strip()] = [float(x) for x in val.split()]
                except ValueError:
                    pass  # рядки з датою/текстом — пропускаємо

    P2 = np.array(params["P_rect_02"]).reshape(3, 4)
    P3 = np.array(params["P_rect_03"]).reshape(3, 4)

    f        = P2[0, 0]
    cx, cy   = P2[0, 2], P2[1, 2]
    baseline = abs(P2[0, 3] - P3[0, 3]) / f   # метри

    print(f"Калібрування: f={f:.1f}px  baseline={baseline:.4f}м  cx={cx:.1f} cy={cy:.1f}")
    return f, baseline, cx, cy


# ─────────────────────────────────────────────
# Крок 2: StereoBM — карта глибини
# ─────────────────────────────────────────────
# StereoBM порівнює горизонтальні блоки пікселів між лівим і правим
# зображеннями і знаходить горизонтальний зсув (диспарність).
# Глибина: depth = f * baseline / (disp / 16)

def make_stereo():
    """Створює налаштований StereoBM для KITTI."""
    s = cv2.StereoBM_create(numDisparities=96, blockSize=15)
    s.setMinDisparity(0)
    s.setSpeckleRange(2)
    s.setSpeckleWindowSize(100)
    s.setUniquenessRatio(10)
    s.setTextureThreshold(10)
    return s

def compute_depth(stereo, left_gray, right_gray, f, baseline):
    """Повертає карту глибини float32 у метрах (0 = невалідний піксель)."""
    disp_raw = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    depth    = np.zeros_like(disp_raw)
    valid    = disp_raw > 0
    depth[valid] = (f * baseline) / disp_raw[valid]
    return depth


# ─────────────────────────────────────────────
# Крок 3: Оптичний потік Фарнебека
# ─────────────────────────────────────────────
# Фарнебек обчислює густий потік: для кожного пікселя (vx, vy).
# Норма вектора ||v|| > FLOW_THRESHOLD → піксель рухається.
# Щоб відокремити рух об'єкта від руху камери, віднімаємо
# медіанний потік всього кадру (апроксимація руху фону).

def compute_flow_mask(prev_gray, curr_gray):
    """
    Повертає бінарну маску (uint8, 255=рух) пікселів,
    що рухаються інакше ніж загальний потік фону.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,   # масштаб піраміди
        levels=3,        # рівнів піраміди
        winsize=15,      # розмір вікна
        iterations=3,    # ітерацій на рівні
        poly_n=5,        # розмір поліноміального сусідства
        poly_sigma=1.2,  # σ гаусіана
        flags=0
    )

    # Потік у двох каналах: fx, fy
    fx, fy = flow[..., 0], flow[..., 1]

    # Медіанний потік — апроксимація руху камери (фону)
    med_fx = float(np.median(fx))
    med_fy = float(np.median(fy))

    # Відносний потік (рух відносно фону)
    rel_fx = fx - med_fx
    rel_fy = fy - med_fy

    # Норма відносного потоку
    magnitude = np.sqrt(rel_fx**2 + rel_fy**2)

    # Маска: де норма перевищує поріг
    mask = (magnitude > FLOW_THRESHOLD).astype(np.uint8) * 255

    # Морфологія: прибираємо шум і з'єднуємо фрагменти
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask


# ─────────────────────────────────────────────
# Крок 4: Об'єднання масок і пошук кандидатів
# ─────────────────────────────────────────────
# Кандидат = піксель, що рухається (flow_mask) І знаходиться
# в діапазоні глибин [MIN_DEPTH_M, MAX_DEPTH_M].
# З кандидатів беремо зв'язні компоненти (blob-и).

def find_candidates(flow_mask, depth_map):
    """
    Повертає список словників:
      {'bbox': (x,y,w,h), 'dist': float, 'cx': int, 'cy': int}
    """
    # Маска глибини: об'єкт в допустимому діапазоні
    depth_mask = (
        (depth_map > MIN_DEPTH_M) & (depth_map < MAX_DEPTH_M)
    ).astype(np.uint8) * 255

    # Перетин двох масок
    combined = cv2.bitwise_and(flow_mask, depth_mask)

    # Знаходимо контури blob-ів
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_BLOB_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Медіанна відстань в регіоні (стійко до шуму)
        roi = depth_map[y:y+h, x:x+w]
        valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M)]
        if len(valid) < 10:
            continue

        dist = float(np.median(valid))
        candidates.append({
            "bbox": (x, y, w, h),
            "dist": dist,
            "cx":   x + w // 2,
            "cy":   y + h // 2,
        })

    # Сортуємо за відстанню — найближчий першим
    candidates.sort(key=lambda c: c["dist"])
    return candidates


# ─────────────────────────────────────────────
# Крок 5: Трекер grow_count
# ─────────────────────────────────────────────
# Для кожного кандидата порівнюємо поточну відстань з попередньою.
# Якщо dist < prev_dist (наближається) — інкремент лічильника,
# інакше — скидаємо до 0.
# Використовуємо словник за ID "слоту" (найближчий кандидат = слот 0).

class ApproachTracker:
    """
    Відстежує кандидатів між кадрами по просторовій близькості центрів.
    Зіставлення по індексу ненадійне — порядок blob-ів змінюється щокадру.
    Тут кожен трек має унікальний ID і зіставляється жадібно з кандидатом,
    чий центр найближче до передбаченого положення.
    """
    MAX_MATCH_DIST = 60   # максимальна відстань між центрами для зіставлення (px)

    def __init__(self):
        # track_id → {'cx': int, 'cy': int, 'prev_dist': float, 'grow_count': int}
        self._tracks  = {}
        self._next_id = 0

    def update(self, candidates):
        """
        Зіставляє кандидатів з активними треками по найближчому центру.
        Повертає список кандидатів з доданим полем 'grow_count'.
        """
        # Помічаємо всіх кандидатів дефолтним grow_count=0
        for c in candidates:
            c["grow_count"] = 0

        used_tracks = set()
        unmatched   = list(range(len(candidates)))

        # Жадібне зіставлення: кожен кандидат → найближчий трек
        for ci in list(unmatched):
            cand = candidates[ci]
            best_id, best_d = None, self.MAX_MATCH_DIST

            for tid, trk in self._tracks.items():
                if tid in used_tracks:
                    continue
                d = np.hypot(cand["cx"] - trk["cx"], cand["cy"] - trk["cy"])
                if d < best_d:
                    best_d, best_id = d, tid

            if best_id is not None:
                trk  = self._tracks[best_id]
                grow = trk["grow_count"] + 1 if cand["dist"] < trk["prev_dist"] else 0
                cand["grow_count"] = grow
                used_tracks.add(best_id)
                unmatched.remove(ci)
                # Оновлюємо трек
                self._tracks[best_id] = {
                    "cx": cand["cx"], "cy": cand["cy"],
                    "prev_dist": cand["dist"], "grow_count": grow,
                }

        # Нові кандидати без пари → створюємо нові треки
        # Одразу додаємо їх ID до used_tracks, щоб cleanup не видалив їх
        for ci in unmatched:
            cand = candidates[ci]
            new_id = self._next_id
            self._tracks[new_id] = {
                "cx": cand["cx"], "cy": cand["cy"],
                "prev_dist": cand["dist"], "grow_count": 0,
            }
            used_tracks.add(new_id)
            self._next_id += 1

        # Видаляємо треки без пари в цьому кадрі (об'єкт зник)
        for tid in list(self._tracks.keys()):
            if tid not in used_tracks:
                del self._tracks[tid]

        return candidates


# ─────────────────────────────────────────────
# Крок 6: Малювання прицела і HUD
# ─────────────────────────────────────────────

def draw_crosshair(img, cx, cy, size=30, color=(0, 0, 255), thickness=2):
    """Малює класичний прицел: коло + хрест."""
    cv2.circle(img, (cx, cy), size,      color, thickness)
    cv2.circle(img, (cx, cy), size // 5, color, -1)           # центральна точка
    # Горизонтальні промені (з розривом)
    gap = size // 3
    cv2.line(img, (cx - size - gap, cy), (cx - gap, cy), color, thickness)
    cv2.line(img, (cx + gap, cy), (cx + size + gap, cy), color, thickness)
    # Вертикальні промені
    cv2.line(img, (cx, cy - size - gap), (cx, cy - gap), color, thickness)
    cv2.line(img, (cx, cy + gap), (cx, cy + size + gap), color, thickness)

def draw_hud(frame, candidates, frame_idx, total):
    """Малює HUD: статус, прицели, відстані, номер кадру."""
    h, w = frame.shape[:2]
    locked = any(c["grow_count"] >= LOCK_FRAMES for c in candidates)

    # ── Статус угорі по центру ──
    status_text  = "TARGET LOCKED" if locked else "SEARCHING..."
    status_color = (0, 0, 255)     if locked else (0, 255, 0)
    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, status_text, ((w - tw) // 2, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # ── Номер кадру ──
    cv2.putText(frame, f"Frame {frame_idx+1}/{total}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Кандидати ──
    for cand in candidates:
        x, y, bw, bh = cand["bbox"]
        cx, cy        = cand["cx"], cand["cy"]
        dist          = cand["dist"]
        grow          = cand["grow_count"]
        is_locked     = grow >= LOCK_FRAMES

        if is_locked:
            # Великий червоний прицел
            draw_crosshair(frame, cx, cy, size=40, color=(0, 0, 255), thickness=2)
            box_color  = (0, 0, 255)
            info_color = (0, 0, 255)
        else:
            # Жовтий прямокутник — підозрілий кандидат
            box_color  = (0, 200, 255)
            info_color = (0, 200, 255)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), box_color, 1)

        # Відстань і grow_count
        label = f"{dist:.1f}m  x{grow}"
        cv2.putText(frame, label, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)

    return frame


# ─────────────────────────────────────────────
# Головна функція
# ─────────────────────────────────────────────

def main():
    # Завантажуємо калібрування
    f, baseline, cx_cam, cy_cam = load_calibration(CALIB_FILE)

    # Збираємо відсортовані шляхи до зображень
    left_paths  = sorted(glob.glob(os.path.join(LEFT_DIR,  "*.png")))
    right_paths = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))
    total = min(len(left_paths), len(right_paths))
    print(f"Кадрів: {total}")

    if total < 2:
        print("Потрібно мінімум 2 кадри для оптичного потоку.")
        return

    # Ініціалізуємо запис відео
    sample    = cv2.imread(left_paths[0])
    h, w      = sample.shape[:2]
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(OUTPUT_FILE, fourcc, 10, (w, h))
    print(f"Вихід: {OUTPUT_FILE}  ({w}×{h} @ 10fps)")

    stereo  = make_stereo()
    tracker = ApproachTracker()

    # Перший кадр для оптичного потоку (потрібен попередній кадр)
    prev_left = cv2.cvtColor(cv2.imread(left_paths[0]), cv2.COLOR_BGR2GRAY)

    for idx in range(total):
        left_bgr  = cv2.imread(left_paths[idx])
        right_bgr = cv2.imread(right_paths[idx])

        if left_bgr is None or right_bgr is None:
            print(f"[{idx}] Не вдалося прочитати зображення, пропускаємо.")
            continue

        left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        # Карта глибини через стерео диспарність
        depth = compute_depth(stereo, left_gray, right_gray, f, baseline)

        # Маска рухомих пікселів через оптичний потік
        flow_mask = compute_flow_mask(prev_left, left_gray)

        # Кандидати: перетин руху і глибини
        candidates = find_candidates(flow_mask, depth)

        # Оновлюємо лічильники наближення
        candidates = tracker.update(candidates)

        # Малюємо HUD на копії лівого кадру
        out_frame = draw_hud(left_bgr.copy(), candidates, idx, total)
        writer.write(out_frame)

        # Оновлюємо попередній кадр
        prev_left = left_gray

        # Прогрес кожні 10 кадрів
        if idx % 10 == 0 or idx == total - 1:
            locked_cnt = sum(1 for c in candidates if c["grow_count"] >= LOCK_FRAMES)
            pct        = (idx + 1) / total * 100
            print(f"  Кадр {idx+1:4d}/{total} ({pct:.0f}%)  "
                  f"кандидатів={len(candidates)}  заблокованих={locked_cnt}")

    writer.release()
    print(f"\nГотово! Збережено: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

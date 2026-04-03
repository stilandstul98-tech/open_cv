"""
Стерео відео з KITTI датасету.
Обчислення карти диспарності та глибини для кожного кадру.
Виявлення об'єктів ближче 30 метрів.
"""

import cv2
import numpy as np
import os
import glob

# ─────────────────────────────────────────────
# Шляхи до файлів
# ─────────────────────────────────────────────
CALIB_FILE  = "2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt"
LEFT_DIR    = "2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data"
RIGHT_DIR   = "2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_03/data"
OUTPUT_FILE = "kitti_result.mp4"

# Порог виявлення об'єктів (метри)
DISTANCE_THRESHOLD = 30.0


# ─────────────────────────────────────────────
# Крок 1: Зчитування параметрів калібрування
# ─────────────────────────────────────────────
# P_rect_02 і P_rect_03 — проекційні матриці лівої і правої камер
# після ректифікації. Четвертий елемент першого рядка (tx) кодує
# горизонтальний зсув: tx = -f * baseline.
# Тому: baseline = (tx_left - tx_right) / f

def load_calibration(calib_path):
    """Читаємо calib_cam_to_cam.txt і повертаємо f, baseline, cx, cy."""
    params = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                try:
                    nums = [float(x) for x in val.split()]
                    params[key.strip()] = nums
                except ValueError:
                    pass  # пропускаємо нечислові рядки (наприклад, дата)

    # P_rect_02: матриця 3×4 лівої кольорової камери
    P2 = np.array(params["P_rect_02"]).reshape(3, 4)
    # P_rect_03: матриця 3×4 правої кольорової камери
    P3 = np.array(params["P_rect_03"]).reshape(3, 4)

    # Фокусна відстань (px) — однакова для обох після ректифікації
    f  = P2[0, 0]
    cx = P2[0, 2]
    cy = P2[1, 2]

    # Базова лінія в метрах: різниця tx-компонент поділена на f
    # P[0,3] = -f * Tx  =>  Tx = -P[0,3] / f
    baseline = (P2[0, 3] - P3[0, 3]) / f   # від'ємне мінус від'ємне = позитивне

    print(f"Калібрування завантажено:")
    print(f"  f        = {f:.2f} px")
    print(f"  cx, cy   = {cx:.2f}, {cy:.2f} px")
    print(f"  baseline = {abs(baseline):.4f} м")

    return f, abs(baseline), cx, cy


# ─────────────────────────────────────────────
# Крок 2: Збір шляхів до зображень (відсортовано)
# ─────────────────────────────────────────────
def get_sorted_images(directory):
    """Повертає список PNG-файлів у папці, відсортованих за іменем."""
    paths = sorted(glob.glob(os.path.join(directory, "*.png")))
    return paths


# ─────────────────────────────────────────────
# Крок 3: Налаштування алгоритму StereoBM
# ─────────────────────────────────────────────
# numDisparities — кількість рівнів диспарності (кратне 16)
# blockSize     — розмір блоку порівняння (непарне 5–255)

def create_stereo_matcher():
    """Створює і налаштовує StereoBM для KITTI."""
    stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
    stereo.setMinDisparity(0)
    stereo.setSpeckleRange(2)
    stereo.setSpeckleWindowSize(100)
    stereo.setUniquenessRatio(10)
    stereo.setTextureThreshold(10)
    return stereo


# ─────────────────────────────────────────────
# Крок 4: Перетворення диспарності → глибина
# ─────────────────────────────────────────────
# Формула: depth = f * baseline / disparity
# StereoBM повертає значення помножені на 16 — ділимо на 16.

def disparity_to_depth(disp_raw, f, baseline):
    """Повертає карту глибини у метрах (float32)."""
    disp = disp_raw.astype(np.float32) / 16.0
    # Де диспарність <= 0 — невалідні пікселі, ставимо нуль
    valid = disp > 0
    depth = np.zeros_like(disp)
    depth[valid] = (f * baseline) / disp[valid]
    return depth


# ─────────────────────────────────────────────
# Крок 5: Виявлення об'єктів ближче DISTANCE_THRESHOLD
# ─────────────────────────────────────────────
# Знаходимо зв'язні регіони пікселів з глибиною < порогу,
# відфільтровуємо дрібні шуми (мін. площа 500 px²),
# малюємо прямокутник і відстань.

def draw_close_objects(frame, depth_map, threshold=DISTANCE_THRESHOLD, min_area=500):
    """
    Малює зелені прямокутники навколо об'єктів ближче threshold метрів.
    Повертає анотований кадр.
    """
    result = frame.copy()

    # Маска: валідна глибина і менше порогу
    mask = ((depth_map > 0) & (depth_map < threshold)).astype(np.uint8) * 255

    # Морфологічне закриття — з'єднуємо близькі фрагменти
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Знаходимо контури зв'язних регіонів
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Мінімальна відстань у регіоні (найближча точка)
        roi_depth = depth_map[y:y+h, x:x+w]
        valid_depths = roi_depth[roi_depth > 0]
        if len(valid_depths) == 0:
            continue
        min_dist = float(np.percentile(valid_depths, 5))  # 5-й перцентиль — стійко до шуму

        # Рамка: червона якщо < 10 м, жовта < 20 м, зелена < 30 м
        if min_dist < 10:
            color = (0, 0, 255)
        elif min_dist < 20:
            color = (0, 200, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        label = f"{min_dist:.1f} m"
        cv2.putText(result, label, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return result


# ─────────────────────────────────────────────
# Крок 6: Візуалізація карти глибини (кольорова)
# ─────────────────────────────────────────────

def colorize_depth(depth_map, max_depth=50.0):
    """Перетворює карту глибини на BGR-зображення з кольоровою шкалою."""
    # Нормалізуємо до [0, 255], далеко = темно, близько = яскраво
    depth_vis = np.clip(depth_map, 0, max_depth)
    depth_norm = (1.0 - depth_vis / max_depth) * 255
    depth_norm[depth_map == 0] = 0          # невалідні пікселі — чорні
    depth_uint8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)


def colorize_disparity(disp_raw):
    """Нормалізує сиру диспарність для відображення."""
    disp = disp_raw.astype(np.float32) / 16.0
    disp_vis = np.clip(disp, 0, disp.max())
    disp_norm = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)


# ─────────────────────────────────────────────
# Головна функція
# ─────────────────────────────────────────────

def main():
    # --- Завантажуємо калібрування ---
    f, baseline, cx, cy = load_calibration(CALIB_FILE)

    # --- Збираємо списки зображень ---
    left_imgs  = get_sorted_images(LEFT_DIR)
    right_imgs = get_sorted_images(RIGHT_DIR)
    total = min(len(left_imgs), len(right_imgs))
    print(f"\nЗнайдено кадрів: {total} (ліво={len(left_imgs)}, право={len(right_imgs)})")

    if total == 0:
        print("Помилка: зображення не знайдено!")
        return

    # --- Визначаємо розмір вихідного відео ---
    sample = cv2.imread(left_imgs[0])
    h, w = sample.shape[:2]
    # Три панелі поряд: ліво + диспарність + глибина
    out_w, out_h = w * 3, h

    # --- Ініціалізуємо VideoWriter ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, 10, (out_w, out_h))
    print(f"Вихідний файл: {OUTPUT_FILE}  ({out_w}×{out_h} @ 10 fps)\n")

    # --- Створюємо стерео-матчер ---
    stereo = create_stereo_matcher()

    # --- Головний цикл по кадрах ---
    for idx in range(total):
        left_bgr  = cv2.imread(left_imgs[idx])
        right_bgr = cv2.imread(right_imgs[idx])

        if left_bgr is None or right_bgr is None:
            print(f"[{idx}] Не вдалося прочитати зображення, пропускаємо.")
            continue

        # Переводимо в сірий для StereoBM
        left_gray  = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        # Обчислюємо карту диспарності
        disp_raw = stereo.compute(left_gray, right_gray)

        # Отримуємо карту глибини в метрах
        depth = disparity_to_depth(disp_raw, f, baseline)

        # Малюємо рамки навколо близьких об'єктів
        annotated = draw_close_objects(left_bgr, depth)

        # Кольорові візуалізації для двох правих панелей
        disp_color  = colorize_disparity(disp_raw)
        depth_color = colorize_depth(depth)

        # Підписи панелей
        for img, txt in [(annotated, "Left + Objects"),
                         (disp_color, "Disparity"),
                         (depth_color, "Depth (m)")]:
            cv2.putText(img, txt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Збираємо три панелі горизонтально
        frame_out = np.hstack([annotated, disp_color, depth_color])
        writer.write(frame_out)

        # Прогрес кожні 10 кадрів
        if idx % 10 == 0 or idx == total - 1:
            pct = (idx + 1) / total * 100
            print(f"  Кадр {idx+1:4d}/{total}  ({pct:.0f}%)")

    writer.release()
    print(f"\nГотово! Збережено: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

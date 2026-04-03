"""
Стерео камера: вимірювання відстані до об'єктів
Датасет: Middlebury Cones 2003
Ліва камера: im2.png, Права камера: im6.png
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────
# ПАРАМЕТРИ КАЛІБРУВАННЯ (Middlebury Cones, quarter size 450x375)
# ─────────────────────────────────────────────
FOCAL_LENGTH = 935.0   # фокусна відстань у пікселях
BASELINE     = 0.16    # відстань між камерами у метрах (16 см)

# ─────────────────────────────────────────────
# КРОК 1: Завантаження зображень
# ─────────────────────────────────────────────
# Читаємо обидва знімки — лівий і правий
img_left_color = cv2.imread("cones-png-2/cones/im2.png")
img_right_color = cv2.imread("cones-png-2/cones/im6.png")

if img_left_color is None or img_right_color is None:
    print("ПОМИЛКА: не вдалось знайти зображення.")
    print("Переконайтесь що ви запускаєте з папки open_cv")
    exit()

print(f"Розмір зображення: {img_left_color.shape[1]}x{img_left_color.shape[0]} пікселів")

# StereoBM потребує сірі зображення (grayscale)
img_left_gray  = cv2.cvtColor(img_left_color,  cv2.COLOR_BGR2GRAY)
img_right_gray = cv2.cvtColor(img_right_color, cv2.COLOR_BGR2GRAY)

# ─────────────────────────────────────────────
# КРОК 2: Побудова карти диспарності
# ─────────────────────────────────────────────
# numDisparities — скільки пікселів зсуву шукати (кратне 16)
# blockSize      — розмір блоку для порівняння (непарне число)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# Обчислюємо диспарність — результат у форматі CV_16S (треба ділити на 16)
disparity_raw = stereo.compute(img_left_gray, img_right_gray)

# Конвертуємо в float і нормалізуємо (ділимо на 16 — формат StereoBM)
disparity = disparity_raw.astype(np.float32) / 16.0

# Замінюємо нулі та від'ємні значення на NaN щоб не ділити на 0
disparity_valid = disparity.copy()
disparity_valid[disparity_valid <= 0] = np.nan

print(f"Діапазон диспарності: {np.nanmin(disparity_valid):.1f} — {np.nanmax(disparity_valid):.1f} пікселів")

# ─────────────────────────────────────────────
# КРОК 3: Перетворення диспарності в глибину (метри)
# ─────────────────────────────────────────────
# Формула: depth = (focal_length * baseline) / disparity
depth_map = (FOCAL_LENGTH * BASELINE) / disparity_valid

print(f"Діапазон глибини: {np.nanmin(depth_map):.2f} — {np.nanmax(depth_map):.2f} метрів")

# ─────────────────────────────────────────────
# КРОК 4: Знаходимо об'єкти на карті диспарності
# ─────────────────────────────────────────────
# Беремо тільки пікселі з хорошою диспарністю (ближні об'єкти)
# і шукаємо зв'язні регіони (contours)

# Нормалізуємо для відображення і порогового аналізу
disp_norm = cv2.normalize(disparity_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Порогова фільтрація — залишаємо тільки близькі об'єкти (яскраві пікселі)
_, thresh = cv2.threshold(disp_norm, 100, 255, cv2.THRESH_BINARY)

# Морфологія — прибираємо шум і заповнюємо дірки
kernel = np.ones((7, 7), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

# Знаходимо контури об'єктів
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ─────────────────────────────────────────────
# КРОК 5: Малюємо результати
# ─────────────────────────────────────────────

# --- Вікно 1: Ліве зображення з прямокутниками навколо об'єктів ---
result_left = img_left_color.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:          # пропускаємо дрібні шуми
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    # Рахуємо середню глибину в середині прямокутника
    roi_depth = depth_map[y:y+h, x:x+w]
    mean_depth = np.nanmedian(roi_depth)  # median стійкіший до шуму

    if np.isnan(mean_depth):
        continue

    # Малюємо прямокутник (зелений)
    cv2.rectangle(result_left, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Підпис з відстанню
    label = f"{mean_depth:.2f} m"
    cv2.putText(result_left, label,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    print(f"  Об'єкт [{x},{y} {w}x{h}px]  →  відстань: {mean_depth:.2f} м")

# --- Вікно 2: Кольорова карта диспарності (COLORMAP_JET — синій=далеко, червоний=близько) ---
disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

# Додаємо легенду
cv2.putText(disp_color, "Червоний = близько", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(disp_color, "Синій = далеко",    (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- Вікно 3: Карта глибини у метрах ---
# Нормалізуємо для відображення (замінюємо NaN на 0)
depth_display = depth_map.copy()
depth_display = np.nan_to_num(depth_display, nan=0.0)

# Обмежуємо діапазон 0–5 метрів для кращого відображення
depth_display = np.clip(depth_display, 0, 5.0)

# Перетворюємо в 8-bit (0–255) і застосовуємо кольорову шкалу
depth_8bit = cv2.normalize(depth_display, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_HOT)

# Підписуємо кілька точок з реальними відстанями
h_img, w_img = depth_map.shape
for py in range(50, h_img, 80):
    for px in range(50, w_img, 100):
        d = depth_map[py, px]
        if not np.isnan(d) and 0.1 < d < 10:
            cv2.putText(depth_color, f"{d:.1f}m",
                        (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (200, 200, 200), 1)

# ─────────────────────────────────────────────
# КРОК 6: Зберігаємо всі три зображення як файли
# ─────────────────────────────────────────────
cv2.imwrite("result_left.png",       result_left)
cv2.imwrite("result_disparity.png",  disp_color)
cv2.imwrite("result_depth.png",      depth_color)

print("\nЗбережено файли:")
print("  result_left.png       — ліве зображення з об'єктами")
print("  result_disparity.png  — карта диспарності (колір)")
print("  result_depth.png      — карта глибини (метри)")

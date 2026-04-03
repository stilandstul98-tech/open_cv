import cv2
import numpy as np

VIDEO_FILE = "bird4.mp4"

MIN_AREA = 20
GROW_FRAMES = 5
MIN_GROWTH_PERCENT = 5

cap = cv2.VideoCapture(VIDEO_FILE)
ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

objects = {}
next_id = 1

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    h_frame, w_frame = frame2.shape[:2]

    # Оптический поток
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    fx, fy = flow[..., 0], flow[..., 1]

    # Убираем движение камеры — вычитаем среднее
    diff_x = fx - np.mean(fx)
    diff_y = fy - np.mean(fy)

    magnitude = np.sqrt(diff_x**2 + diff_y**2)

    # Маска объектов которые движутся иначе чем фон
    mask = (magnitude > 2).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_ids = []

    for cont in contours:
        area = cv2.contourArea(cont)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cont)
        cx, cy = x + w // 2, y + h // 2

        # Ищем ближайший объект из предыдущего кадра
        closest_id = None
        min_dist = 80
        for obj_id, obj in objects.items():
            d = distance((cx, cy), obj["pos"])
            if d < min_dist:
                min_dist = d
                closest_id = obj_id

        if closest_id is not None:
            obj = objects[closest_id]
            prev_area = obj["area"]
            grow_count = obj["grow_count"]

            # Процент роста
            if prev_area > 0:
                growth = (area - prev_area) / prev_area * 100
            else:
                growth = 0

            if growth >= MIN_GROWTH_PERCENT:
                grow_count += 1
            else:
                grow_count = 0

            objects[closest_id] = {
                "pos": (cx, cy),
                "area": area,
                "grow_count": grow_count,
                "lost": 0
            }
            current_ids.append(closest_id)
            obj_id = closest_id
        else:
            # Новый объект
            objects[next_id] = {
                "pos": (cx, cy),
                "area": area,
                "grow_count": 0,
                "lost": 0
            }
            current_ids.append(next_id)
            obj_id = next_id
            grow_count = 0
            next_id += 1

        # Рисуем
        if grow_count >= GROW_FRAMES:
            # Цель подтверждена — красный прицел
            cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.line(frame2, (cx-20, cy), (cx+20, cy), (0, 0, 255), 2)
            cv2.line(frame2, (cx, cy-20), (cx, cy+20), (0, 0, 255), 2)
            cv2.putText(frame2, "TARGET", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif grow_count >= 2:
            # Подозрительный объект — жёлтый
            cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 255), 1)

    # Обновляем потерянные объекты
    for obj_id in list(objects.keys()):
        if obj_id not in current_ids:
            objects[obj_id]["lost"] += 1
            if objects[obj_id]["lost"] > 10:
                del objects[obj_id]

    prev_gray = curr_gray

    locked = sum(1 for o in objects.values() if o["grow_count"] >= GROW_FRAMES)
    cv2.putText(frame2, f"TARGET: {locked}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if locked > 0 else (255, 255, 255), 2)

    cv2.imshow("Drone Detection", frame2)

    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
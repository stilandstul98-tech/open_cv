import cv2
import numpy as np
import math

# ============================================
# DASHCAM OBJECT DETECTION SYSTEM
# USING OPTICAL FLOW FOR MOVING VEHICLES
# ============================================
VIDEO_FILE = "bird4.mp4"

# ===== SETTINGS =====
MIN_AREA = 100                   # Minimum object area (noise filter)
MIN_FLOW_MAGNITUDE = 2           # Minimum optical flow magnitude
FLOW_THRESHOLD = 0.5             # Threshold for detecting anomalies in flow
GROW_FRAMES_THRESHOLD = 4        # How many consecutive frames object must grow
MAX_DISTANCE = 100               # Maximum distance for object tracking
MAX_FRAMES_LOST = 5              # Maximum frames object can be lost

# ============================================
# FUNCTION: CALCULATE DISTANCE BETWEEN POINTS
# ============================================
def calculate_distance(point1, point2):
    """Calculates distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ============================================
# FUNCTION: CALCULATE GROWTH PERCENTAGE
# ============================================
def calculate_growth_percent(current_area, prev_area):
    """Calculates area growth percentage"""
    if prev_area == 0:
        return 0
    return ((current_area - prev_area) / prev_area) * 100

# ============================================
# FUNCTION: FIND CLOSEST OBJECT
# ============================================
def find_closest_object(new_pos, objects_dict, max_distance):
    """Finds closest object from dictionary to new position"""
    closest_id = None
    min_dist = max_distance
    
    for obj_id, obj_data in objects_dict.items():
        old_pos = obj_data["position"]
        dist = calculate_distance(new_pos, old_pos)
        if dist < min_dist:
            min_dist = dist
            closest_id = obj_id
    
    return closest_id

# ============================================
# FUNCTION: DETECT ANOMALIES IN OPTICAL FLOW
# ============================================
def detect_flow_anomalies(flow, threshold=0.5):
    """
    Detects pixels where optical flow differs from background.
    
    Optical flow = vector of pixel movement
    Background moves uniformly (camera moving forward)
    Vehicles move differently (slower or different direction)
    
    Returns: binary mask where anomalies are white
    """
    # Calculate magnitude and angle of optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Calculate average flow magnitude (background movement)
    avg_magnitude = np.median(magnitude)
    
    # Find pixels where flow differs significantly from background
    # If pixel moves much slower than background = vehicle ahead!
    anomaly_mask = np.zeros(magnitude.shape, dtype=np.uint8)
    
    # Pixels that move slower than background (vehicles)
    slow_pixels = magnitude < (avg_magnitude * threshold)
    anomaly_mask[slow_pixels] = 255
    
    return anomaly_mask

# ============================================
# MAIN PROGRAM
# ============================================
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print(f"ERROR: Cannot open video '{VIDEO_FILE}'")
    exit()

prev_gray = None
objects = {}  # Dictionary: ID → object data
next_object_id = 1  # Counter for generating IDs
frame_number = 0

print("=" * 70)
print("DASHCAM OBJECT DETECTION SYSTEM")
print("USING OPTICAL FLOW")
print("=" * 70)
print(f"Video: {VIDEO_FILE}")
print(f"Minimum flow magnitude: {MIN_FLOW_MAGNITUDE}")
print(f"Flow threshold: {FLOW_THRESHOLD}")
print(f"Frames for confirmation: {GROW_FRAMES_THRESHOLD}")
print(f"Press 'q' to exit")
print("=" * 70)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("\nVideo finished!")
        break
    
    frame_number += 1
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Remember first frame
    if prev_gray is None:
        prev_gray = gray
        continue
    
    # ===== CALCULATE OPTICAL FLOW =====
    # This calculates the movement vector for each pixel
    # flow[y, x] = (vx, vy) where vx, vy are movement vectors
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None,           # flow (output)
        0.5,            # pyr_scale (pyramid scale)
        3,              # levels (pyramid levels)
        15,             # winsize (averaging window size)
        3,              # iterations
        5,              # poly_n (polynomial expansion)
        1.2,            # poly_sigma
        0               # flags
    )
    
    # ===== DETECT ANOMALIES IN OPTICAL FLOW =====
    # Find pixels that move differently from background
    anomaly_mask = detect_flow_anomalies(flow, FLOW_THRESHOLD)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, kernel)
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the anomaly mask
    conturs, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List of objects found in current frame
    current_frame_objects = []
    
    # ===== PROCESS ALL FOUND OBJECTS =====
    for cont in conturs:
        area = cv2.contourArea(cont)
        
        # Filter noise
        if area < MIN_AREA:
            continue
        
        # Get rectangle coordinates
        x, y, w, h = cv2.boundingRect(cont)
        center_x = x + w // 2
        center_y = y + h // 2
        new_pos = (center_x, center_y)
        
        # Find closest existing object
        closest_id = find_closest_object(new_pos, objects, MAX_DISTANCE)
        
        if closest_id is not None:
            # Object found - update its data
            prev_area = objects[closest_id]["prev_area"]
            grow_count = objects[closest_id]["grow_count"]
            obj_id = closest_id
        else:
            # New object - create new ID
            prev_area = 0
            grow_count = 0
            obj_id = next_object_id
            next_object_id += 1
        
        # ===== KEY LOGIC: CHECK AREA GROWTH =====
        # Calculate area growth percentage
        growth_percent = calculate_growth_percent(area, prev_area)
        
        # If area grew - this vehicle is approaching!
        if growth_percent >= 5 and prev_area > 0:
            grow_count += 1
        else:
            # If no growth - reset counter
            grow_count = 0
        
        # Check: should we lock this target?
        is_locked = objects.get(obj_id, {}).get("is_locked", False)
        
        # If object grows 4 consecutive frames - LOCK it!
        if grow_count >= GROW_FRAMES_THRESHOLD:
            is_locked = True
        
        # Save object with its ID
        objects[obj_id] = {
            "position": new_pos,
            "prev_area": area,
            "grow_count": grow_count,
            "bbox": (x, y, w, h),
            "growth_percent": growth_percent,
            "area": area,
            "frames_lost": 0,  # Counter of lost frames
            "is_locked": is_locked  # Lock flag: if True - show red square
        }
        
        current_frame_objects.append(obj_id)
        
        # ===== VISUALIZATION =====
        # Determine color based on status
        if is_locked:
            # RED square = locked target (approaching vehicle)
            color = (0, 0, 255)
        else:
            # GRAY square = detected object (not yet confirmed)
            color = (128, 128, 128)
        
        # Draw rectangle (just square without text)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # If locked - draw crosshair
        if is_locked:
            cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
            cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)
            if grow_count >= GROW_FRAMES_THRESHOLD:
                print(f"[Frame {frame_number}] VEHICLE DETECTED! Area: {int(area)}, Growth: {growth_percent:.1f}%")
    
    # ===== UPDATE LOST OBJECTS =====
    # Objects that didn't appear in current frame
    objects_to_remove = []
    for obj_id in objects.keys():
        if obj_id not in current_frame_objects:
            # Object lost on this frame
            objects[obj_id]["frames_lost"] += 1
            
            # If lost too long - remove
            if objects[obj_id]["frames_lost"] > MAX_FRAMES_LOST:
                objects_to_remove.append(obj_id)
    
    # Remove lost objects
    for obj_id in objects_to_remove:
        del objects[obj_id]
    
    # Update previous frame
    prev_gray = gray
    
    # ===== SCREEN INFO =====
    locked_count = sum(1 for obj in objects.values() if obj.get("is_locked", False))
    detected_count = len(objects)
    
    cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Locked: {locked_count} | Detected: {detected_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if locked_count > 0 else (255, 255, 255), 2)
    
    # Show current settings
    cv2.putText(frame, f"Min Area: {MIN_AREA} | Threshold: {GROW_FRAMES_THRESHOLD} | Max Lost: {MAX_FRAMES_LOST}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show video
    cv2.imshow("Dashcam Vehicle Detection", frame)
    
    # Exit on 'q' press
    if cv2.waitKey(25) == ord('q'):
        print("\nProgram stopped by user")
        break

# Close everything
cap.release()
cv2.destroyAllWindows()

print("=" * 70)
print("PROGRAM FINISHED")
print("=" * 70)

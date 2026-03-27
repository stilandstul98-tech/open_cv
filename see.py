import cv2
# import numpy as np

cap = cv2.VideoCapture("people.mp4")


first_frame = None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    
    if first_frame is None:
        first_frame = gray
        continue
    screen = cv2.absdiff(first_frame, gray)
    _, threshold = cv2.threshold(screen, 25, 255, cv2.THRESH_BINARY)
    
    conturs, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    for cont in conturs:
        if cv2.contourArea(cont) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    
    
    first_frame = gray
    
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(25) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
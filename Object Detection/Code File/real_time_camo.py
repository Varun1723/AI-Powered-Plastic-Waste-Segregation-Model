# real_time_camo_fixed.py

import cv2
import time
from ultralytics import YOLO
import numpy as np

def main():
    # Load your trained model
    model = YOLO("waste_detection/baseline_train4/weights/best.pt")
    
    # Initialize video capture with MSMF backend
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    
    if not cap.isOpened():
        print("❌ Cannot open Camo Studio camera.")
        return
        
    print("Raw frame size:",
          cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          "×",
          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Performance metrics
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Create trackbars for adjusting parameters
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Confidence', 'Controls', 20, 100, lambda x: None)
    cv2.createTrackbar('IOU Threshold', 'Controls', 30, 100, lambda x: None)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame not received.")
            break
            
        # Get current parameter values from trackbars
        conf_threshold = cv2.getTrackbarPos('Confidence', 'Controls') / 100
        iou_threshold = cv2.getTrackbarPos('IOU Threshold', 'Controls') / 100
        
        # Make a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Run detection with current parameters
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        
        # Draw boxes + labels
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            
            # Use different colors based on confidence
            color = (0, int(255 * min(conf * 1.5, 1.0)), int(255 * (1 - min(conf * 1.5, 1.0))))
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Improved text display with background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add parameter display
        cv2.putText(display_frame, f"Conf: {conf_threshold:.2f}, IOU: {iou_threshold:.2f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show raw feed in smaller window
        small_raw = cv2.resize(frame, (320, 240))
        cv2.imshow("Raw Feed", small_raw)
        
        # Show detection results
        cv2.imshow("Waste Detection", display_frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break
        elif key == ord('s'):  # Save a screenshot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"detection_{timestamp}.jpg", display_frame)
            print(f"✅ Screenshot saved as detection_{timestamp}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
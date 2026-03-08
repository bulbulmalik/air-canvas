import cv2
import numpy as np
from collections import deque
from datetime import datetime

class AirCanvasOpenCV:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        
        print("Initializing Air Canvas (OpenCV Version)...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Cannot access webcam!")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        self.points = deque(maxlen=512)
        
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'purple': (255, 0, 255),
            'yellow': (0, 255, 255),
            'pink': (255, 192, 203),
            'orange': (0, 165, 255)
        }
        
        self.current_color = self.colors['purple']
        self.color_names = list(self.colors.keys())
        self.color_index = self.color_names.index('purple')
        
        self.brush_size = 5
        self.is_erasing = False
        
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        print("✅ Air Canvas initialized successfully (No MediaPipe)!")
    
    def detect_hand_position(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 1000:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy), True
        
        return (0, 0), False
    
    def draw_ui(self, frame):
        height, width = frame.shape[:2]
        
        ui_height = 90
        ui = np.ones((ui_height, width, 3), dtype=np.uint8) * 240
        
        cv2.putText(ui, "Happy Women's Day - Air Canvas (OpenCV)", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 127), 2)
        
        color_name = self.color_names[self.color_index].upper()
        status = "ERASING" if self.is_erasing else f"DRAWING ({color_name})"
        cv2.putText(ui, status, (450, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2)
        
        cv2.putText(ui, f"Size: {self.brush_size}", (850, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        
        cv2.putText(ui, 
                    "C:Color | E:Erase | L:Clear | S:Save | +/-:Size | Q:Quit", 
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        frame_with_ui = np.vstack([ui, frame])
        return frame_with_ui
    
    def add_womens_day_text(self):
        text = "Happy Women's Day!"
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 2
        thickness = 4
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = 120
        
        cv2.putText(self.canvas, text, (text_x, text_y), font, 
                    font_scale, (255, 20, 147), thickness)
        
        cv2.putText(self.canvas, "Celebrate Your Strength", 
                    (self.width//2 - 200, text_y + 80), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (220, 20, 60), 2)
    
    def run(self):
        print("\n" + "="*60)
        print("AIR CANVAS - WOMEN'S DAY EDITION (OpenCV Only)")
        print("="*60)
        print("\nControls:")
        print("  C - Change color")
        print("  E - Toggle eraser")
        print("  L - Clear canvas")
        print("  S - Save drawing")
        print("  + - Increase brush size")
        print("  - - Decrease brush size")
        print("  Q - Quit")
        print("\nDetection: Move your hand in frame to draw")
        print("           Keep hand still to pause")
        print("="*60 + "\n")
        
        self.add_womens_day_text()
        
        hand_stable_count = 0
        previous_pos = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            hand_pos, hand_detected = self.detect_hand_position(frame)
            
            if hand_detected:
                hand_x, hand_y = hand_pos
                
                if previous_pos is not None:
                    distance = np.sqrt((hand_x - previous_pos[0])**2 + (hand_y - previous_pos[1])**2)
                    
                    if distance < 30:
                        hand_stable_count += 1
                    else:
                        hand_stable_count = 0
                    
                    if hand_stable_count < 5:
                        if not self.is_erasing:
                            cv2.circle(self.canvas, hand_pos, self.brush_size, 
                                     self.current_color, -1)
                            if len(self.points) > 0:
                                cv2.line(self.canvas, self.points[-1], hand_pos, 
                                       self.current_color, self.brush_size)
                        else:
                            cv2.circle(self.canvas, hand_pos, self.brush_size + 10, 
                                     (255, 255, 255), -1)
                        
                        self.points.append(hand_pos)
                    else:
                        self.points.clear()
                
                previous_pos = hand_pos
                
                cv2.circle(frame, hand_pos, 8, (0, 255, 0), -1)
                cv2.circle(frame, hand_pos, 8, (0, 0, 0), 2)
            else:
                previous_pos = None
                hand_stable_count = 0
                self.points.clear()
            
            display_frame = cv2.addWeighted(frame, 0.4, self.canvas, 0.6, 0)
            display_with_ui = self.draw_ui(display_frame)
            
            cv2.imshow("Air Canvas - Women's Day (OpenCV)", display_with_ui)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Closing Air Canvas. Thank you!")
                break
            elif key == ord('c'):
                self.is_erasing = False
                self.color_index = (self.color_index + 1) % len(self.color_names)
                self.current_color = self.colors[self.color_names[self.color_index]]
                print(f"✓ Color: {self.color_names[self.color_index]}")
            elif key == ord('e'):
                self.is_erasing = not self.is_erasing
                print(f"✓ Eraser: {'ON' if self.is_erasing else 'OFF'}")
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"womens_day_{timestamp}.png"
                cv2.imwrite(filename, self.canvas)
                print(f"✓ Saved: {filename}")
            elif key == ord('+'):
                self.brush_size = min(self.brush_size + 1, 25)
                print(f"✓ Size: {self.brush_size}")
            elif key == ord('-'):
                self.brush_size = max(self.brush_size - 1, 2)
                print(f"✓ Size: {self.brush_size}")
            elif key == ord('l'):
                self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
                self.points.clear()
                self.add_womens_day_text()
                print("✓ Canvas cleared!")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AirCanvasOpenCV()
    app.run()
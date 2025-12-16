import cv2
import sys
import time

# Configuration constants
BLUR_KERNEL_SIZE = (99, 99)
BLUR_SIGMA = 30
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 4
WINDOW_NAME = "Face Anonymizer"
QUIT_KEY = 'q'
FRAME_DELAY = 7

def initialize_face_detector():
    """Initialize Haar cascade face detector"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        sys.exit(1)
    
    return face_cascade

def initialize_camera():
    """Initialize video capture"""
    video = cv2.VideoCapture(0)
    
    if not video.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
    
    return video

def anonymize_faces(frame, faces):
    """Apply blur to detected faces"""
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, BLUR_KERNEL_SIZE, BLUR_SIGMA)
        frame[y:y+h, x:x+w] = blurred_face
    
    return frame

def main():
    """Main application loop"""
    face_cascade = initialize_face_detector()
    video = initialize_camera()
    
    # FPS calculation variables
    prev_time = time.time()
    fps = 0
    
    print(f"Face Anonymizer started. Press '{QUIT_KEY}' to quit.")
    
    try:
        while True:
            ret, frame = video.read()
            
            if not ret:
                print("Warning: Failed to read frame")
                continue
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)
            
            frame = anonymize_faces(frame, faces)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            
            # Display FPS on frame
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(WINDOW_NAME, frame)
            
            if cv2.waitKey(FRAME_DELAY) & 0xFF == ord(QUIT_KEY):
                break
    
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    
    finally:
        video.release()
        cv2.destroyAllWindows()
        print("Resources cleaned up successfully")

if __name__ == "__main__":
    main()        
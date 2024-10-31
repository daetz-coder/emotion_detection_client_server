import cv2
import logging
from deepface import DeepFace
from tqdm import tqdm

def analyze_video(input_path, output_path):
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {input_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    progress = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        progress.update(1)  # Update progress immediately after reading the frame

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if 'region' in result and 'dominant_emotion' in result:
                face = result['region']
                x, y, w, h = face['x'], face['y'], face['w'], face['h']
                dominant_emotion = result['dominant_emotion']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Continue even if no face or emotion is detected
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
        
        out.write(frame)

    cap.release()
    out.release()
    progress.close()

    return output_path

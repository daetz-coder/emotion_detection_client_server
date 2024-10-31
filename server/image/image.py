# image.py
import cv2
import logging
from deepface import DeepFace

def analyze_image(img_path, output_path):
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Cannot load image: {img_path}")
            return None

        results = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=True)

        if isinstance(results, list):
            faces = results
        else:
            faces = [results]

        for idx, face in enumerate(faces):
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            emotions = face['emotion']
            dominant_emotion = face['dominant_emotion']
            emotion_score = emotions[dominant_emotion]

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{dominant_emotion} ({emotion_score:.2f})"
            text_x = x
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            print(f"Face {idx + 1} detected at: ({x}, {y}, {w}, {h})")
            print("Emotion scores:")
            for emotion, score in emotions.items():
                print(f"  {emotion}: {score:.2f}")

            cv2.imwrite(output_path, img)

        return output_path
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return None

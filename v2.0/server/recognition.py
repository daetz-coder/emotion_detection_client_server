import cv2
import logging
import numpy as np
import logging
from tqdm import tqdm
from deepface import DeepFace

# -------------------- 图像分析功能 --------------------
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

            logging.info(f"Face {idx + 1} detected at: ({x}, {y}, {w}, {h})")
            logging.info("Emotion scores:")
            for emotion, score in emotions.items():
                logging.info(f"  {emotion}: {score:.2f}")

            cv2.imwrite(output_path, img)

        return output_path
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        return None

# -------------------- 视频分析功能 --------------------
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

# -------------------- 实时分析功能 --------------------
def analyze_online(file):
    logging.info("Starting online analysis...")
    try:
        nparr = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logging.error("Failed to decode image")
            return {"error": "Image decoding failed"}
        
        logging.info("Image decoded successfully, starting DeepFace analysis")
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        
        annotations = []
        faces = results if isinstance(results, list) else [results]
        for result in faces:
            region = result.get('region', {})
            bbox = [
                region.get('x', 0),
                region.get('y', 0),
                region.get('w', 0),
                region.get('h', 0)
            ]
            annotations.append({
                "bbox": bbox,
                "label": result.get('dominant_emotion', 'neutral')
            })

        logging.info("Analysis completed successfully")
        return {"annotations": annotations}

    except Exception as e:
        logging.error(f"Error in analyze_online: {e}")
        raise e  # 保留原始异常，便于调试

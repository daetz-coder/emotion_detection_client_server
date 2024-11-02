import cv2
import numpy as np
import logging
from tqdm import tqdm
from deepface import DeepFace
from flask import Flask, request, jsonify

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

            print(f"Face {idx + 1} detected at: ({x}, {y}, {w}, {h})")
            print("Emotion scores:")
            for emotion, score in emotions.items():
                print(f"  {emotion}: {score:.2f}")

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


# -------------------- 视频分析功能 --------------------
def analyze_online():
    try:
        # 从请求中读取视频帧
        file = request.files['frame'].read()
        nparr = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 使用 DeepFace 进行实时分析
        # 只进行情感分析，禁用面部检测
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        annotations = []

        # DeepFace.analyze 可能返回多个结果（如果有多张人脸）
        if isinstance(results, list):
            for result in results:
                region = result.get('region', {})
                dominant_emotion = result.get('dominant_emotion', 'neutral')

                bbox = [
                    region.get('x', 0),
                    region.get('y', 0),
                    region.get('w', 0),
                    region.get('h', 0)
                ]

                annotations.append({
                    "bbox": bbox,
                    "label": dominant_emotion
                })
        else:
            # 单张人脸的情况
            region = results.get('region', {})
            dominant_emotion = results.get('dominant_emotion', 'neutral')

            bbox = [
                region.get('x', 0),
                region.get('y', 0),
                region.get('w', 0),
                region.get('h', 0)
            ]

            annotations.append({
                "bbox": bbox,
                "label": dominant_emotion
            })

        return jsonify({"annotations": annotations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

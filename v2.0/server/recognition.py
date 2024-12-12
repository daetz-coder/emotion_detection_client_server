import cv2
import logging
import numpy as np
from tqdm import tqdm
from collections import deque
from deepface import DeepFace

# 对情绪进行平滑处理的函数
def smooth_emotions(emotion_queue, current_emotions, queue_size=5):
    emotion_queue.append(current_emotions)
    if len(emotion_queue) > queue_size:
        emotion_queue.popleft()

    # 平均情绪分布
    all_keys = list(current_emotions.keys())
    avg_emotions = {k:0.0 for k in all_keys}
    for e in emotion_queue:
        for k,v in e.items():
            avg_emotions[k] += v
    for k in avg_emotions:
        avg_emotions[k] /= len(emotion_queue)

    # 找到平均分布中的dominant emotion
    dominant_emotion = max(avg_emotions, key=avg_emotions.get)
    return avg_emotions, dominant_emotion

# 对人脸框进行平滑处理的函数
def smooth_bounding_box(box_queue, current_box, queue_size=5):
    # current_box是(x, y, w, h)
    box_queue.append(current_box)
    if len(box_queue) > queue_size:
        box_queue.popleft()
    
    # 平均bounding box
    avg_x = sum(b[0] for b in box_queue) / len(box_queue)
    avg_y = sum(b[1] for b in box_queue) / len(box_queue)
    avg_w = sum(b[2] for b in box_queue) / len(box_queue)
    avg_h = sum(b[3] for b in box_queue) / len(box_queue)
    
    return int(avg_x), int(avg_y), int(avg_w), int(avg_h)

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

            # 显示情绪分布
            offset = 0
            for emotion, score in emotions.items():
                cv2.putText(img, f"{emotion}: {score:.2f}", (x, y+h+20+offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                offset += 20

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

    emotion_queue = deque(maxlen=5)
    box_queue = deque(maxlen=5)

    progress = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    frame_count = 0

    # 我们需要在非分析帧使用上一帧的结果，因此存储当前平滑后的情绪和框线信息
    current_dominant_emotion = "neutral"
    current_avg_emotions = {"neutral":1.0}
    current_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        progress.update(1)

        # 每5帧进行一次检测和分析
        if frame_count % 5 == 0: 
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
                faces = result if isinstance(result, list) else [result]

                if faces and 'region' in faces[0]:
                    face_data = faces[0]
                    emotions = face_data['emotion']
                    avg_emotions, dominant_emotion = smooth_emotions(emotion_queue, emotions)

                    # 平滑人脸框
                    x, y = face_data['region']['x'], face_data['region']['y']
                    w, h = face_data['region']['w'], face_data['region']['h']
                    avg_x, avg_y, avg_w, avg_h = smooth_bounding_box(box_queue, (x,y,w,h))

                    current_avg_emotions = avg_emotions
                    current_dominant_emotion = dominant_emotion
                    current_box = (avg_x, avg_y, avg_w, avg_h)
                else:
                    # 未检测到人脸，保持上一帧的结果不变（如果希望在长时间无检测时消失，可增加计数器逻辑）
                    pass

            except Exception as e:
                logging.error(f"Error processing frame: {e}")
        # 非检测帧：直接使用上一帧的结果（情绪和人脸框线）
        # current_box和current_avg_emotions已在上一次检测帧中更新
        # 如果多帧不检测到人脸，可以考虑逐渐淡出显示

        # 显示当前平滑后的结果
        if current_box is not None:
            avg_x, avg_y, avg_w, avg_h = current_box
            cv2.rectangle(frame, (avg_x, avg_y), (avg_x + avg_w, avg_y + avg_h), (0, 255, 0), 2)
            cv2.putText(frame, current_dominant_emotion, (avg_x, avg_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 显示情绪分布
            offset = 0
            # 将情绪按分值排序以便更美观
            sorted_emotions = sorted(current_avg_emotions.items(), key=lambda x:x[1], reverse=True)
            for emotion, score in sorted_emotions:
                cv2.putText(frame, f"{emotion}: {score:.2f}", (avg_x, avg_y + avg_h + 20 + offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                offset += 20

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    progress.close()
    return output_path


# -------------------- 实时分析功能 --------------------
# 类似于视频分析的思路，也对实时输入的图像进行平滑处理。
# 假设此函数被持续调用（例如从摄像头流中读取每一帧），可以在外部维护全局队列实现平滑。
# 如果只是单张图片的分析，那就无法平滑，需要在调用方多次传入连续的图像帧。

# 为了便于演示，这里假设analyze_online被重复调用，并通过闭包或全局变量维持状态。
# 在真实项目中，你可能需要将这些队列作为对象属性保存。

# 全局队列，用于简单示范
global_emotion_queue = deque(maxlen=5)
global_box_queue = deque(maxlen=5)
global_current_dominant_emotion = "neutral"
global_current_avg_emotions = {"neutral":1.0}
global_current_box = (0,0,100,100) # 随便一个初始框,真实情况中应当在有检测结果后更新

def analyze_online(file):
    logging.info("Starting online analysis...")
    global global_emotion_queue, global_box_queue
    global global_current_dominant_emotion, global_current_avg_emotions, global_current_box

    try:
        nparr = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logging.error("Failed to decode image")
            return {"error": "Image decoding failed"}

        # 假设每次调用都进行一次检测（若要降低频率，可在调用方逻辑中实现）
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        faces = results if isinstance(results, list) else [results]

        annotations = []
        if faces and 'region' in faces[0]:
            face_data = faces[0]
            emotions = face_data['emotion']
            avg_emotions, dominant_emotion = smooth_emotions(global_emotion_queue, emotions)

            x, y = face_data['region']['x'], face_data['region']['y']
            w, h = face_data['region']['w'], face_data['region']['h']
            avg_x, avg_y, avg_w, avg_h = smooth_bounding_box(global_box_queue, (x,y,w,h))

            global_current_avg_emotions = avg_emotions
            global_current_dominant_emotion = dominant_emotion
            global_current_box = (avg_x, avg_y, avg_w, avg_h)

            annotations.append({
                "bbox": [avg_x, avg_y, avg_w, avg_h],
                "dominant_emotion": dominant_emotion,
                "emotions": global_current_avg_emotions
            })
        else:
            # 未检测到人脸，使用之前的结果
            # 在实际中可以选择返回空或维持上一次状态
            annotations.append({
                "bbox": list(global_current_box),
                "dominant_emotion": global_current_dominant_emotion,
                "emotions": global_current_avg_emotions
            })

        logging.info("Analysis completed successfully")
        return {"annotations": annotations}

    except Exception as e:
        logging.error(f"Error in analyze_online: {e}")
        return {"error": str(e)}

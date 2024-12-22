import cv2
import logging
import numpy as np
from collections import deque
from EmtionDetection import DeepFace

# 配置日志
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# 对情绪进行平滑处理的函数
def smooth_emotions(emotion_queue, current_emotions, queue_size=5):
    emotion_queue.append(current_emotions)
    if len(emotion_queue) > queue_size:
        emotion_queue.popleft()

    # 平均情绪分布
    all_keys = list(current_emotions.keys())
    avg_emotions = {k: 0.0 for k in all_keys}
    for e in emotion_queue:
        for k, v in e.items():
            avg_emotions[k] += v
    for k in avg_emotions:
        avg_emotions[k] /= len(emotion_queue)

    # 找到平均分布中的dominant emotion
    dominant_emotion = max(avg_emotions, key=avg_emotions.get)
    return avg_emotions, dominant_emotion

# 对人脸框进行平滑处理的函数
def smooth_bounding_box(box_queue, current_box, queue_size=5):
    box_queue.append(current_box)
    if len(box_queue) > queue_size:
        box_queue.popleft()

    # 平均bounding box
    avg_x = sum(b[0] for b in box_queue) / len(box_queue)
    avg_y = sum(b[1] for b in box_queue) / len(box_queue)
    avg_w = sum(b[2] for b in box_queue) / len(box_queue)
    avg_h = sum(b[3] for b in box_queue) / len(box_queue)

    return int(avg_x), int(avg_y), int(avg_w), int(avg_h)

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 0为默认摄像头，如果有多个摄像头，可以更改为1, 2等

    if not cap.isOpened():
        logging.error("无法打开摄像头")
        return

    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # 默认帧率

    # 初始化平滑队列
    emotion_queue = deque(maxlen=5)
    box_queue = deque(maxlen=5)

    # 当前平滑后的情绪和框线信息
    current_dominant_emotion = "neutral"
    current_avg_emotions = {"neutral": 1.0}
    current_box = None

    frame_count = 0
    analyze_every_n_frames = 5  # 每隔5帧进行一次情绪分析

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("无法读取帧")
            break

        display_frame = frame.copy()

        # 每隔n帧进行一次情绪分析
        if frame_count % analyze_every_n_frames == 0:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # DeepFace分析，enforce_detection=False避免无法检测到人脸时报错
                results = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)

                faces = results if isinstance(results, list) else [results]

                if faces and 'region' in faces[0]:
                    face_data = faces[0]
                    emotions = face_data['emotion']
                    avg_emotions, dominant_emotion = smooth_emotions(emotion_queue, emotions)

                    # 平滑人脸框
                    region = face_data['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    avg_x, avg_y, avg_w, avg_h = smooth_bounding_box(box_queue, (x, y, w, h))

                    current_avg_emotions = avg_emotions
                    current_dominant_emotion = dominant_emotion
                    current_box = (avg_x, avg_y, avg_w, avg_h)
                else:
                    # 未检测到人脸，保持上一帧的结果不变
                    pass

            except Exception as e:
                logging.error(f"处理帧时出错: {e}")

        # 如果有检测到人脸，显示平滑后的结果
        if current_box is not None:
            avg_x, avg_y, avg_w, avg_h = current_box
            cv2.rectangle(display_frame, (avg_x, avg_y), (avg_x + avg_w, avg_y + avg_h), (0, 255, 0), 2)
            cv2.putText(display_frame, current_dominant_emotion, (avg_x, avg_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 显示情绪分布
            offset = 0
            sorted_emotions = sorted(current_avg_emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions:
                cv2.putText(display_frame, f"{emotion}: {score:.2f}", (avg_x, avg_y + avg_h + 20 + offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                offset += 20

        # 显示视频
        cv2.imshow('Real-time Emotion Recognition', display_frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

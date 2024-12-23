import cv2
import logging
from logging.handlers import RotatingFileHandler
from EmotionDetection import DeepFace

# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# 创建文件处理器，最大文件大小为5MB，保留3个备份
file_handler = RotatingFileHandler("emotion_recognition.log", maxBytes=5*1024*1024, backupCount=3)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 创建控制台处理器，仅输出ERROR级别的日志
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    # 初始化摄像头（0为默认摄像头）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return

    # 可选：设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化分析频率控制
    frame_count = 0
    analyze_every_n_frames = 5  # 每隔5帧进行一次情绪、年龄和性别分析
    current_dominant_emotion = "neutral"
    current_emotions = {}
    current_age = "N/A"
    current_gender = "N/A"
    region = {'x': 0, 'y': 0, 'w': 100, 'h': 100}  # 初始化区域以避免未定义错误

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("无法读取帧")
            break

        try:
            if frame_count % analyze_every_n_frames == 0:
                # DeepFace 分析当前帧，包括情绪、年龄和性别
                results = DeepFace.analyze(
                    frame, 
                    actions=['emotion', 'age', 'gender'], 
                    enforce_detection=False,
                    detector_backend='opencv'  # 可根据需要更改
                )

                # 如果检测到多张人脸，可以遍历 results
                if isinstance(results, list):
                    faces = results
                else:
                    faces = [results]

                face_detected = False

                for face in faces:
                    if face['region'] is None:
                        # 未检测到人脸
                        continue

                    face_detected = True
                    region = face['region']
                    emotions = face['emotion']
                    dominant_emotion = face['dominant_emotion']
                    emotion_score = emotions[dominant_emotion]
                    age = face.get('age', 'N/A')
                    gender = face.get('gender', 'N/A')

                    # 更新当前情绪、年龄和性别
                    current_dominant_emotion = dominant_emotion
                    current_emotions = emotions
                    current_age = age
                    current_gender = gender

                if not face_detected:
                    # 没有检测到任何人脸，保持上一次的结果
                    pass

        except Exception as e:
            logger.error(f"处理帧时出错: {e}")
            # 仅在出现错误时显示提示
            cv2.putText(frame, "检测出错", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 绘制当前情绪、年龄和性别信息（如果有）
        if current_emotions:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 显示主导情绪
            emotion_text = f"{current_dominant_emotion} ({current_emotions[current_dominant_emotion]:.2f})"
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 显示年龄和性别
            age_text = f"Age: {current_age}"
            gender_text = f"Gender: {current_gender}"
            cv2.putText(frame, age_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, gender_text, (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 可选：显示所有情绪分数
            offset = 0
            for emotion, score in current_emotions.items():
                cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y + h + 20 + offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                offset += 20
        else:
            # 显示提示
            cv2.putText(frame, "未检测到人脸", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示视频帧
        cv2.imshow('Real-time Emotion, Age & Gender Recognition', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

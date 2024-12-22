import cv2
import logging
from EmtionDetection import DeepFace

# 配置日志
logging.basicConfig(
    level=logging.ERROR,  # 仅记录ERROR及以上级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_recognition.log"),  # 将日志输出到文件
        logging.StreamHandler()  # 仅在出现ERROR时在控制台输出
    ]
)

def main():
    # 初始化摄像头（0为默认摄像头）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("无法打开摄像头")
        return

    # 可选：设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    # 初始化分析频率控制
    frame_count = 0
    analyze_every_n_frames = 5  # 每隔5帧进行一次情绪分析
    current_dominant_emotion = "neutral"
    current_emotions = {}
    region = {'x': 0, 'y': 0, 'w': 100, 'h': 100}  # 初始化区域以避免未定义错误

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("无法读取帧")
            break

        try:
            if frame_count % analyze_every_n_frames == 0:
                # DeepFace 分析当前帧
                results = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend='opencv'  # 可根据需要更改
                    # detector_backend='yolov11'  
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

                    # 更新当前情绪
                    current_dominant_emotion = dominant_emotion
                    current_emotions = emotions

                if not face_detected:
                    # 没有检测到任何人脸，保持上一次的结果
                    pass

        except Exception as e:
            logging.error(f"处理帧时出错: {e}")
            # 仅在出现错误时显示提示
            cv2.putText(frame, "检测出错", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 绘制当前情绪信息（如果有）
        if current_emotions:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{current_dominant_emotion} ({current_emotions[current_dominant_emotion]:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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
        cv2.imshow('Real-time Emotion Recognition', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

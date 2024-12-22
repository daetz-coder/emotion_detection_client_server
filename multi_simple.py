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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("无法打开摄像头")
        return

    # 可选：设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    frame_count = 0
    analyze_every_n_frames = 5  # 每隔多少帧进行一次情绪分析

    # 这个列表用来存储每一帧中识别到的人脸信息
    # detected_faces 结构: [{'region':..., 'dominant_emotion':..., 'emotions':...}, ...]
    detected_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("无法读取帧")
            break

        # 每隔 analyze_every_n_frames 帧进行一次情绪识别
        if frame_count % analyze_every_n_frames == 0:
            try:
                # 使用 DeepFace 进行情绪分析
                results = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],            # 仅识别情绪，你也可增加 'age', 'gender', 'race' 等
                    enforce_detection=False,        # 设置为 False 对于低分辨率情况下不强制报错
                    detector_backend='opencv'       # 你可以尝试 'yolov11' 等其它检测器
                )

                # 如果检测到多张人脸，结果会是一个 list；如果只检测到一张，则是 dict
                if isinstance(results, list):
                    faces = results
                else:
                    faces = [results]

                # 存储本帧中所有检测到的人脸
                detected_faces.clear()
                for face in faces:
                    if face.get('region') is not None:
                        detected_faces.append({
                            'region': face['region'],
                            'dominant_emotion': face['dominant_emotion'],
                            'emotions': face['emotion']
                        })

            except Exception as e:
                logging.error(f"处理帧时出错: {e}")

        # 无论是否分析，这里都要把已存储的人脸信息绘制出来
        if detected_faces:
            for face_info in detected_faces:
                region = face_info['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                # 绘制矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 绘制主情绪及得分
                dominant_emotion = face_info['dominant_emotion']
                emotions = face_info['emotions']
                text = f"{dominant_emotion} ({emotions[dominant_emotion]:.2f})"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 也可以把所有情绪分数都打印出来
                offset = 0
                for emo_name, emo_score in emotions.items():
                    cv2.putText(frame, f"{emo_name}: {emo_score:.2f}",
                                (x, y + h + 20 + offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    offset += 20
        else:
            # 未检测到人脸
            cv2.putText(frame, "未检测到人脸", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示视频帧
        cv2.imshow('Real-time Emotion Recognition', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import requests
import json
import threading
import time
import numpy as np

# 配置
SERVER_URL = 'http://10.11.154.66:7777/analyze'  # 服务器地址
SEND_INTERVAL = 0.05            # 发送帧的间隔时间（秒），例如 20 FPS
DISPLAY_WINDOW_NAME = '原始与处理后视频'  # 显示窗口名称
FRAME_RESIZE_WIDTH = 640        # 发送到服务器的帧宽度
FRAME_RESIZE_HEIGHT = 480       # 发送到服务器的帧高度

# 共享变量和锁
latest_frame = None
latest_frame_lock = threading.Lock()
processed_frame = None
processed_frame_lock = threading.Lock()

def send_frame_to_server():
    """持续发送最新帧到服务器并处理响应。"""
    global latest_frame, processed_frame
    session = requests.Session()  # 使用会话对象复用连接
    while True:
        frame_to_send = None
        with latest_frame_lock:
            if latest_frame is not None:
                frame_to_send = latest_frame.copy()

        if frame_to_send is not None:
            try:
                # 调整帧大小以加快传输速度
                resized_frame = cv2.resize(frame_to_send, (FRAME_RESIZE_WIDTH, FRAME_RESIZE_HEIGHT))
                # 将帧编码为 JPEG 格式
                _, img_encoded = cv2.imencode('.jpg', resized_frame)
                # 发送 POST 请求，携带编码后的图像
                response = session.post(
                    SERVER_URL,
                    files={'frame': img_encoded.tobytes()},
                    timeout=5  # 设置合适的超时时间
                )
                response.raise_for_status()  # 对于错误状态码抛出异常

                # 假设服务器返回包含标注数据的 JSON
                result = response.json()

                # 创建一份原始帧的副本用于绘制标注
                annotated_frame = frame_to_send.copy()

                # 绘制边界框和标签
                for annotation in result.get('annotations', []):
                    bbox = annotation.get('bbox', [])
                    label = annotation.get('label', '')
                    if len(bbox) == 4:
                        x, y, w, h = bbox
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # 线程安全地更新处理后的帧
                with processed_frame_lock:
                    processed_frame = annotated_frame

            except requests.exceptions.RequestException as e:
                print(f"发送帧到服务器时出错: {e}")
            except json.JSONDecodeError:
                print("解析服务器响应为 JSON 时出错。")

        # 控制发送间隔
        time.sleep(SEND_INTERVAL)

def capture_frames():
    """捕捉视频帧并更新最新帧。"""
    global latest_frame
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法从摄像头读取帧。")
            break

        # 更新最新帧
        with latest_frame_lock:
            latest_frame = frame.copy()

        # 准备显示内容
        with processed_frame_lock:
            if processed_frame is not None:
                # 调整两帧的尺寸，使其高度相同
                height = min(frame.shape[0], processed_frame.shape[0])
                width = min(frame.shape[1], processed_frame.shape[1])

                original_resized = cv2.resize(frame, (width, height))
                processed_resized = cv2.resize(processed_frame, (width, height))

                # 将原始帧和处理后帧水平拼接
                combined_frame = np.hstack((original_resized, processed_resized))
            else:
                # 如果尚未有处理后的帧，则仅显示原始帧
                combined_frame = frame.copy()

        # 显示拼接后的帧
        cv2.imshow(DISPLAY_WINDOW_NAME, combined_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 启动发送线程
    sender_thread = threading.Thread(target=send_frame_to_server, daemon=True)
    sender_thread.start()

    # 开始捕捉和显示帧
    capture_frames()

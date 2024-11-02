import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import io
import cv2
import json
import threading
import time
import numpy as np

# 配置服务器URL
IMAGE_ANALYSIS_URL = 'http://10.11.154.66:5000/image'
VIDEO_ANALYSIS_URL = 'http://10.11.154.66:5000/video'
REALTIME_ANALYSIS_URL = 'http://10.11.154.66:5000/realtime'

# -------------------- 图像分析功能 --------------------
class ImageAnalysisWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('图像分析')
        self.geometry('850x700')
        self.image_data = None

        # 按钮
        open_button = tk.Button(self, text="打开文件", command=self.open_file)
        open_button.grid(row=0, column=0, padx=10, pady=10)

        download_button = tk.Button(self, text="下载图像", command=self.save_image)
        download_button.grid(row=0, column=1, padx=10, pady=10)

        exit_button = tk.Button(self, text="退出", command=self.destroy)
        exit_button.grid(row=0, column=2, padx=10, pady=10)

        # 图像显示区域
        self.image_label = tk.Label(self)
        self.image_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

    def resize_image(self, pil_image, max_width=800, max_height=600):
        original_width, original_height = pil_image.size
        ratio = min(max_width/original_width, max_height/original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)
        return resized_image

    def send_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(IMAGE_ANALYSIS_URL, files=files)
                if response.headers['Content-Type'] == 'application/json':
                    response_data = response.json()
                    messagebox.showinfo("服务器响应", str(response_data))
                    return None
                elif response.headers['Content-Type'] in ['image/jpeg', 'image/png']:
                    return response.content
                else:
                    messagebox.showerror("错误", "意外的MIME类型: " + response.headers['Content-Type'])
                    return None
        except Exception as e:
            messagebox.showerror("错误", f"发送文件失败: {e}")
            return None

    def download_and_show_image(self, image_data):
        try:
            image_bytes = io.BytesIO(image_data)
            pil_image = Image.open(image_bytes)
            resized_image = self.resize_image(pil_image, max_width=800, max_height=600)
            tk_image = ImageTk.PhotoImage(resized_image)

            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # 避免图片被垃圾回收
            self.image_data = image_data  # 保存图像数据以便下载
        except Exception as e:
            messagebox.showerror("错误", f"加载或显示图像失败: {e}")

    def save_image(self):
        if not self.image_data:
            messagebox.showinfo("信息", "没有图像数据可保存。")
            return
        filetypes = (
            ('PNG 文件', '*.png'),
            ('JPEG 文件', '*.jpg'),
            ('所有文件', '*.*')
        )
        filepath = filedialog.asksaveasfilename(title="保存图像", filetypes=filetypes, defaultextension=filetypes)
        if not filepath:
            return
        try:
            with open(filepath, 'wb') as f:
                f.write(self.image_data)
            messagebox.showinfo("保存图像", "图像保存成功！")
        except Exception as e:
            messagebox.showerror("保存图像", f"保存图像失败: {e}")

    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")])
        if filepath:
            image_data = self.send_file(filepath)
            if image_data:
                self.download_and_show_image(image_data)
            else:
                messagebox.showinfo("信息", "未收到图像数据。")

# -------------------- 视频分析功能 --------------------
class VideoAnalysisWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('视频分析')
        self.geometry('400x150')
        self.video_data = None

        # 按钮
        open_button = tk.Button(self, text="打开视频文件", command=self.open_file)
        open_button.grid(row=0, column=0, padx=10, pady=10)

        download_button = tk.Button(self, text="下载视频", command=self.save_video)
        download_button.grid(row=0, column=1, padx=10, pady=10)

        exit_button = tk.Button(self, text="退出", command=self.destroy)
        exit_button.grid(row=0, column=2, padx=10, pady=10)

    def send_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(VIDEO_ANALYSIS_URL, files=files)
                if response.headers['Content-Type'] == 'application/json':
                    response_data = response.json()
                    messagebox.showinfo("服务器响应", str(response_data))
                    return None
                elif response.headers['Content-Type'] in ['video/mp4', 'video/x-matroska', 'video/avi']:
                    return response.content
                else:
                    messagebox.showerror("错误", "意外的MIME类型: " + response.headers['Content-Type'])
                    return None
        except Exception as e:
            messagebox.showerror("错误", f"发送文件失败: {e}")
            return None

    def save_video(self):
        if not self.video_data:
            messagebox.showinfo("信息", "没有视频数据可保存。")
            return
        filetypes = (
            ('MP4 文件', '*.mp4'),
            ('AVI 文件', '*.avi'),
            ('所有文件', '*.*')
        )
        filepath = filedialog.asksaveasfilename(title="保存视频", filetypes=filetypes, defaultextension=filetypes)
        if not filepath:
            return
        try:
            with open(filepath, 'wb') as f:
                f.write(self.video_data)
            messagebox.showinfo("保存视频", "视频保存成功！")
        except Exception as e:
            messagebox.showerror("保存视频", f"保存视频失败: {e}")

    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4;*.avi;*.mkv"), ("所有文件", "*.*")])
        if filepath:
            video_data = self.send_file(filepath)
            if video_data:
                self.video_data = video_data
                self.save_video()
            else:
                messagebox.showinfo("信息", "未收到视频数据。")

# -------------------- 实时分析功能 --------------------
class RealTimeAnalysis:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("实时分析", "实时分析已经在运行。")
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        messagebox.showinfo("实时分析", "实时分析已启动。")

    def run(self):
        # 配置
        SERVER_URL = REALTIME_ANALYSIS_URL
        SEND_INTERVAL = 0.05  # 发送帧的间隔时间（秒），例如 20 FPS
        DISPLAY_WINDOW_NAME = '原始与处理后视频'  # 显示窗口名称
        FRAME_RESIZE_WIDTH = 640        # 发送到服务器的帧宽度
        FRAME_RESIZE_HEIGHT = 480       # 发送到服务器的帧高度

        # 共享变量和锁
        latest_frame = None
        latest_frame_lock = threading.Lock()
        processed_frame = None
        processed_frame_lock = threading.Lock()

        def send_frame_to_server():
            nonlocal latest_frame, processed_frame
            session = requests.Session()  # 使用会话对象复用连接
            while not self.stop_event.is_set():
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
            nonlocal latest_frame, processed_frame
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("错误: 无法打开摄像头。")
                messagebox.showerror("实时分析", "无法打开摄像头。")
                return

            sender_thread = threading.Thread(target=send_frame_to_server, daemon=True)
            sender_thread.start()

            while not self.stop_event.is_set():
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
                    self.stop()
                    break

            # 释放资源
            cap.release()
            cv2.destroyAllWindows()

        capture_frames()

    def stop(self):
        self.stop_event.set()
        messagebox.showinfo("实时分析", "实时分析已停止。")

# -------------------- 主应用程序 --------------------
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('文件上传与分析工具')
        self.geometry('400x250')

        # 图像分析按钮
        img_button = tk.Button(self, text="图像分析", width=25, height=2, command=self.open_image_analysis)
        img_button.pack(pady=10)

        # 视频分析按钮
        video_button = tk.Button(self, text="视频分析", width=25, height=2, command=self.open_video_analysis)
        video_button.pack(pady=10)

        # 实时分析按钮
        realtime_button = tk.Button(self, text="实时分析", width=25, height=2, command=self.start_realtime_analysis)
        realtime_button.pack(pady=10)

        # 退出按钮
        exit_button = tk.Button(self, text="退出", width=25, height=2, command=self.quit)
        exit_button.pack(pady=10)

        # 实时分析实例
        self.realtime_analysis = RealTimeAnalysis()

    def open_image_analysis(self):
        ImageAnalysisWindow(self)

    def open_video_analysis(self):
        VideoAnalysisWindow(self)

    def start_realtime_analysis(self):
        self.realtime_analysis.start()

# -------------------- 运行应用程序 --------------------
if __name__ == '__main__':
    app = Application()
    app.mainloop()

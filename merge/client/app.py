import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests
import io
import cv2
import json
import threading
import time
import numpy as np

# Server URLs Configuration
IMAGE_ANALYSIS_URL = 'http://10.11.154.66:5000/analyze_image'
VIDEO_ANALYSIS_URL = 'http://10.11.154.66:5000/analyze_video'
REALTIME_ANALYSIS_URL = 'http://10.11.154.66:5000/analyze_online'

# -------------------- Image Analysis Functionality --------------------
class ImageAnalysisWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Image Analysis')
        self.geometry('850x700')
        self.image_data = None

        # Use ttk style
        self.style = ttk.Style(self)
        self.style.theme_use('clam')  # You can choose other themes like 'default', 'alt', 'classic'

        # Button Frame
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        # Buttons
        open_button = ttk.Button(button_frame, text="Open File", command=self.open_file)
        open_button.grid(row=0, column=0, padx=10, pady=10)

        download_button = ttk.Button(button_frame, text="Download Image", command=self.save_image)
        download_button.grid(row=0, column=1, padx=10, pady=10)

        exit_button = ttk.Button(button_frame, text="Exit", command=self.destroy)
        exit_button.grid(row=0, column=2, padx=10, pady=10)

        # Image Display Area
        self.image_label = ttk.Label(self)
        self.image_label.pack(padx=10, pady=10, expand=True)

    def resize_image(self, pil_image, max_width=800, max_height=600):
        original_width, original_height = pil_image.size
        ratio = min(max_width/original_width, max_height/original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    def send_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(IMAGE_ANALYSIS_URL, files=files)
                content_type = response.headers.get('Content-Type', '')

                if content_type.startswith('application/json'):
                    response_data = response.json()
                    if 'error' in response_data:
                        messagebox.showerror("Server Error", response_data['error'])
                    else:
                        messagebox.showinfo("Server Response", str(response_data))
                    return None
                elif content_type.startswith('image/'):
                    return response.content
                else:
                    messagebox.showerror("Error", "Unexpected MIME type: " + content_type)
                    return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send file: {e}")
            return None

    def download_and_show_image(self, image_data):
        try:
            image_bytes = io.BytesIO(image_data)
            pil_image = Image.open(image_bytes)
            resized_image = self.resize_image(pil_image, max_width=800, max_height=600)
            tk_image = ImageTk.PhotoImage(resized_image)

            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # Prevent garbage collection
            self.image_data = image_data  # Save image data for download
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or display image: {e}")

    def save_image(self):
        if not self.image_data:
            messagebox.showinfo("Info", "No image data to save.")
            return
        filetypes = (
            ('PNG Files', '*.png'),
            ('JPEG Files', '*.jpg'),
            ('All Files', '*.*')
        )
        filepath = filedialog.asksaveasfilename(title="Save Image", filetypes=filetypes, defaultextension='.png')
        if not filepath:
            return
        try:
            with open(filepath, 'wb') as f:
                f.write(self.image_data)
            messagebox.showinfo("Save Image", "Image saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Image", f"Failed to save image: {e}")

    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")])
        if filepath:
            image_data = self.send_file(filepath)
            if image_data:
                self.download_and_show_image(image_data)
            else:
                messagebox.showinfo("Info", "No image data received.")

# -------------------- Video Analysis Functionality --------------------
class VideoAnalysisWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title('Video Analysis')
        self.geometry('500x200')
        self.video_data = None

        # Use ttk style
        self.style = ttk.Style(self)
        self.style.theme_use('clam')  # You can choose other themes

        # Button Frame
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=20)

        # Buttons
        open_button = ttk.Button(button_frame, text="Open Video File", command=self.open_file)
        open_button.grid(row=0, column=0, padx=10, pady=10)

        download_button = ttk.Button(button_frame, text="Download Video", command=self.save_video)
        download_button.grid(row=0, column=1, padx=10, pady=10)

        exit_button = ttk.Button(button_frame, text="Exit", command=self.destroy)
        exit_button.grid(row=0, column=2, padx=10, pady=10)

    def send_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                files = {'file': f}
                response = requests.post(VIDEO_ANALYSIS_URL, files=files)
                content_type = response.headers.get('Content-Type', '')

                if content_type.startswith('application/json'):
                    response_data = response.json()
                    if 'error' in response_data:
                        messagebox.showerror("Server Error", response_data['error'])
                    else:
                        messagebox.showinfo("Server Response", str(response_data))
                    return None
                elif content_type.startswith('video/'):
                    return response.content
                else:
                    messagebox.showerror("Error", "Unexpected MIME type: " + content_type)
                    return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send file: {e}")
            return None

    def save_video(self):
        if not self.video_data:
            messagebox.showinfo("Info", "No video data to save.")
            return
        filetypes = (
            ('MP4 Files', '*.mp4'),
            ('AVI Files', '*.avi'),
            ('All Files', '*.*')
        )
        filepath = filedialog.asksaveasfilename(title="Save Video", filetypes=filetypes, defaultextension='.mp4')
        if not filepath:
            return
        try:
            with open(filepath, 'wb') as f:
                f.write(self.video_data)
            messagebox.showinfo("Save Video", "Video saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Video", f"Failed to save video: {e}")

    def open_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv"), ("All Files", "*.*")])
        if filepath:
            video_data = self.send_file(filepath)
            if video_data:
                self.video_data = video_data
                self.save_video()
            else:
                messagebox.showinfo("Info", "No video data received.")

# -------------------- Real-Time Analysis Functionality --------------------
class RealTimeAnalysis:
    def __init__(self, master=None):
        self.stop_event = threading.Event()
        self.thread = None
        self.master = master
        self.control_window = None

    def start(self):
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("Real-Time Analysis", "Real-time analysis is already running.")
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        self.create_control_window()
        messagebox.showinfo("Real-Time Analysis", "Real-time analysis started.")

    def create_control_window(self):
        self.control_window = tk.Toplevel(self.master)
        self.control_window.title("Real-Time Analysis Control")
        self.control_window.geometry("300x100")

        # Use ttk style
        self.style = ttk.Style(self.control_window)
        self.style.theme_use('clam')  # You can choose other themes

        # Exit Button
        exit_button = ttk.Button(self.control_window, text="Exit", command=self.stop)
        exit_button.pack(pady=20)

        # Handle window closing
        self.control_window.protocol("WM_DELETE_WINDOW", self.stop)

    def run(self):
        # Configuration
        SERVER_URL = REALTIME_ANALYSIS_URL
        SEND_INTERVAL = 0.05  # Frame send interval in seconds (e.g., 20 FPS)
        DISPLAY_WINDOW_NAME = 'Original and Processed Video'  # Display window name
        FRAME_RESIZE_WIDTH = 640        # Width of frame sent to server
        FRAME_RESIZE_HEIGHT = 480       # Height of frame sent to server

        # Shared variables and locks
        latest_frame = None
        latest_frame_lock = threading.Lock()
        processed_frame = None
        processed_frame_lock = threading.Lock()

        def send_frame_to_server():
            nonlocal latest_frame, processed_frame
            session = requests.Session()  # Use session to reuse connections
            while not self.stop_event.is_set():
                frame_to_send = None
                with latest_frame_lock:
                    if latest_frame is not None:
                        frame_to_send = latest_frame.copy()

                if frame_to_send is not None:
                    try:
                        # Resize frame to speed up transmission
                        resized_frame = cv2.resize(frame_to_send, (FRAME_RESIZE_WIDTH, FRAME_RESIZE_HEIGHT))
                        # Encode frame as JPEG
                        _, img_encoded = cv2.imencode('.jpg', resized_frame)
                        # Send POST request with encoded image
                        response = session.post(
                            SERVER_URL,
                            files={'frame': img_encoded.tobytes()},
                            timeout=5  # Set appropriate timeout
                        )
                        response.raise_for_status()  # Raise exception for HTTP errors

                        # Assume server returns JSON with annotation data
                        result = response.json()

                        # Create a copy of the original frame for annotation
                        annotated_frame = frame_to_send.copy()

                        # Draw bounding boxes and labels
                        for annotation in result.get('annotations', []):
                            bbox = annotation.get('bbox', [])
                            label = annotation.get('label', '')
                            if len(bbox) == 4:
                                x, y, w, h = bbox
                                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, label, (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                        # Thread-safe update of processed frame
                        with processed_frame_lock:
                            processed_frame = annotated_frame

                    except requests.exceptions.RequestException as e:
                        print(f"Error sending frame to server: {e}")
                    except json.JSONDecodeError:
                        print("Error decoding server response as JSON.")

                # Control send interval
                time.sleep(SEND_INTERVAL)

        def capture_frames():
            nonlocal latest_frame, processed_frame
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Error: Unable to open camera.")
                messagebox.showerror("Real-Time Analysis", "Unable to open camera.")
                self.stop()
                return

            sender_thread = threading.Thread(target=send_frame_to_server, daemon=True)
            sender_thread.start()

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Unable to read frame from camera.")
                    break

                # Update latest frame
                with latest_frame_lock:
                    latest_frame = frame.copy()

                # Prepare display content
                with processed_frame_lock:
                    if processed_frame is not None:
                        # Resize both frames to have the same height
                        height = min(frame.shape[0], processed_frame.shape[0])
                        width = min(frame.shape[1], processed_frame.shape[1])

                        original_resized = cv2.resize(frame, (width, height))
                        processed_resized = cv2.resize(processed_frame, (width, height))

                        # Concatenate original and processed frames horizontally
                        combined_frame = np.hstack((original_resized, processed_resized))
                    else:
                        # If no processed frame yet, display only original frame
                        combined_frame = frame.copy()

                # Display concatenated frame
                cv2.imshow(DISPLAY_WINDOW_NAME, combined_frame)

                # Exit if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

            # Release resources
            cap.release()
            cv2.destroyAllWindows()

        capture_frames()

    def stop(self):
        self.stop_event.set()
        if self.control_window:
            self.control_window.destroy()
        messagebox.showinfo("Real-Time Analysis", "Real-time analysis stopped.")

# -------------------- Main Application --------------------
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('File Upload and Analysis Tool')
        self.geometry('500x300')

        # Use ttk style
        self.style = ttk.Style(self)
        self.style.theme_use('clam')  # You can choose other themes

        # Main Frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(expand=True, fill='both')

        # Image Analysis Button
        img_button = ttk.Button(main_frame, text="Image Analysis", width=30, command=self.open_image_analysis)
        img_button.pack(pady=10)

        # Video Analysis Button
        video_button = ttk.Button(main_frame, text="Video Analysis", width=30, command=self.open_video_analysis)
        video_button.pack(pady=10)

        # Real-Time Analysis Button
        realtime_button = ttk.Button(main_frame, text="Real-Time Analysis", width=30, command=self.start_realtime_analysis)
        realtime_button.pack(pady=10)

        # Exit Button
        exit_button = ttk.Button(main_frame, text="Exit", width=30, command=self.quit)
        exit_button.pack(pady=10)

        # Real-Time Analysis Instance
        self.realtime_analysis = RealTimeAnalysis(self)

    def open_image_analysis(self):
        ImageAnalysisWindow(self)

    def open_video_analysis(self):
        VideoAnalysisWindow(self)

    def start_realtime_analysis(self):
        self.realtime_analysis.start()

# -------------------- Run Application --------------------
if __name__ == '__main__':
    app = Application()
    app.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox
import requests
import io
import cv2
from PIL import Image, ImageTk

def send_file(filepath):
    url = 'http://10.11.154.66:5000/analyze'
    try:
        with open(filepath, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
            if response.headers['Content-Type'] == 'application/json':
                response_data = response.json()
                messagebox.showinfo("Server Response", str(response_data))
                return None
            elif response.headers['Content-Type'] == 'video/mp4':
                return response.content
            else:
                messagebox.showerror("Error", "Unexpected MIME type: " + response.headers['Content-Type'])
                return None
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send file: {e}")
        return None

def save_video(video_data):
    filetypes = (
        ('MP4 files', '*.mp4'),
        ('All files', '*.*')
    )
    filepath = filedialog.asksaveasfilename(title="Save Video", filetypes=filetypes, defaultextension=filetypes)
    if not filepath:
        return
    try:
        with open(filepath, 'wb') as f:
            f.write(video_data)
        messagebox.showinfo("Save Video", "Video saved successfully!")
    except Exception as e:
        messagebox.showerror("Save Video", f"Failed to save video: {e}")

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All video files", "*.*")])
    if filepath:
        video_data = send_file(filepath)
        if video_data:
            save_video(video_data)
        else:
            messagebox.showinfo("Info", "No video data received.")

# Setup the main window
window = tk.Tk()
window.title('File Upload and Analyze Tool')

# Buttons for user interaction
open_button = tk.Button(window, text="Open File", command=open_file)
open_button.grid(row=0, column=0, padx=10, pady=10)

download_button = tk.Button(window, text="Download Video", command=lambda: save_video(window.video_data if hasattr(window, 'video_data') else None))
download_button.grid(row=0, column=1, padx=10, pady=10)

exit_button = tk.Button(window, text="Exit", command=window.quit)
exit_button.grid(row=0, column=2, padx=10, pady=10)

window.mainloop()

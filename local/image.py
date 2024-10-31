import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import io

def resize_image(pil_image, max_width=800, max_height=600):
    original_width, original_height = pil_image.size
    ratio = min(max_width/original_width, max_height/original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

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
            elif response.headers['Content-Type'] in ['image/jpeg', 'image/png']:
                return response.content
            else:
                messagebox.showerror("Error", "Unexpected MIME type: " + response.headers['Content-Type'])
                return None
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send file: {e}")
        return None

def download_and_show_image(image_data):
    try:
        image_bytes = io.BytesIO(image_data)
        pil_image = Image.open(image_bytes)
        resized_image = resize_image(pil_image, max_width=800, max_height=600)
        tk_image = ImageTk.PhotoImage(resized_image)

        image_label = tk.Label(window, image=tk_image)
        image_label.image = tk_image  # 避免图片被垃圾收集
        image_label.grid(row=2, column=0, columnspan=3)
        window.image_data = image_data  # 保存图像数据以便下载
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load or show image: {e}")

def save_image(image_data):
    filetypes = (
        ('PNG files', '*.png'),
        ('JPEG files', '*.jpg'),
        ('All files', '*.*')
    )
    filepath = filedialog.asksaveasfilename(title="Save Image", filetypes=filetypes, defaultextension=filetypes)
    if not filepath:
        return
    try:
        with open(filepath, 'wb') as f:
            f.write(image_data)
        messagebox.showinfo("Save Image", "Image saved successfully!")
    except Exception as e:
        messagebox.showerror("Save Image", f"Failed to save image: {e}")

def open_file():
    filepath = filedialog.askopenfilename()
    if filepath:
        image_data = send_file(filepath)
        if image_data:
            download_and_show_image(image_data)
        else:
            messagebox.showinfo("Info", "No image data received.")

# Setup the main window
window = tk.Tk()
window.title('File Upload and Analyze Tool')

# Buttons for user interaction
open_button = tk.Button(window, text="Open File", command=open_file)
open_button.grid(row=0, column=0, padx=10, pady=10)

download_button = tk.Button(window, text="Download Image", command=lambda: save_image(window.image_data if hasattr(window, 'image_data') else None))
download_button.grid(row=0, column=1, padx=10, pady=10)

exit_button = tk.Button(window, text="Exit", command=window.quit)
exit_button.grid(row=0, column=2, padx=10, pady=10)

window.mainloop()

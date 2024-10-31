# server.py
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from image import analyze_image  # 引入 image.py 中的处理函数

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        save_path = 'uploads'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filepath = os.path.join(save_path, filename)
        file.save(filepath)

        # 指定处理后图像的保存路径
        processed_filepath = os.path.join(save_path, f"processed_{filename}")
        result_path = analyze_image(filepath, processed_filepath)

        if result_path:
            return send_from_directory('uploads', f"processed_{filename}")
        else:
            return jsonify({"error": "Failed to process image"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

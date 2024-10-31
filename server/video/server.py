from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import os
from video import analyze_video

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

        processed_filepath = os.path.join(save_path, f"processed_{filename}")
        result_path = analyze_video(filepath, processed_filepath)

        if result_path:
            return send_file(result_path, mimetype='video/mp4')
        else:
            return jsonify({"error": "Failed to process video"}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

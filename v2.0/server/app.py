from recognition import analyze_image, analyze_video, analyze_online
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)

# 配置上传文件大小限制（例如，最大50MB）
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# 创建存储上传文件的目录
save_path = 'uploads'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 允许的文件扩展名
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# -------------------- 图像分析功能 --------------------
@app.route('/analyze_image', methods=['POST'])
def analyze_image_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(save_path, unique_filename)
        file.save(filepath)

        processed_filename = f"processed_{unique_filename}"
        processed_filepath = os.path.join(save_path, processed_filename)
        result_path = analyze_image(filepath, processed_filepath)

        if result_path:
            return send_from_directory('uploads', processed_filename)
        else:
            return jsonify({"error": "Failed to process image"}), 500
    else:
        return jsonify({"error": "Unsupported file type"}), 400

# -------------------- 视频分析功能 --------------------
@app.route('/analyze_video', methods=['POST'])
def analyze_video_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(save_path, unique_filename)
        file.save(filepath)

        processed_filename = f"processed_{unique_filename}"
        processed_filepath = os.path.join(save_path, processed_filename)
        result_path = analyze_video(filepath, processed_filepath)

        if result_path:
            return send_file(result_path, mimetype='video/mp4')
        else:
            return jsonify({"error": "Failed to process video"}), 500
    else:
        return jsonify({"error": "Unsupported file type"}), 400

# -------------------- 实时分析功能 --------------------
@app.route('/analyze_online', methods=['POST'])
def analyze_online_route():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame provided"}), 400
    file = request.files['frame'].read()
    
    try:
        annotations = analyze_online(file)
        return jsonify(annotations)  # 在这里使用 jsonify
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- 文件访问功能 --------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

# -------------------- IP和端口 --------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
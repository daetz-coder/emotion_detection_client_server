from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['frame'].read()  # 从请求中读取视频帧
        nparr = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 使用 DeepFace 进行实时分析
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        annotations = []

        # DeepFace.analyze 可能返回多个结果（如果有多张人脸）
        if isinstance(results, list):
            for result in results:
                region = result.get('region', {})
                emotions = result.get('emotion', {})
                dominant_emotion = result.get('dominant_emotion', 'neutral')

                bbox = [
                    region.get('x', 0),
                    region.get('y', 0),
                    region.get('w', 0),
                    region.get('h', 0)
                ]

                annotations.append({
                    "bbox": bbox,
                    "label": dominant_emotion
                })
        else:
            # 单张人脸的情况
            region = results.get('region', {})
            emotions = results.get('emotion', {})
            dominant_emotion = results.get('dominant_emotion', 'neutral')

            bbox = [
                region.get('x', 0),
                region.get('y', 0),
                region.get('w', 0),
                region.get('h', 0)
            ]

            annotations.append({
                "bbox": bbox,
                "label": dominant_emotion
            })

        return jsonify({"annotations": annotations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True)

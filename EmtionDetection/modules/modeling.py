# built-in dependencies
# useful
from typing import Any

# project dependencies
from EmtionDetection.models.facial_recognition import (
    VGGFace
)
from EmtionDetection.models.face_detection import (
    OpenCv,
    Yolo as YoloFaceDetector
)
from EmtionDetection.models.demography import Emotion

from EmtionDetection.models.spoofing import FasNet


def build_model(task: str, model_name: str) -> Any:
    """
    This function loads a pre-trained models as singletonish way
    Parameters:
        task (str): facial_recognition, facial_attribute, face_detector, spoofing
        model_name (str): model identifier
            - VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
                ArcFace, SFace and GhostFaceNet for face recognition
            - Age, Gender, Emotion, Race for facial attributes
            - opencv, mtcnn, ssd, dlib, retinaface, mediapipe, yolov8, 'yolov11n',
                'yolov11s', 'yolov11m', yunet, fastmtcnn or centerface for face detectors
            - Fasnet for spoofing
    Returns:
            built model class
    """

    # singleton design pattern
    global cached_models

    models = {
        "facial_recognition": {
            "VGG-Face": VGGFace.VggFaceClient
        },
        "spoofing": {
            "Fasnet": FasNet.Fasnet,
        },
        "facial_attribute": {
            "Emotion": Emotion.EmotionClient
        },
        "face_detector": {
            "opencv": OpenCv.OpenCvClient,
            "yolov8": YoloFaceDetector.YoloDetectorClientV8n,
            "yolov11n": YoloFaceDetector.YoloDetectorClientV11n,
            "yolov11s": YoloFaceDetector.YoloDetectorClientV11s,
            "yolov11m": YoloFaceDetector.YoloDetectorClientV11m
        },
    }

    if models.get(task) is None:
        raise ValueError(f"unimplemented task - {task}")

    if not "cached_models" in globals():
        cached_models = {current_task: {} for current_task in models.keys()}

    if cached_models[task].get(model_name) is None:
        model = models[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return cached_models[task][model_name]

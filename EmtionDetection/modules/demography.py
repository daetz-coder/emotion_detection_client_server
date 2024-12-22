# useful
# built-in dependencies
from typing import Any, Dict, List, Union

# 3rd party dependencies
import numpy as np
from tqdm import tqdm

# project dependencies
from EmtionDetection.modules import modeling, detection, preprocessing
from EmtionDetection.models.demography import Emotion


def analyze(
    img_path: Union[str, np.ndarray],
    actions: Union[tuple, list] = ("emotion"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:


    # if actions is passed as tuple with single item, interestingly it becomes str here
    if isinstance(actions, str):
        actions = (actions,)

    # check if actions is not an iterable or empty.
    if not hasattr(actions, "__getitem__") or not actions:
        raise ValueError("`actions` must be a list of strings.")

    actions = list(actions)

    # For each action, check if it is valid
    for action in actions:
        if action not in ("emotion"):
            raise ValueError(
                f"Invalid action passed ({repr(action)})). "
                "Valid actions are `emotion`."
            )
    # ---------------------------------
    resp_objects = []

    img_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        grayscale=False,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in the given image.")

        img_content = img_obj["face"]
        img_region = img_obj["facial_area"]
        img_confidence = img_obj["confidence"]
        if img_content.shape[0] == 0 or img_content.shape[1] == 0:
            continue

        # rgb to bgr
        img_content = img_content[:, :, ::-1]

        # resize input image
        img_content = preprocessing.resize_image(img=img_content, target_size=(224, 224))

        obj = {}
        # facial attribute analysis
        pbar = tqdm(
            range(0, len(actions)),
            desc="Finding actions",
            disable=silent if len(actions) > 1 else True,
        )
        for index in pbar:
            action = actions[index]
            pbar.set_description(f"Action: {action}")

            if action == "emotion":
                emotion_predictions = modeling.build_model(
                    task="facial_attribute", model_name="Emotion"
                ).predict(img_content)
                sum_of_predictions = emotion_predictions.sum()

                obj["emotion"] = {}
                for i, emotion_label in enumerate(Emotion.labels):
                    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                    obj["emotion"][emotion_label] = emotion_prediction

                obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

            # -----------------------------
            # mention facial areas
            obj["region"] = img_region
            # include image confidence
            obj["face_confidence"] = img_confidence

        resp_objects.append(obj)

    return resp_objects

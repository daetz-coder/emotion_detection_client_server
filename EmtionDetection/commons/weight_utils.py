# built-in dependencies
import os
from typing import Optional
import zipfile
import bz2

# 3rd party dependencies
import gdown

# project dependencies
from EmtionDetection.commons import folder_utils, package_utils
from EmtionDetection.commons.logger import Logger


tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Sequential
else:
    from tensorflow.keras.models import Sequential

logger = Logger()

# pylint: disable=line-too-long, use-maxsplit-arg

ALLOWED_COMPRESS_TYPES = ["zip", "bz2"]


def download_weights_if_necessary(
    file_name: str, source_url: str, compress_type: Optional[str] = None
) -> str:
    """
    Download the weights of a pre-trained model from external source if not downloaded yet.
    Args:
        file_name (str): target file name with extension
        source_url (url): source url to be downloaded
        compress_type (optional str): compress type e.g. zip or bz2
    Returns
        target_file (str): exact path for the target file
    """
    home = folder_utils.get_deepface_home()

    target_file = os.path.normpath(os.path.join(home, ".deepface/weights", file_name))

    if os.path.isfile(target_file):
        logger.debug(f"{file_name} is already available at {target_file}")
        return target_file

    if compress_type is not None and compress_type not in ALLOWED_COMPRESS_TYPES:
        raise ValueError(f"unimplemented compress type - {compress_type}")

    try:
        logger.info(f"🔗 {file_name} will be downloaded from {source_url} to {target_file}...")

        if compress_type is None:
            gdown.download(source_url, target_file, quiet=False)
        elif compress_type is not None and compress_type in ALLOWED_COMPRESS_TYPES:
            gdown.download(source_url, f"{target_file}.{compress_type}", quiet=False)

    except Exception as err:
        raise ValueError(
            f"⛓️‍💥 An exception occurred while downloading {file_name} from {source_url}. "
            f"Consider downloading it manually to {target_file}."
        ) from err

    # uncompress downloaded file
    if compress_type == "zip":
        with zipfile.ZipFile(f"{target_file}.zip", "r") as zip_ref:
            zip_ref.extractall(os.path.join(home, ".deepface/weights"))
            logger.info(f"{target_file}.zip unzipped")
    elif compress_type == "bz2":
        bz2file = bz2.BZ2File(f"{target_file}.bz2")
        data = bz2file.read()
        with open(target_file, "wb") as f:
            f.write(data)
        logger.info(f"{target_file}.bz2 unzipped")

    return target_file


def load_model_weights(model: Sequential, weight_file: str) -> Sequential:
    """
    Load pre-trained weights for a given model
    Args:
        model (keras.models.Sequential): pre-built model
        weight_file (str): exact path of pre-trained weights
    Returns:
        model (keras.models.Sequential): pre-built model with
            updated weights
    """
    try:
        model.load_weights(weight_file)
    except Exception as err:
        raise ValueError(
            f"An exception occurred while loading the pre-trained weights from {weight_file}."
            "This might have happened due to an interruption during the download."
            "You may want to delete it and allow DeepFace to download it again during the next run."
            "If the issue persists, consider downloading the file directly from the source "
            "and copying it to the target folder."
        ) from err
    return model


def download_all_models_in_one_shot() -> None:
    """
    Download all model weights in one shot
    """

    # import model weights from module here to avoid circular import issue
    from EmtionDetection.models.facial_recognition.VGGFace import WEIGHTS_URL as VGGFACE_WEIGHTS
    from EmtionDetection.models.demography.Emotion import WEIGHTS_URL as EMOTION_WEIGHTS
    from EmtionDetection.models.spoofing.FasNet import (
        FIRST_WEIGHTS_URL as FASNET_1ST_WEIGHTS,
        SECOND_WEIGHTS_URL as FASNET_2ND_WEIGHTS,
    )
    from EmtionDetection.models.face_detection.Yolo import (
        WEIGHT_URLS as YOLO_WEIGHTS,
        WEIGHT_NAMES as YOLO_WEIGHT_NAMES,
        YoloModel
    )

    WEIGHTS = [
        # facial recognition
        VGGFACE_WEIGHTS,
        # demography
        EMOTION_WEIGHTS,
        # spoofing
        FASNET_1ST_WEIGHTS,
        FASNET_2ND_WEIGHTS,
        # face detection
        SSD_MODEL,
        SSD_WEIGHTS,
        {
            "filename": YOLO_WEIGHT_NAMES[YoloModel.V8N.value],
            "url": YOLO_WEIGHTS[YoloModel.V8N.value],
        },
        {
            "filename": YOLO_WEIGHT_NAMES[YoloModel.V11N.value],
            "url": YOLO_WEIGHTS[YoloModel.V11N.value],
        },
        {
            "filename": YOLO_WEIGHT_NAMES[YoloModel.V11S.value],
            "url": YOLO_WEIGHTS[YoloModel.V11S.value],
        },
        {
            "filename": YOLO_WEIGHT_NAMES[YoloModel.V11M.value],
            "url": YOLO_WEIGHTS[YoloModel.V11M.value],
        }
    ]

    for i in WEIGHTS:
        if isinstance(i, str):
            url = i
            filename = i.split("/")[-1]
            compress_type = None
            # if compressed file will be downloaded, get rid of its extension
            if filename.endswith(tuple(ALLOWED_COMPRESS_TYPES)):
                for ext in ALLOWED_COMPRESS_TYPES:
                    compress_type = ext
                    if filename.endswith(f".{ext}"):
                        filename = filename[: -(len(ext) + 1)]
                        break
        elif isinstance(i, dict):
            filename = i["filename"]
            url = i["url"]
        else:
            raise ValueError("unimplemented scenario")
        logger.info(
            f"Downloading {url} to ~/.deepface/weights/{filename} with {compress_type} compression"
        )
        download_weights_if_necessary(
            file_name=filename, source_url=url, compress_type=compress_type
        )

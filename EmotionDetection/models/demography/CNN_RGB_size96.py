# 3rd party dependencies
import numpy as np
import cv2
import os

# project dependencies
from EmotionDetection.commons import package_utils, weight_utils
from EmotionDetection.models.Demography import Demography
from EmotionDetection.commons.logger import Logger
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, MaxPool2D
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam

# 依赖配置
tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
        BatchNormalization,
        MaxPool2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam

# 情绪标签
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 初始化日志记录器
logger = Logger()

# 权重文件的本地路径
WEIGHTS_FILE_PATH = "./pth/best_model_CNN_RGB_size96.h5"

# pylint: disable=line-too-long, disable=too-few-public-methods

class EmotionClient(Demography):
    """
    情绪识别模型类
    """

    def __init__(self):
        # 加载模型
        self.model = load_model()
        self.model_name = "Emotion"

    def predict(self, img: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (96, 96))
        # print(f"image_resized: {image_resized.shape}")
        # 归一化图像数据（假设模型在 0-1 范围内训练）
        image = np.expand_dims(image_resized, axis=0)
        # print(f"image_batch: {image_batch.shape}")
        # image_batch = image_batch.astype('float32') / 255.0

        # 进行预测，避免使用 `model.predict` 以减少内存问题
        emotion_predictions = self.model(image, training=False).numpy()[0, :]

        return emotion_predictions


def load_model() -> Sequential:
    """
    构建情绪识别模型，加载本地权重文件
    """
    model = Sequential()
    num_classes = 7
    input_shape = (96, 96, 3)  # 输入图像尺寸为224x224，RGB三通道

    model = Sequential()

    model.add(Conv2D(32, (3,3), activation="selu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3,3), activation="selu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation="selu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3,3), activation="selu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation="selu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3,3), activation="selu"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation="selu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.6))

    model.add(Flatten())
    model.add(Dense(128, activation='selu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # 输出层
    model.add(Dense(num_classes, activation="softmax"))

    # # 打印模型摘要
    # model.summary()

    # 加载本地权重文件
    if os.path.exists(WEIGHTS_FILE_PATH):
        model.load_weights(WEIGHTS_FILE_PATH)
        logger.info(f"已加载权重文件：{WEIGHTS_FILE_PATH}")
    else:
        logger.error(f"未找到权重文件：{WEIGHTS_FILE_PATH}")
        raise FileNotFoundError(f"未找到权重文件：{WEIGHTS_FILE_PATH}")

    return model

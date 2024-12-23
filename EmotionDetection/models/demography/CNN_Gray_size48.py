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
WEIGHTS_FILE_PATH = "./pth/best_model_CNN_Gray_size48.h5"

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
        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = np.expand_dims(img_gray, axis=0)

        # model.predict causes memory issue when it is called in a for loop
        # emotion_predictions = self.model.predict(img_gray, verbose=0)[0, :]
        emotion_predictions = self.model(img_gray, training=False).numpy()[0, :]

        return emotion_predictions


def load_model() -> Sequential:
    """
    构建情绪识别模型，加载本地权重文件
    """
    # 定义模型架构
    model = Sequential()
    input_shape = (48, 48, 1)  # 输入图像尺寸

    # 1st convolution layer
    model.add(Conv2D(32, (3,3), activation="selu", input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # 2nd convolution layer
    model.add(Conv2D(64, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    # 3rd convolution layer
    model.add(Conv2D(128, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    # 4th convolution layer
    model.add(Conv2D(256, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.6))

    # Flatten layer
    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(128, activation='selu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer with softmax activation
    model.add(Dense(7, activation='softmax'))

    # # 编译模型
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

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

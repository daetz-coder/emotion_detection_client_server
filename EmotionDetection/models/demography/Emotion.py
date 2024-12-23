# 3rd party dependencies
import numpy as np
import cv2
import os
import tensorflow as tf
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
        MaxPool2D,
        GlobalAveragePooling2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam

# 情绪标签
labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# 初始化日志记录器
logger = Logger()

# 权重文件的本地路径
WEIGHTS_FILE_PATH = "./pth/best_model_EffectiveV2_RGB_size224.h5"

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
        image_resized = cv2.resize(image_rgb, (224, 224))
        # print(f"image_resized: {image_resized.shape}")
        # 归一化图像数据（假设模型在 0-1 范围内训练）
        image = np.expand_dims(image_resized, axis=0)
        # print(f"image_batch: {image_batch.shape}")
        # 进行预测，避免使用 `model.predict` 以减少内存问题
        emotion_predictions = self.model(image, training=False).numpy()[0, :]

        return emotion_predictions


def load_model() -> Sequential:
    """
    构建情绪识别模型，加载本地权重文件
    """
    # 加载预训练模型（不包括顶层）
    base_model = tf.keras.applications.ConvNeXtTiny(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # 冻结基础模型的层，防止它们在推理时被训练
    base_model.trainable = True  # 可以取消冻结，若你希望在推理过程中做微调
    fine_tune_at = len(base_model.layers) - 20  # 选择冻结的层
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # 创建新模型
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())  # 使用全局平均池化代替Flatten
    model.add(BatchNormalization())  # 添加批量归一化层
    model.add(Dense(128, activation='relu'))  # 增加模型容量
    model.add(Dropout(0.3))  # Dropout层，减少过拟合
    model.add(Dense(7, activation='softmax'))  # 7个类别的输出层

    # 打印模型结构
    model.summary()

    # 加载预训练权重
    if os.path.exists(WEIGHTS_FILE_PATH):
        model.load_weights(WEIGHTS_FILE_PATH)
        logger.info(f"已加载权重文件：{WEIGHTS_FILE_PATH}")
    else:
        logger.error(f"未找到权重文件：{WEIGHTS_FILE_PATH}")
        raise FileNotFoundError(f"未找到权重文件：{WEIGHTS_FILE_PATH}")

    # 返回模型，不进行编译
    return model

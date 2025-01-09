# 实时面部情绪识别

说明：本项目重点在于情绪识别，至于人脸检测使用opencv内置的 **Haar 特征级联分类器**（Haar Cascade Classifier）来进行人脸检测



>完整的代码见：[daetz-coder/emotion_detection_client_server: Real-Time Facial EmotionDetection With Server and Local (github.com)](https://github.com/daetz-coder/emotion_detection_client_server) 或者[emotion_detection_client_server: Real-Time Facial EmotionDetection (gitee.com)](https://gitee.com/daetz_0/emotion_detection_client_server)
>
>权重和数据集见：[Release v1.0 · daetz-coder/emotion_detection_client_server (github.com)](https://github.com/daetz-coder/emotion_detection_client_server/releases/tag/model-weight-dataset-upload)
>
>框架部分参考了[serengil/deepface: A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python (github.com)](https://github.com/serengil/deepface)



## 简介

本项目旨在实现实时面部情绪识别。其核心流程首先使用 OpenCV 内置的 Haar 级联分类器进行人脸检测，再通过多种卷积神经网络模型对提取的人脸进行情绪分类识别。项目提供了多种模型架构选择，包括基于灰度图（48×48）、RGB 图像（96×96、224×224）的自定义卷积网络，以及基于预训练的 EfficientNetV2 模型，并在七种情绪（如快乐、愤怒、悲伤、惊讶等）上进行了训练和微调。为弥补现有数据集中亚洲人脸数据较少的问题，项目团队还自行收集了包含亚洲人面部表情的数据集用于进一步优化模型。通过对比不同模型的训练效果、参数规模和运算复杂度，用户可根据实时性要求选用合适的模型。同时，项目开放了完整的代码、训练记录、模型权重及使用说明，使开发者能够方便地在本地或服务器环境中部署和应用这一实时情绪检测系统。尽管该实时面部情绪识别系统在技术上取得了一定进展，但仍存在一些不足之处。首先，由于情绪数据集的标注工作难度较大，现有数据集的质量和多样性有限，这对模型的泛化能力和准确性造成了制约。其次，模型架构选择、参数设置等因素也影响了最终的识别性能，部分模型在特定情绪的区分上表现不足，并存在过拟合现象。面对这些挑战，未来的工作将集中于：扩大和优化标注数据集，特别是引入更多亚洲面孔及多样化表情的数据以丰富训练样本；探索更先进的深度学习模型和优化算法，提高模型的鲁棒性和精确度；以及在模型压缩、加速推理等方面开展研究，以满足实时检测对于速度与准确率的双重要求。这些改进将有助于进一步提升系统在实际应用中的表现。



## 一、数据集

### 1、预训练数据

[Facial Emotion Recognition Image Dataset (kaggle.com)](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)

该数据集包含 6 种不同的情绪：快乐、愤怒、悲伤、中立、惊讶等表情。该数据集是通过抓取 Facebook 和 Instagram 等社交网络、抓取 YouTube 视频和已有的 IMDB 和 AffectNet 数据集收集的。



### 2、收集的数据集

由于网上常见的数据集大部分的都是欧洲的人脸，对亚洲，国人的人脸比较少，这里我们自己进行收集，用于微调，我们从影视剧、互联网上收集了一个包含七种情感的人脸数据集（亚洲、国人），用于对模型进行微调。由于部分部分表情之间难以区分且是人工标注，数据集质量有待提升，最终包含如下几种类型，一共有882张数据，类型分布如下：

<img src="https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232156895.png" alt="image-20241223215644766" style="zoom: 33%;" />





部分数据集展示如下

![image-20241223220034836](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232200324.png)

完整的数据集可访问：

+ [emotion_detection_client_server_release](https://gitee.com/daetz_0/emotion_detection_client_server/releases)

+ https://gitee.com/daetz_0/emotion_detection_client_server/releases/download/model-weight-data-upload/emotion_dataset.zip



## 二、模型



### 1、卷积模型（GRAY+size48）



模型架构如下,使用灰度图 ，并且图像的尺寸是48

```python
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

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型摘要
model.summary()

```





### 2、卷积模型（RGB+size96）



```python
model_4 = Sequential()

model_4.add(Conv2D(32, (3,3), activation="selu", input_shape=input_shape))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.3))

model_4.add(Conv2D(64, (3,3), activation="selu"))
model_4.add(BatchNormalization())
model_4.add(Conv2D(64, (3,3), activation="selu"))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.4))

model_4.add(Conv2D(128, (3,3), activation="selu"))
model_4.add(BatchNormalization())
model_4.add(Conv2D(128, (3,3), activation="selu"))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.5))

model_4.add(Conv2D(256, (3,3), activation="selu"))
model_4.add(BatchNormalization())
model_4.add(Conv2D(256, (3,3), activation="selu"))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.6))

model_4.add(Flatten())
model_4.add(Dense(128, activation='selu', kernel_regularizer=l2(0.01)))
model_4.add(BatchNormalization())
model_4.add(Dropout(0.5))
model_4.add(Dense(7, activation='softmax'))

model_4.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_4.summary()
```



### 3、卷积模型（RGB+size224）

```python
# 定义模型架构
model_4 = Sequential()
num_classes = 7
input_shape = (224, 224, 3)  # 输入图像尺寸为224x224，RGB三通道

# 第一组卷积层
model_4.add(Conv2D(32, (3,3), activation="selu", input_shape=input_shape, padding='same'))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.3))

# 第二组卷积层
model_4.add(Conv2D(64, (3,3), activation="selu", padding='same'))
model_4.add(BatchNormalization())
model_4.add(Conv2D(64, (3,3), activation="selu", padding='same'))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.4))

# 第三组卷积层
model_4.add(Conv2D(128, (3,3), activation="selu", padding='same'))
model_4.add(BatchNormalization())
model_4.add(Conv2D(128, (3,3), activation="selu", padding='same'))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.5))

# 第四组卷积层
model_4.add(Conv2D(256, (3,3), activation="selu", padding='same'))
model_4.add(BatchNormalization())
model_4.add(Conv2D(256, (3,3), activation="selu", padding='same'))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.6))

# 第五组卷积层（可选，进一步增加复杂性）
model_4.add(Conv2D(512, (3,3), activation="selu", padding='same'))
model_4.add(BatchNormalization())
model_4.add(Conv2D(512, (3,3), activation="selu", padding='same'))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.7))

# Flatten层
model_4.add(Flatten())

# 全连接层
model_4.add(Dense(512, activation='selu', kernel_regularizer=l2(0.01)))
model_4.add(BatchNormalization())
model_4.add(Dropout(0.5))

model_4.add(Dense(256, activation='selu', kernel_regularizer=l2(0.01)))
model_4.add(BatchNormalization())
model_4.add(Dropout(0.5))

# 输出层
model_4.add(Dense(num_classes, activation="softmax"))

# 编译模型
model_4.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型摘要
model_4.summary()

```



### 4、EffectiveV2模型（RGB+size224）

这里使用了imagenet的权重，我们进在此基础上进行微调

这里还可以使用Facial Emotion Recognition Image Dataset 用于在人脸情绪方面进行训练，保留权重后再自己收集的数据集上微调(可选)

```python
# Import the EfficientNetV2M model pre-trained on ImageNet without the top layers
base_model = tf.keras.applications.ConvNeXtTiny(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the base model layers to prevent them from being trained
base_model.trainable = True  #Unfreeze Some Layers for Fine-Tuning

fine_tune_at = len(base_model.layers) - 20  
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Create  model
keras_model = keras.models.Sequential()
keras_model.add(base_model)
keras_model.add(keras.layers.GlobalAveragePooling2D())  # Replace Flatten with GlobalAveragePooling
keras_model.add(keras.layers.BatchNormalization())  # Add a Batch Normalization Layer
keras_model.add(keras.layers.Dense(128, activation='relu'))  # Increase the Model Capacity
keras_model.add(keras.layers.Dropout(0.3))  # Reduce Regularization
keras_model.add(keras.layers.Dense(7, activation=tf.nn.softmax))  # 6 output units for classification

# Display the model's architecture
keras_model.summary()

```



### 5、数据加载区别

![image-20241223221258667](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232212800.png)



**完整的代码和运行记录可访问**

[emotion_detection_client_server: Real-Time Facial EmotionDetection - Gitee.com](https://gitee.com/daetz_0/emotion_detection_client_server/tree/main/Tutorial/model)

或者[daetz-coder/emotion_detection_client_server: Real-Time Facial EmotionDetection With Server and Local (github.com)](https://github.com/daetz-coder/emotion_detection_client_server)

+ best_model_CNN_GRAY_size48.ipynb
+ best_model_CNN_RGB_size96
+ best_model_CNN_RGB_size224
+ best_model_EffectiveV2_RGB_size224



## 三、结果



### 1、无预训练

在这里不使用任何预训练的数据集或者模型结构，仅仅使用自己收集的数据集进行训练和微调

#### 1)、卷积模型（GRAY+size48）

```bash
Epoch 40/100
23/23 [==============================] - ETA: 0s - loss: 2.9194 - accuracy: 0.4553
Epoch 40: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05.
23/23 [==============================] - 1s 23ms/step - loss: 2.9194 - accuracy: 0.4553 - val_loss: 3.2149 - val_accuracy: 0.3722 - lr: 3.1250e-05
Epoch 41/100
23/23 [==============================] - 1s 23ms/step - loss: 2.9375 - accuracy: 0.4413 - val_loss: 3.2338 - val_accuracy: 0.3667 - lr: 1.5625e-05
Epoch 42/100
22/23 [===========================>..] - ETA: 0s - loss: 2.8755 - accuracy: 0.4659
Epoch 42: ReduceLROnPlateau reducing learning rate to 7.812500371073838e-06.
23/23 [==============================] - 1s 24ms/step - loss: 2.8805 - accuracy: 0.4637 - val_loss: 3.2164 - val_accuracy: 0.3833 - lr: 1.5625e-05
Epoch 43/100
23/23 [==============================] - 1s 24ms/step - loss: 2.8275 - accuracy: 0.4846 - val_loss: 3.2361 - val_accuracy: 0.3667 - lr: 7.8125e-06
Epoch 44/100
22/23 [===========================>..] - ETA: 0s - loss: 2.7978 - accuracy: 0.4787
Epoch 44: ReduceLROnPlateau reducing learning rate to 3.906250185536919e-06.
23/23 [==============================] - 1s 24ms/step - loss: 2.7997 - accuracy: 0.4777 - val_loss: 3.2567 - val_accuracy: 0.3611 - lr: 7.8125e-06
```



![image-20241223222511357](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232225416.png)



#### 2、卷积模型（RGB+size96）



```bash
Epoch 42/3000
23/23 [==============================] - 1s 25ms/step - loss: 2.1134 - accuracy: 0.6006 - val_loss: 2.4665 - val_accuracy: 0.5000 - lr: 3.9063e-06
Epoch 43/3000
22/23 [===========================>..] - ETA: 0s - loss: 2.1313 - accuracy: 0.5824
Epoch 43: ReduceLROnPlateau reducing learning rate to 1.9531250927684596e-06.
23/23 [==============================] - 1s 24ms/step - loss: 2.1382 - accuracy: 0.5810 - val_loss: 2.4613 - val_accuracy: 0.5056 - lr: 3.9063e-06
Epoch 44/3000
23/23 [==============================] - 1s 24ms/step - loss: 2.1184 - accuracy: 0.5950 - val_loss: 2.4612 - val_accuracy: 0.5056 - lr: 1.9531e-06
Epoch 45/3000
22/23 [===========================>..] - ETA: 0s - loss: 2.1409 - accuracy: 0.5597
Epoch 45: ReduceLROnPlateau reducing learning rate to 9.765625463842298e-07.
23/23 [==============================] - 1s 23ms/step - loss: 2.1338 - accuracy: 0.5615 - val_loss: 2.4612 - val_accuracy: 0.5056 - lr: 1.9531e-06
```

![image-20241223222630714](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232226780.png)



#### 3、卷积模型（RGB+size224）

```bash
Epoch 90/100
23/23 [==============================] - 3s 111ms/step - loss: 2.2417 - accuracy: 0.9749 - val_loss: 4.3400 - val_accuracy: 0.5111 - lr: 3.9063e-06
Epoch 91/100
22/23 [===========================>..] - ETA: 0s - loss: 2.2400 - accuracy: 0.9759
Epoch 91: ReduceLROnPlateau reducing learning rate to 1.9531250927684596e-06.
23/23 [==============================] - 3s 112ms/step - loss: 2.2397 - accuracy: 0.9763 - val_loss: 4.3405 - val_accuracy: 0.5056 - lr: 3.9063e-06
Epoch 92/100
23/23 [==============================] - 3s 113ms/step - loss: 2.2288 - accuracy: 0.9874 - val_loss: 4.3479 - val_accuracy: 0.5056 - lr: 1.9531e-06
Epoch 93/100
23/23 [==============================] - ETA: 0s - loss: 2.2212 - accuracy: 0.9846
Epoch 93: ReduceLROnPlateau reducing learning rate to 9.765625463842298e-07.
23/23 [==============================] - 3s 115ms/step - loss: 2.2212 - accuracy: 0.9846 - val_loss: 4.3524 - val_accuracy: 0.5056 - lr: 1.9531e-06
Epoch 94/100
23/23 [==============================] - 3s 114ms/step - loss: 2.2349 - accuracy: 0.9791 - val_loss: 4.3532 - val_accuracy: 0.5056 - lr: 9.7656e-07
Epoch 95/100
23/23 [==============================] - ETA: 0s - loss: 2.2182 - accuracy: 0.9832
Epoch 95: ReduceLROnPlateau reducing learning rate to 4.882812731921149e-07.
23/23 [==============================] - 3s 117ms/step - loss: 2.2182 - accuracy: 0.9832 - val_loss: 4.3539 - val_accuracy: 0.5056 - lr: 9.7656e-07
```

![image-20241223222732866](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232227934.png)



### 2、含预训练

#### 4) EffectiveV2模型（RGB+size224）

![image-20241223222900241](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232229595.png)

![image-20241223222833220](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232228306.png)



虽然表现的比之前好，但是出现了明显的过拟合的问题



### 3、参数比较

四种模型的完整权重可访问，权重文件需要放入`pth`文件夹下

+ https://gitee.com/daetz_0/emotion_detection_client_server/releases

+ [ best_model_CNN_Gray_size48.h5 ](https://gitee.com/daetz_0/emotion_detection_client_server/releases/download/model-weight-data-upload/best_model_CNN_Gray_size48.h5)

+ [ best_model_CNN_RGB_size96.h5 ](https://gitee.com/daetz_0/emotion_detection_client_server/releases/download/model-weight-data-upload/best_model_CNN_RGB_size96.h5)

+ https://pan.baidu.com/s/1TgOv-Eeojh62JPSAx6AeWw?pwd=2024

从实验结果来看，参数越多模型的检测性能越好，但是随之而来的是复杂的计算量，参数量成倍提升，由于实时检测对延时的要求较高，需要自行权衡

![image-20241223223409956](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202412232234037.png)





## 四、使用

### 1、用于训练

[emotion_detection_client_server: Real-Time Facial EmotionDetection - Gitee.com](https://gitee.com/daetz_0/emotion_detection_client_server/tree/main/Tutorial/model)

+ best_model_CNN_GRAY_size48.ipynb
+ best_model_CNN_RGB_size96
+ best_model_CNN_RGB_size224
+ best_model_EffectiveV2_RGB_size224

仅需要使用上述的内容，建议 tensorflow>=2.9.0 Python 3.8(ubuntu20.04) Cuda 11.2

```less
drwxr-xr-x  4 root root   46 Dec 23 22:41 ./
drwxr-xr-x 13 root root 4096 Dec 23 22:39 ../
drwxr-xr-x  9 root root  133 Dec 23 22:39 dataset/
drwxr-xr-x  2 root root 4096 Dec 23 22:40 model/
-rw-r--r-- 1 root root  187223 Dec 23 22:40 best_model_CNN_GRAY_size48.ipynb
-rw-r--r-- 1 root root  270266 Dec 23 22:40 best_model_CNN_RGB_size224.ipynb
-rw-r--r-- 1 root root  260946 Dec 23 22:40 best_model_CNN_RGB_size96.ipynb
-rw-r--r-- 1 root root 1029572 Dec 23 22:40 best_model_EffectiveV2_RGB_size224.ipynb
```



### 2、用于检测

对于上述的四种方法，我们分别提供四种不同的加载文件，分别使用下述内容，上传不同格式的图像

+ cv2.COLOR_BGR2GRAY
+ cv2.COLOR_BGR2RGB
+ cv2.resize(img_gray, (48, 48))
+ cv2.resize(image_rgb, (96, 96))
+ cv2.resize(image_rgb, (224, 224))

关键部分代码如下,完整的内容可访问`EmotionDetection/modules`

```python
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
    input_shape = (224, 224, 3)  # 输入图像尺寸为224x224，RGB三通道

    # 第一组卷积层
    model.add(Conv2D(32, (3,3), activation="selu", input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # 第二组卷积层
    model.add(Conv2D(64, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    # 第三组卷积层
    model.add(Conv2D(128, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    # 第四组卷积层
    model.add(Conv2D(256, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.6))

    # 第五组卷积层（可选，进一步增加复杂性）
    model.add(Conv2D(512, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), activation="selu", padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.7))

    # Flatten层
    model.add(Flatten())

    # 全连接层
    model.add(Dense(512, activation='selu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='selu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # 输出层
    model.add(Dense(num_classes, activation="softmax"))

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

```

+ CNN_GRAY_size48.py
+ CNN_RGB_size96.py

+ CNN_RGB_size224.py
+ Emotion_EffectiveV2.py

如果需要使用，请替换原`Emotion.py`的内容



### 3、直接使用

```bash
git clone https://gitee.com/daetz_0/emotion_detection_client_server.git
# git clone https://github.com/daetz-coder/emotion_detection_client_server.git
cd emotion_detection_client_server
pip install -r requirements.txt
python main.py
# use the "q" exit
```





## 五、参考链接



+ [GitHub - opencv/opencv: Open Source Computer Vision Library](https://github.com/opencv/opencv)

+ [GitHub - serengil/deepface: A Lightweight Face Recognition and Facial Attribute Analysis](https://github.com/serengil/deepface)
+ [Facial Emotion Recognition (kaggle.com)](https://www.kaggle.com/code/gauravsharma99/facial-emotion-recognition/notebook)
+ [Facial Expression Recognition(FER)Challenge (kaggle.com)](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge)

+ [📈 EfficientNetV2 😃💡📊 Emotion Recognition 🤖 (kaggle.com)](https://www.kaggle.com/code/guanlintao/efficientnetv2-emotion-recognition)
+ [Facial Emotion Recognition Image Dataset (kaggle.com)](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)
# v2.0

## 一、Description:

Perform lightweight operations locally on Windows, while handling complex computations on the cloud using Linux.

+ Clinet： Windows
+ Server： Ubuntu

## 二、Installation

```bash
git clone https://github.com/daetz-coder/emotion_detection_server_local.git
```

```less
emotion_detection_server_local:
├── README.md
└── v2.0
    ├── README.md
    ├── requirements.ipynb
    ├── client
    │   ├── app.py
    │   └── requirements.txt
    └── server
        ├── app.py
        ├── recognition.py
        └── requirements.txt
```

### **Installation Steps:**

1.Navigate to the `v2.0` directory:

```
cd v2.0
```

2.Client Setup:

```bash
cd client
pip install -r requirements.txt
```

3.Server Setup:

```bash
cd server
pip install -r requirements.txt
```

**Note:**
Complete dependencies on the server and their corresponding CUDA versions can be found in `requirements.ipynb`.

## 三、Startup

### 1、Client

```bash
# Navigate to the client directory
cd v2.0/client/
python app.py
```

### 2、Server

```bash
# Navigate to the server directory
cd v2.0/server
python app.py
```

**Default IP and Port:**

- **IP:** `0.0.0.0`
- **Port:** `5000`

```bash
2024-11-03 11:04:39.125776: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-11-03 11:04:39.553741: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.103:5000
Press CTRL+C to quit
```

## 四、Features

![image-20241103110829933](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202411031108193.png)

### 1、Image Recognition

![image-20241103110907672](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202411031109829.png)

```bash
2024-11-03 11:09:29.580677: I tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:432] Loaded cuDNN version 8907
2024-11-03 11:09:30.476590: I tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:606] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
100.68.1.119 - - [03/Nov/2024 11:09:30] "POST /analyze_image HTTP/1.1" 200 -
2024-11-03 11:09:30,495 - INFO - 100.68.1.119 - - [03/Nov/2024 11:09:30] "POST /analyze_image HTTP/1.1" 200 -
```

![image-20241103111016930](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202411031110631.png)

### 2、Video Recognition

![image-20241103111124756](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202411031111899.png)

```bash
Processing Video:  26%|█████▍               | 77/300 [00:20<01:12,  3.08frame/s]
```

### 3、Online Recognition

![image-20241103111433014](https://daetz-image.oss-cn-hangzhou.aliyuncs.com/img/202411031114531.png)

```bash
100.68.1.119 - - [03/Nov/2024 11:14:38] "POST /analyze_online HTTP/1.1" 200 -
INFO:werkzeug:100.68.1.119 - - [03/Nov/2024 11:14:38] "POST /analyze_online HTTP/1.1" 200 -
100.68.1.119 - - [03/Nov/2024 11:14:39] "POST /analyze_online HTTP/1.1" 200 -
INFO:werkzeug:100.68.1.119 - - [03/Nov/2024 11:14:39] "POST /analyze_online HTTP/1.1" 200 -
100.68.1.119 - - [03/Nov/2024 11:14:39] "POST /analyze_online HTTP/1.1" 200 -
INFO:werkzeug:100.68.1.119 - - [03/Nov/2024 11:14:39] "POST /analyze_online HTTP/1.1" 200 -
100.68.1.119 - - [03/Nov/2024 11:14:39] "POST /analyze_online HTTP/1.1" 200 -
INFO:werkzeug:100.68.1.119 - - [03/Nov/2024 11:14:39] "POST /analyze_online HTTP/1.1" 200 -
100.68.1.119 - - [03/Nov/2024 11:14:40] "POST /analyze_online HTTP/1.1" 200 -
INFO:werkzeug:100.68.1.119 - - [03/Nov/2024 11:14:40] "POST /analyze_online HTTP/1.1" 200 -
100.68.1.119 - - [03/Nov/2024 11:14:40] "POST /analyze_online HTTP/1.1" 200 -
INFO:werkzeug:100.68.1.119 - - [03/Nov/2024 11:14:40] "POST /analyze_online HTTP/1.1" 200 -
100.68.1.119 - - [03/Nov/2024 11:14:40] "POST /analyze_online HTTP/1.1" 200 -
INFO:werkzeug:100.68.1.119 - - [03/Nov/2024 11:14:40] "POST /analyze_online HTTP/1.1" 200 -
100.68.1.119 - - [03/Nov/2024 11:14:41] "POST /analyze_online HTTP/1.1" 200 -
INFO:werkzeug:100.68.1.119 - - [03/Nov/2024 11:14:41] "POST /analyze_online HTTP/1.1" 200 -
```

## 五、Summary

*This project involves a client-server architecture where lightweight operations are handled locally on a Windows machine (client), and more complex computations are loaded to a cloud-based Ubuntu server (server). The setup includes cloning the repository, installing necessary dependencies for both client and server, and starting the respective applications. The functionalities supported include image recognition, video recognition, and online recognition, each with corresponding logs demonstrating their operation.*

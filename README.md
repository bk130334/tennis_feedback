## Project
- 테니스 자세에 대한 피드백 제공

## Data 
- input : 테니스 자세 이미지

## Folder  
```
├── data
│ └── backhand
│ └── forehand
│ └── serve
│ └── stand
│
└── main.py
│
└── model.pkl
│
└── scaler.pkl
│
└── pose_data.csv
│
└── requirements.txt
```

## Environment
- Anaconda Prompt

## Code
0. 가상환경이 없는 경우 다음 명령어로 가상환경 만들기

```
conda create --name <환경이름> python=3.9
```
⬇️ e.g.
```
conda create --name pose_env python=3.9
```

1. 가상환경 활성화 및 필요한 라이브러리 설치
```
conda activate <환경 이름>
pip install -r .\requirements.txt
```
(만약 다음과 같은 오류 코드가 출력되었다면 필요한 라이브러리를 직접 설치)
```
ERROR: Ignored the following versions that require a different python version: 0.28.0 Requires-Python >=3.7, <3.11; 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11
ERROR: Could not find a version that satisfies the requirement tensorflow-io-gcs-filesystem==0.37.1 (from versions: 0.29.0, 0.30.0, 0.31.0)
ERROR: No matching distribution found for tensorflow-io-gcs-filesystem==0.37.1
```
라이브러리 설치 명령어
```
pip install <라이브러리 이름>
```
⬇️ e.g.
```
ModuleNotFoundError: No module named 'tensorflow'
--> pip install tensorflow
ModuleNotFoundError: No module named 'cv2'
--> pip install opencv-python
ModuleNotFoundError: No module named 'mediapipe'
--> pip install mediapipe
ModuleNotFoundError: No module named 'joblib'
--> pip install joblib
ModuleNotFoundError: No module named 'sklearn'
--> pip install scikit-learn
```

2. 이미지 파일 입력
```
main.py와 같은 경로에 분석을 원하는 이미지 파일을 "image.jpg" 라는 이름으로 배치
```

3. 코드 실행
```
python main.py
```

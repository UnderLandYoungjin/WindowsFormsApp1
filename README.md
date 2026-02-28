# YOLOv8 USB Camera Real-time Object Detection
## C# WinForms 프로젝트 개요
---


![Video Project 3](https://github.com/user-attachments/assets/413440ee-fbba-495e-97e6-1e5e548dcd6a)




## 1. 프로젝트 소개

### 1.1 프로젝트 목적
본 프로젝트는 **USB 카메라로 실시간 영상을 캡처**하고, **YOLOv8 딥러닝 모델**을 이용하여 영상 속 객체를 실시간으로 탐지(Object Detection)하는 Windows 데스크톱 애플리케이션입니다.

### 1.2 핵심 기능
| 기능 | 설명 |
|------|------|
| USB 카메라 실시간 영상 표시 | OpenCvSharp을 통한 VideoCapture로 프레임 획득 및 화면 출력 |
| YOLOv8 객체 탐지 | ONNX Runtime으로 YOLOv8 모델 추론, COCO 80클래스 탐지 |
| 바운딩 박스 시각화 | 탐지된 객체에 클래스명, 신뢰도, 색상별 박스 표시 |
| Confidence Threshold 조절 | TrackBar로 탐지 민감도 실시간 조절 (0.10 ~ 0.95) |
| 카메라 선택 | Camera 0~4 중 선택 가능 |
| 탐지 ON/OFF 토글 | 순수 카메라 영상만 보기 / 탐지 적용 전환 |
| 실시간 FPS 표시 | 초당 프레임 수 모니터링 |

### 1.3 사용 기술 스택
| 구분 | 기술 |
|------|------|
| 언어 | C# (.NET Framework 4.7.2+) |
| UI 프레임워크 | Windows Forms (WinForms) |
| 컴퓨터 비전 | OpenCvSharp4 (OpenCV의 C# 래퍼) |
| 딥러닝 추론 | Microsoft.ML.OnnxRuntime |
| 모델 | YOLOv8n (Ultralytics, ONNX 형식) |
| IDE | Visual Studio 2022 |

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────┐
│                    MainForm (UI Thread)                  │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │ PictureBox│  │ Controls │  │ Detection Result List  │ │
│  │ (카메라   │  │ (버튼,    │  │ (탐지 결과 텍스트)      │ │
│  │  영상)    │  │ 슬라이더) │  │                        │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │ BeginInvoke (UI 업데이트)
                     │
┌────────────────────┴────────────────────────────────────┐
│              CaptureLoop (Background Thread)             │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │VideoCapture│──→│ YoloV8Detector│──→│ DrawDetections │  │
│  │(프레임획득)│    │ (ONNX 추론)   │    │ (박스 그리기)  │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │   YOLOv8n.onnx      │
              │   (ONNX 모델 파일)   │
              └─────────────────────┘
```

### 2.2 데이터 흐름 (파이프라인)

```
USB 카메라 → VideoCapture.Read() → Mat (BGR 프레임)
    │
    ▼
[전처리 - Preprocess()]
    ├─ Resize (Letterbox, 640×640)
    ├─ BGR → RGB 변환
    ├─ 정규화 (0~255 → 0.0~1.0)
    └─ Tensor 변환 [1, 3, 640, 640] (NCHW)
    │
    ▼
[추론 - InferenceSession.Run()]
    └─ ONNX Runtime에서 YOLOv8 모델 실행
    │
    ▼
[후처리 - PostProcess()]
    ├─ 출력 텐서 파싱 [1, 84, 8400]
    ├─ Confidence Threshold 필터링
    ├─ 좌표 역변환 (Letterbox → 원본 크기)
    └─ NMS (Non-Maximum Suppression)
    │
    ▼
[시각화 - DrawDetections()]
    ├─ 바운딩 박스 그리기
    ├─ 클래스명 + 신뢰도 라벨
    └─ FPS 오버레이
    │
    ▼
BitmapConverter.ToBitmap() → PictureBox.Image (UI 표시)
```

---

## 3. 파일 구조 및 클래스 설명

### 3.1 파일 구성

```
WindowsFormsApp1/
├── Properties/
│   └── AssemblyInfo.cs          # 어셈블리 메타데이터
├── App.config                    # 앱 설정 파일
├── Detection.cs                  # 탐지 결과 데이터 클래스
├── MainForm.cs                   # 메인 폼 (UI + 카메라 루프)
├── YoloV8Detector.cs             # YOLOv8 ONNX 추론 엔진
├── Program.cs                    # 프로그램 진입점
├── packages.config               # NuGet 패키지 목록
└── WindowsFormsApp1.csproj       # 프로젝트 설정 파일
```

### 3.2 Detection.cs - 탐지 결과 데이터 클래스

**역할:** YOLOv8 추론 결과 하나를 담는 POCO(Plain Old CLR Object) 클래스

```csharp
public class Detection
{
    public int ClassId { get; set; }       // COCO 클래스 ID (0~79)
    public string ClassName { get; set; }  // 클래스 이름 (예: "person")
    public float Confidence { get; set; }  // 신뢰도 (0.0 ~ 1.0)
    public float X { get; set; }           // 바운딩 박스 좌상단 X 좌표
    public float Y { get; set; }           // 바운딩 박스 좌상단 Y 좌표
    public float Width { get; set; }       // 바운딩 박스 너비
    public float Height { get; set; }      // 바운딩 박스 높이
}
```

**강의 포인트:**
- 데이터 전달 객체(DTO) 패턴의 기본 예시
- 프로퍼티(Property)를 이용한 캡슐화
- `ToString()` 오버라이드를 통한 디버깅 편의성

---

### 3.3 YoloV8Detector.cs - YOLO 추론 엔진 (핵심 클래스)

**역할:** ONNX 모델 로드, 이미지 전처리, 추론 실행, 후처리(NMS)까지 전체 탐지 파이프라인 담당

#### 주요 메서드 상세

| 메서드 | 역할 | 입력 | 출력 |
|--------|------|------|------|
| `Detect(Mat frame)` | 전체 탐지 파이프라인 실행 | OpenCV Mat (BGR) | `List<Detection>` |
| `DrawDetections(...)` | 바운딩 박스 + 라벨 그리기 | Mat + Detections | Mat에 직접 그림 |
| `Preprocess(...)` | Letterbox + 정규화 + 텐서화 | Mat | `DenseTensor<float>` |
| `PostProcess(...)` | 출력 파싱 + NMS | Tensor | `List<Detection>` |
| `NMS(...)` | 중복 박스 제거 | boxes, scores | 유지할 인덱스 |
| `ComputeIoU(...)` | 두 박스의 겹침 비율 계산 | Rect a, Rect b | float (0~1) |

#### 3.3.1 전처리 (Preprocess) 상세

```
원본 이미지 (예: 640×480)
        │
        ▼
[비율 계산] ratio = min(640/640, 640/480) = 1.0
        │
        ▼
[리사이즈] 640×480 → 640×480 (비율 유지)
        │
        ▼
[Letterbox 패딩] 640×640 (상하 80px씩 회색 패딩)
  ┌────────────────┐
  │  회색 (114,114) │  ← padY = 80
  ├────────────────┤
  │                │
  │   실제 이미지   │  ← 640×480
  │                │
  ├────────────────┤
  │  회색 (114,114) │  ← padY = 80
  └────────────────┘
        │
        ▼
[BGR → RGB] 색상 채널 순서 변환
        │
        ▼
[정규화] 0~255 → 0.0~1.0 (÷255)
        │
        ▼
[텐서 변환] HWC → NCHW [1, 3, 640, 640]
```

**Letterbox를 사용하는 이유:**
- YOLO 모델은 고정 크기(640×640) 입력을 요구
- 단순 리사이즈는 종횡비(aspect ratio)를 왜곡시켜 탐지 성능 저하
- Letterbox는 비율을 유지하며 나머지를 패딩으로 채움

**강의 포인트:**
- `unsafe` 키워드: 포인터를 이용한 고성능 픽셀 접근
- NCHW 텐서 포맷: Batch, Channel, Height, Width 순서
- 왜 BGR→RGB 변환이 필요한지 (OpenCV는 BGR, 모델 학습은 RGB)

#### 3.3.2 YOLOv8 출력 구조

```
YOLOv8 출력 텐서: [1, 84, 8400]

    84 = 4 (바운딩 박스) + 80 (COCO 클래스 수)
  8400 = 탐지 후보 수 (anchor-free 방식)

인덱스 구조:
  output[0, 0, i] = center_x   ┐
  output[0, 1, i] = center_y   ├─ 바운딩 박스 (xywh)
  output[0, 2, i] = width      │
  output[0, 3, i] = height     ┘
  output[0, 4, i] = class_0 score (person)
  output[0, 5, i] = class_1 score (bicycle)
  ...
  output[0, 83, i] = class_79 score (toothbrush)
```

**YOLOv5 vs YOLOv8 출력 차이 (주의!):**
| 항목 | YOLOv5 | YOLOv8 |
|------|--------|--------|
| 출력 형태 | [1, 25200, 85] | [1, 84, 8400] |
| Objectness score | 있음 (별도) | 없음 (통합) |
| 텐서 레이아웃 | [batch, detections, data] | [batch, data, detections] |
| Confidence 계산 | obj_score × class_score | class_score 직접 사용 |

#### 3.3.3 NMS (Non-Maximum Suppression) 알고리즘

```
입력: 여러 개의 바운딩 박스 + 각 박스의 신뢰도 점수

Step 1: 신뢰도 기준 내림차순 정렬
Step 2: 가장 높은 점수의 박스를 선택 (keep)
Step 3: 선택된 박스와 나머지 박스의 IoU 계산
Step 4: IoU > threshold (0.45) 인 박스 제거 (suppress)
Step 5: 남은 박스 중 Step 2~4 반복

예시:
  Box A (0.95) ─ keep
  Box B (0.90) ─ IoU(A,B) = 0.8 > 0.45 → suppress (A와 같은 물체)
  Box C (0.85) ─ IoU(A,C) = 0.1 < 0.45 → keep (다른 물체)
  Box D (0.70) ─ IoU(C,D) = 0.7 > 0.45 → suppress (C와 같은 물체)

결과: Box A, Box C만 유지
```

**IoU (Intersection over Union) 계산:**
```
    ┌─────────┐
    │    A    │
    │   ┌─────┼────┐
    │   │ 교집│합  │
    └───┼─────┘    │
        │     B    │
        └──────────┘

IoU = 교집합 면적 / 합집합 면적
    = intersection / (area_A + area_B - intersection)
```

---

### 3.4 MainForm.cs - 메인 폼 (UI + 카메라 루프)

**역할:** WinForms UI 구성, 카메라 캡처 스레드 관리, 사용자 인터랙션 처리

#### UI 레이아웃

```
┌──────────────────────────────────────────────────────┐
│  YOLOv8 USB Camera Detection                     [─][□][✕]│
├──────────────────────────────┬───────────────────────┤
│                              │ Camera: [Camera 0  ▼] │
│                              │                       │
│                              │ [Select ONNX Model]   │
│                              │                       │
│    카메라 영상               │ Confidence Threshold:  │
│    (PictureBox)              │ ──●────────── 0.50    │
│    800 × 600                 │                       │
│                              │ ☑ Enable Detection    │
│                              │                       │
│                              │ [▶ Start] [■ Stop]   │
│                              │                       │
│                              │ FPS: 24.5             │
│                              │ Status: Running...    │
│                              │                       │
│                              │ Detections (3):       │
│                              │ * person (92%)        │
│                              │ * chair (85%)         │
│                              │ * laptop (78%)        │
└──────────────────────────────┴───────────────────────┘
```

#### 멀티스레딩 구조

```
[UI Thread (메인)]                    [CaptureLoop Thread (백그라운드)]
      │                                        │
      │  btnStart_Click()                      │
      │  ──→ new Thread(CaptureLoop)           │
      │       .Start() ──────────────→         │
      │                                 while(_isRunning)
      │                                   │ capture.Read(frame)
      │                                   │ detector.Detect(frame)
      │                                   │ detector.DrawDetections()
      │                                   │ BitmapConverter.ToBitmap()
      │                                   │
      │  ←──── BeginInvoke() ─────────────┘
      │  pictureBox.Image = bitmap
      │  lblFps.Text = fps
      │  lblDetections.Text = ...
      │
      │  btnStop_Click()
      │  ──→ _isRunning = false
      │       thread.Join(2000)
```

**강의 포인트:**
- **volatile 키워드:** `_isRunning` 변수에 사용 → 스레드 간 가시성 보장
- **BeginInvoke:** 백그라운드 스레드에서 UI 컨트롤 접근 시 반드시 필요 (크로스 스레드 예외 방지)
- **Thread.Join(2000):** 스레드 종료 대기 (최대 2초 타임아웃)
- **IDisposable 패턴:** Mat, VideoCapture 등 네이티브 리소스 해제

---

### 3.5 Program.cs - 프로그램 진입점

```csharp
static class Program
{
    [STAThread]  // COM 컴포넌트 호환을 위한 STA 스레드 모델
    static void Main()
    {
        Application.EnableVisualStyles();           // 시각 스타일 활성화
        Application.SetCompatibleTextRenderingDefault(false); // GDI+ 텍스트 렌더링
        Application.Run(new MainForm());            // 메인 폼 실행 (메시지 루프 시작)
    }
}
```

---

## 4. NuGet 패키지 의존성

| 패키지 | 버전 | 역할 |
|--------|------|------|
| OpenCvSharp4 | 4.9.0.20240103 | OpenCV C# 래퍼 (핵심 라이브러리) |
| OpenCvSharp4.runtime.win | 4.9.0.20240103 | Windows용 네이티브 바이너리 |
| OpenCvSharp4.Extensions | 4.9.0.20240103 | BitmapConverter 등 확장 기능 |
| Microsoft.ML.OnnxRuntime | 1.17.1 | ONNX 모델 추론 엔진 (CPU) |

**GPU 가속 사용 시:**
- `Microsoft.ML.OnnxRuntime` 대신 `Microsoft.ML.OnnxRuntime.Gpu` 설치
- NVIDIA CUDA Toolkit + cuDNN 설치 필요

---

## 5. ONNX 모델 준비

### 5.1 YOLOv8 모델 변환 (PyTorch → ONNX)

```bash
# Ultralytics 설치
pip install ultralytics

# YOLOv8n (nano) 모델을 ONNX로 변환
yolo export model=yolov8n.pt format=onnx imgsz=640
```

### 5.2 YOLOv8 모델 종류별 비교

| 모델 | 파라미터 수 | 크기 (MB) | mAP50-95 | 추론 속도 (CPU) | 용도 |
|------|-----------|-----------|----------|----------------|------|
| YOLOv8n | 3.2M | 6.2 | 37.3 | 빠름 | 실시간, 경량 |
| YOLOv8s | 11.2M | 22.5 | 44.9 | 보통 | 균형 |
| YOLOv8m | 25.9M | 52.0 | 50.2 | 느림 | 정확도 우선 |
| YOLOv8l | 43.7M | 87.7 | 52.9 | 매우 느림 | 고정밀 |
| YOLOv8x | 68.2M | 136.7 | 53.9 | 가장 느림 | 최고 정확도 |

### 5.3 COCO 데이터셋 80 클래스

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird,
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
donut, cake, chair, couch, potted plant, bed, dining table, toilet,
tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

---

## 6. 환경 설정 및 실행 가이드

### 6.1 필수 요구사항
- Windows 10/11 (64비트)
- Visual Studio 2022 (Community 이상)
- .NET Framework 4.7.2 이상
- Python 3.8+ (ONNX 모델 변환용)
- USB 웹캠

### 6.2 설치 순서

```
1. Visual Studio 2022에서 WinForms 프로젝트 생성
2. NuGet 패키지 4개 설치 (패키지 관리자 콘솔):
   - Install-Package OpenCvSharp4 -Version 4.9.0.20240103
   - Install-Package OpenCvSharp4.runtime.win -Version 4.9.0.20240103
   - Install-Package OpenCvSharp4.Extensions -Version 4.9.0.20240103
   - Install-Package Microsoft.ML.OnnxRuntime -Version 1.17.1
3. 프로젝트 속성 → 빌드 → "안전하지 않은 코드 허용" 체크
4. 소스 코드 파일 추가 (Detection.cs, YoloV8Detector.cs, MainForm.cs, Program.cs)
5. YOLOv8 ONNX 모델 변환 후 실행 폴더에 복사
6. 빌드(Ctrl+Shift+B) → 실행(F5)
```

---

## 7. 핵심 개념 요약 (강의 정리)

### 7.1 컴퓨터 비전 기초
- **BGR vs RGB:** OpenCV는 BGR 순서, 딥러닝 모델은 RGB 순서 사용
- **Letterbox Resizing:** 종횡비를 유지하면서 고정 크기로 리사이즈하는 기법
- **정규화 (Normalization):** 픽셀 값을 0~1 범위로 변환하여 모델 학습/추론 안정성 확보

### 7.2 딥러닝 추론
- **ONNX (Open Neural Network Exchange):** 다양한 프레임워크 간 모델 교환 표준 형식
- **ONNX Runtime:** Microsoft의 고성능 추론 엔진 (CPU/GPU 지원)
- **Tensor:** 다차원 배열, NCHW 형식은 (Batch, Channel, Height, Width)

### 7.3 객체 탐지
- **Bounding Box:** 탐지된 객체를 감싸는 사각형 (x, y, width, height)
- **Confidence Score:** 모델이 예측한 해당 클래스일 확률 (0~1)
- **NMS (Non-Maximum Suppression):** 같은 객체에 대한 중복 탐지 제거
- **IoU (Intersection over Union):** 두 박스의 겹침 비율, NMS의 핵심 지표

### 7.4 C# 프로그래밍
- **멀티스레딩:** UI 응답성을 유지하면서 백그라운드 작업 수행
- **volatile 키워드:** 스레드 간 변수 가시성 보장
- **BeginInvoke:** 크로스 스레드 UI 접근을 안전하게 처리
- **IDisposable:** 네이티브 리소스(카메라, 모델 세션)의 적절한 해제
- **unsafe 코드:** 포인터를 이용한 고성능 메모리 접근 (이미지 처리 최적화)

---

## 8. 확장 과제 (심화 학습)

| 난이도 | 과제 | 힌트 |
|--------|------|------|
| ★☆☆ | 탐지 결과를 CSV 파일로 저장 | StreamWriter + 타임스탬프 |
| ★☆☆ | 특정 클래스만 필터링 표시 | ComboBox로 클래스 선택 |
| ★★☆ | 탐지 영상 녹화 (AVI/MP4) | VideoWriter 클래스 활용 |
| ★★☆ | 커스텀 YOLO 모델 적용 | Ultralytics로 자체 데이터 학습 후 ONNX 변환 |
| ★★☆ | 알림 기능 (특정 객체 감지 시) | 소리 재생 또는 이메일 발송 |
| ★★★ | GPU 가속 적용 | OnnxRuntime.Gpu + CUDA |
| ★★★ | 다중 카메라 동시 모니터링 | 멀티 VideoCapture + 탭 UI |
| ★★★ | 객체 추적 (Tracking) 추가 | SORT/DeepSORT 알고리즘 |

---

## 9. 자주 발생하는 문제 (Troubleshooting)

| 증상 | 원인 | 해결 |
|------|------|------|
| 카메라가 열리지 않음 | 카메라 인덱스 불일치 또는 다른 프로그램에서 사용 중 | Camera 인덱스 변경, 다른 프로그램 종료 |
| ONNX 모델 로드 실패 | 파일 경로 오류 또는 호환되지 않는 ONNX 버전 | 모델 파일 경로 확인, opset 버전 확인 |
| FPS가 매우 낮음 (1~5) | CPU 추론의 한계 | YOLOv8n 사용, GPU 가속 적용, 입력 해상도 축소 |
| 빌드 에러 (unsafe) | 안전하지 않은 코드 허용 미설정 | 프로젝트 속성 → 빌드 → "안전하지 않은 코드 허용" 체크 |
| BitmapConverter 에러 | OpenCvSharp4.Extensions 미설치 | NuGet에서 패키지 추가 설치 |
| 크로스 스레드 예외 | UI 스레드 외부에서 컨트롤 접근 | BeginInvoke 사용 |

---

## 10. 참고 자료

- [Ultralytics YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [OpenCvSharp GitHub](https://github.com/shimat/opencvsharp)
- [ONNX Runtime 공식 문서](https://onnxruntime.ai/docs/)
- [COCO Dataset](https://cocodataset.org/)
- [Microsoft.ML.OnnxRuntime NuGet](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime)

---

> **작성 기준:** Visual Studio 2022, .NET Framework 4.7.2, YOLOv8n (Ultralytics v8.4.x), C#

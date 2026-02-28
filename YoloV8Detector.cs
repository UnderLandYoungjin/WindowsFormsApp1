using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace WindowsFormsApp1
{
    public class YoloV8Detector : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _inputWidth = 640;
        private readonly int _inputHeight = 640;
        private readonly float _confThreshold;
        private readonly float _iouThreshold;

        private static readonly string[] CocoClassNames = new string[]
        {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        };

        private static readonly Scalar[] ClassColors;

        static YoloV8Detector()
        {
            var rng = new Random(42);
            ClassColors = new Scalar[80];
            for (int i = 0; i < 80; i++)
            {
                ClassColors[i] = new Scalar(rng.Next(50, 255), rng.Next(50, 255), rng.Next(50, 255));
            }
        }

        public YoloV8Detector(string modelPath, float confThreshold = 0.5f, float iouThreshold = 0.45f)
        {
            _confThreshold = confThreshold;
            _iouThreshold = iouThreshold;

            var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            try
            {
                sessionOptions.AppendExecutionProvider_CUDA();
                System.Diagnostics.Debug.WriteLine("CUDA provider enabled");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("CUDA provider FAILED: " + ex.ToString());
            }

            _session = new InferenceSession(modelPath, sessionOptions);
        }

        public List<Detection> Detect(Mat frame)
        {
            if (frame.Empty()) return new List<Detection>();

            int origWidth = frame.Width;
            int origHeight = frame.Height;

            float ratioX, ratioY;
            int padX, padY;
            var inputTensor = Preprocess(frame, out ratioX, out ratioY, out padX, out padY);

            var inputName = _session.InputNames[0];
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            using (var results = _session.Run(inputs))
            {
                var outputTensor = results.First().AsTensor<float>();
                var detections = PostProcess(outputTensor, ratioX, ratioY, padX, padY, origWidth, origHeight);
                return detections;
            }
        }

        public void DrawDetections(Mat frame, List<Detection> detections)
        {
            foreach (var det in detections)
            {
                int x1 = (int)det.X;
                int y1 = (int)det.Y;
                int x2 = (int)(det.X + det.Width);
                int y2 = (int)(det.Y + det.Height);

                var color = ClassColors[det.ClassId % 80];

                Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), color, 2);

                string label = string.Format("{0} {1:F2}", det.ClassName, det.Confidence);
                int baseline;
                var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.6, 1, out baseline);
                int labelY = Math.Max(y1 - 5, textSize.Height + 5);

                Cv2.Rectangle(frame,
                    new Point(x1, labelY - textSize.Height - 5),
                    new Point(x1 + textSize.Width + 5, labelY + 5),
                    color, -1);

                Cv2.PutText(frame, label,
                    new Point(x1 + 2, labelY),
                    HersheyFonts.HersheySimplex, 0.6,
                    new Scalar(255, 255, 255), 1);
            }
        }

        private DenseTensor<float> Preprocess(Mat frame, out float ratioX, out float ratioY, out int padX, out int padY)
        {
            int origW = frame.Width;
            int origH = frame.Height;

            float ratio = Math.Min((float)_inputWidth / origW, (float)_inputHeight / origH);
            int newW = (int)(origW * ratio);
            int newH = (int)(origH * ratio);

            padX = (_inputWidth - newW) / 2;
            padY = (_inputHeight - newH) / 2;

            ratioX = ratio;
            ratioY = ratio;

            Mat resized = new Mat();
            Cv2.Resize(frame, resized, new Size(newW, newH));

            Mat padded = new Mat(_inputHeight, _inputWidth, MatType.CV_8UC3, new Scalar(114, 114, 114));
            resized.CopyTo(padded[new Rect(padX, padY, newW, newH)]);

            Mat rgb = new Mat();
            Cv2.CvtColor(padded, rgb, ColorConversionCodes.BGR2RGB);

            var tensor = new DenseTensor<float>(new int[] { 1, 3, _inputHeight, _inputWidth });

            unsafe
            {
                byte* ptr = (byte*)rgb.Data;
                for (int y = 0; y < _inputHeight; y++)
                {
                    for (int x = 0; x < _inputWidth; x++)
                    {
                        int idx = (y * _inputWidth + x) * 3;
                        tensor[0, 0, y, x] = ptr[idx] / 255.0f;
                        tensor[0, 1, y, x] = ptr[idx + 1] / 255.0f;
                        tensor[0, 2, y, x] = ptr[idx + 2] / 255.0f;
                    }
                }
            }

            resized.Dispose();
            padded.Dispose();
            rgb.Dispose();

            return tensor;
        }

        private List<Detection> PostProcess(Tensor<float> output, float ratioX, float ratioY,
                                             int padX, int padY, int origWidth, int origHeight)
        {
            var detections = new List<Detection>();
            var boxes = new List<Rect>();
            var scores = new List<float>();
            var classIds = new List<int>();

            int numClasses = 80;
            int numDetections = (int)output.Dimensions[2];

            for (int i = 0; i < numDetections; i++)
            {
                float maxScore = 0;
                int maxClassId = 0;

                for (int c = 0; c < numClasses; c++)
                {
                    float score = output[0, c + 4, i];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClassId = c;
                    }
                }

                if (maxScore < _confThreshold) continue;

                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

                float x1 = (cx - w / 2 - padX) / ratioX;
                float y1 = (cy - h / 2 - padY) / ratioY;
                float bw = w / ratioX;
                float bh = h / ratioY;

                x1 = Math.Max(0, Math.Min(x1, origWidth - 1));
                y1 = Math.Max(0, Math.Min(y1, origHeight - 1));
                bw = Math.Min(bw, origWidth - x1);
                bh = Math.Min(bh, origHeight - y1);

                boxes.Add(new Rect((int)x1, (int)y1, (int)bw, (int)bh));
                scores.Add(maxScore);
                classIds.Add(maxClassId);
            }

            if (boxes.Count > 0)
            {
                var indices = NMS(boxes, scores, _iouThreshold);

                foreach (int idx in indices)
                {
                    detections.Add(new Detection
                    {
                        ClassId = classIds[idx],
                        ClassName = CocoClassNames[classIds[idx]],
                        Confidence = scores[idx],
                        X = boxes[idx].X,
                        Y = boxes[idx].Y,
                        Width = boxes[idx].Width,
                        Height = boxes[idx].Height
                    });
                }
            }

            return detections;
        }

        private List<int> NMS(List<Rect> boxes, List<float> scores, float iouThreshold)
        {
            var indices = Enumerable.Range(0, scores.Count)
                .OrderByDescending(i => scores[i])
                .ToList();

            var keep = new List<int>();
            var suppressed = new HashSet<int>();

            foreach (int i in indices)
            {
                if (suppressed.Contains(i)) continue;
                keep.Add(i);

                for (int j = indices.IndexOf(i) + 1; j < indices.Count; j++)
                {
                    int jIdx = indices[j];
                    if (suppressed.Contains(jIdx)) continue;

                    float iou = ComputeIoU(boxes[i], boxes[jIdx]);
                    if (iou > iouThreshold)
                    {
                        suppressed.Add(jIdx);
                    }
                }
            }

            return keep;
        }

        private float ComputeIoU(Rect a, Rect b)
        {
            int x1 = Math.Max(a.X, b.X);
            int y1 = Math.Max(a.Y, b.Y);
            int x2 = Math.Min(a.X + a.Width, b.X + b.Width);
            int y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

            int interW = Math.Max(0, x2 - x1);
            int interH = Math.Max(0, y2 - y1);
            float inter = interW * interH;

            float areaA = a.Width * a.Height;
            float areaB = b.Width * b.Height;
            float union = areaA + areaB - inter;

            return union > 0 ? inter / union : 0;
        }

        public void Dispose()
        {
            if (_session != null) _session.Dispose();
        }
    }
}

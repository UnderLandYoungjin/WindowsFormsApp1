//MainForm.cs

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace WindowsFormsApp1
{
    public class MainForm : Form
    {
        // UI 컨트롤
        private PictureBox _pictureBox;
        private Label _lblFps;
        private Label _lblStatus;
        private Label _lblDetections;
        private Button _btnStart;
        private Button _btnStop;
        private Button _btnSelectModel;
        private TrackBar _tbConfidence;
        private Label _lblConfValue;
        private ComboBox _cbCameraIndex;
        private CheckBox _chkDetection;

        // 카메라 + 탐지
        private VideoCapture _capture;
        private YoloV8Detector _detector;
        private Thread _captureThread;
        private volatile bool _isRunning;
        private string _modelPath = "yolov8n.onnx";
        private float _confThreshold = 0.5f;
        private int _cameraIndex = 0;
        private bool _enableDetection = true;

        public MainForm()
        {
            InitializeComponents();
        }

        private void InitializeComponents()
        {
            this.Text = "YOLOv8 USB Camera Detection";
            this.Size = new System.Drawing.Size(1100, 720);
            this.MinimumSize = new System.Drawing.Size(900, 600);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(30, 30, 30);
            this.ForeColor = Color.White;

            // ── 좌측: 카메라 영상 ──
            _pictureBox = new PictureBox
            {
                Location = new System.Drawing.Point(10, 10),
                Size = new System.Drawing.Size(800, 600),
                BackColor = Color.Black,
                SizeMode = PictureBoxSizeMode.Zoom,
                BorderStyle = BorderStyle.FixedSingle
            };

            // ── 우측 패널 ──
            int panelX = 820;

            var lblCamera = new Label
            {
                Text = "Camera:",
                Location = new System.Drawing.Point(panelX, 15),
                AutoSize = true
            };

            _cbCameraIndex = new ComboBox
            {
                Location = new System.Drawing.Point(panelX, 35),
                Size = new System.Drawing.Size(250, 25),
                DropDownStyle = ComboBoxStyle.DropDownList,
                BackColor = Color.FromArgb(50, 50, 50),
                ForeColor = Color.White
            };
            for (int i = 0; i < 5; i++) _cbCameraIndex.Items.Add("Camera " + i);
            _cbCameraIndex.SelectedIndex = 0;
            _cbCameraIndex.SelectedIndexChanged += delegate { _cameraIndex = _cbCameraIndex.SelectedIndex; };

            _btnSelectModel = new Button
            {
                Text = "Select ONNX Model",
                Location = new System.Drawing.Point(panelX, 75),
                Size = new System.Drawing.Size(250, 35),
                FlatStyle = FlatStyle.Flat,
                BackColor = Color.FromArgb(60, 60, 60)
            };
            _btnSelectModel.Click += BtnSelectModel_Click;

            var lblConf = new Label
            {
                Text = "Confidence Threshold:",
                Location = new System.Drawing.Point(panelX, 125),
                AutoSize = true
            };

            _tbConfidence = new TrackBar
            {
                Location = new System.Drawing.Point(panelX, 145),
                Size = new System.Drawing.Size(200, 30),
                Minimum = 10,
                Maximum = 95,
                Value = 50,
                TickFrequency = 10,
                BackColor = Color.FromArgb(30, 30, 30)
            };
            _tbConfidence.ValueChanged += delegate
            {
                _confThreshold = _tbConfidence.Value / 100f;
                _lblConfValue.Text = _confThreshold.ToString("F2");
            };

            _lblConfValue = new Label
            {
                Text = "0.50",
                Location = new System.Drawing.Point(panelX + 205, 150),
                AutoSize = true
            };

            _chkDetection = new CheckBox
            {
                Text = "Enable Detection",
                Location = new System.Drawing.Point(panelX, 185),
                AutoSize = true,
                Checked = true,
                ForeColor = Color.LightGreen
            };
            _chkDetection.CheckedChanged += delegate
            {
                _enableDetection = _chkDetection.Checked;
                _chkDetection.ForeColor = _enableDetection ? Color.LightGreen : Color.Gray;
            };

            _btnStart = new Button
            {
                Text = "▶ Start",
                Location = new System.Drawing.Point(panelX, 225),
                Size = new System.Drawing.Size(120, 40),
                FlatStyle = FlatStyle.Flat,
                BackColor = Color.FromArgb(0, 120, 0),
                Font = new Font("Segoe UI", 11, FontStyle.Bold)
            };
            _btnStart.Click += BtnStart_Click;

            _btnStop = new Button
            {
                Text = "■ Stop",
                Location = new System.Drawing.Point(panelX + 130, 225),
                Size = new System.Drawing.Size(120, 40),
                FlatStyle = FlatStyle.Flat,
                BackColor = Color.FromArgb(150, 0, 0),
                Enabled = false,
                Font = new Font("Segoe UI", 11, FontStyle.Bold)
            };
            _btnStop.Click += BtnStop_Click;

            _lblFps = new Label
            {
                Text = "FPS: --",
                Location = new System.Drawing.Point(panelX, 290),
                AutoSize = true,
                Font = new Font("Consolas", 14, FontStyle.Bold),
                ForeColor = Color.Cyan
            };

            _lblStatus = new Label
            {
                Text = "Status: Ready",
                Location = new System.Drawing.Point(panelX, 320),
                AutoSize = true,
                Font = new Font("Segoe UI", 10),
                ForeColor = Color.LightGray
            };

            _lblDetections = new Label
            {
                Text = "Detections:\r\n(none)",
                Location = new System.Drawing.Point(panelX, 355),
                Size = new System.Drawing.Size(260, 250),
                Font = new Font("Consolas", 9),
                ForeColor = Color.LightYellow
            };

            this.Controls.AddRange(new Control[]
            {
                _pictureBox, lblCamera, _cbCameraIndex,
                _btnSelectModel, lblConf, _tbConfidence, _lblConfValue,
                _chkDetection, _btnStart, _btnStop,
                _lblFps, _lblStatus, _lblDetections
            });
        }

        private void BtnSelectModel_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "ONNX Model|*.onnx";
                ofd.Title = "Select YOLOv8 ONNX Model";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    _modelPath = ofd.FileName;
                    _lblStatus.Text = "Model: " + Path.GetFileName(_modelPath);
                }
            }
        }

        private void BtnStart_Click(object sender, EventArgs e)
        {
            if (_isRunning) return;

            try
            {
                if (!File.Exists(_modelPath))
                {
                    MessageBox.Show(
                        "ONNX model not found: " + _modelPath + "\r\n\r\n" +
                        "Please export YOLOv8 model:\r\n" +
                        "pip install ultralytics\r\n" +
                        "yolo export model=yolov8n.pt format=onnx imgsz=640",
                        "Model Not Found", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                _lblStatus.Text = "Loading model...";
                _lblStatus.Refresh();

                if (_detector != null)
                {
                    _detector.Dispose();
                    _detector = null;
                }
                _detector = new YoloV8Detector(_modelPath, _confThreshold);

                _lblStatus.Text = "Opening camera...";
                _lblStatus.Refresh();

                _capture = new VideoCapture(_cameraIndex);
                _capture.Set(VideoCaptureProperties.FrameWidth, 640);
                _capture.Set(VideoCaptureProperties.FrameHeight, 480);

                if (!_capture.IsOpened())
                {
                    MessageBox.Show("Cannot open Camera " + _cameraIndex + ".\r\nPlease check USB camera connection.",
                        "Camera Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                _isRunning = true;
                _btnStart.Enabled = false;
                _btnStop.Enabled = true;
                _cbCameraIndex.Enabled = false;
                _lblStatus.Text = "Running...";

                _captureThread = new Thread(CaptureLoop);
                _captureThread.IsBackground = true;
                _captureThread.Name = "CameraCapture";
                _captureThread.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error: " + ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                StopCapture();
            }
        }

        private void BtnStop_Click(object sender, EventArgs e)
        {
            StopCapture();
        }

        private void CaptureLoop()
        {
            var sw = new Stopwatch();
            var frame = new Mat();
            double fps = 0;
            int frameCount = 0;
            var fpsTimer = Stopwatch.StartNew();

            while (_isRunning)
            {
                try
                {
                    sw.Restart();

                    if (_capture == null || !_capture.IsOpened()) break;
                    if (!_capture.Read(frame) || frame.Empty()) continue;

                    // ✅ 좌우 반전(거울모드)
                    Cv2.Flip(frame, frame, FlipMode.Y);

                    List<Detection> detections = new List<Detection>();

                    if (_enableDetection && _detector != null)
                    {
                        detections = _detector.Detect(frame);
                        _detector.DrawDetections(frame, detections);
                    }

                    frameCount++;
                    if (fpsTimer.ElapsedMilliseconds >= 1000)
                    {
                        fps = frameCount * 1000.0 / fpsTimer.ElapsedMilliseconds;
                        frameCount = 0;
                        fpsTimer.Restart();
                    }

                    Cv2.PutText(frame, string.Format("FPS: {0:F1}", fps), new OpenCvSharp.Point(10, 30),
                        HersheyFonts.HersheySimplex, 1.0, new Scalar(0, 255, 0), 2);

                    var bitmap = BitmapConverter.ToBitmap(frame);

                    var capturedDetections = detections;
                    var capturedFps = fps;

                    try
                    {
                        this.BeginInvoke(new Action(delegate
                        {
                            var old = _pictureBox.Image;
                            _pictureBox.Image = bitmap;
                            if (old != null) old.Dispose();

                            _lblFps.Text = string.Format("FPS: {0:F1}", capturedFps);

                            if (capturedDetections.Count > 0)
                            {
                                var detText = string.Join("\r\n",
                                    capturedDetections.Take(10).Select(d => string.Format("* {0} ({1:P0})", d.ClassName, d.Confidence)));
                                if (capturedDetections.Count > 10)
                                    detText += string.Format("\r\n... +{0} more", capturedDetections.Count - 10);
                                _lblDetections.Text = string.Format("Detections ({0}):\r\n{1}", capturedDetections.Count, detText);
                            }
                            else
                            {
                                _lblDetections.Text = "Detections:\r\n(none)";
                            }
                        }));
                    }
                    catch (ObjectDisposedException)
                    {
                        break;
                    }
                    catch (InvalidOperationException)
                    {
                        break;
                    }

                    sw.Stop();
                    int delay = Math.Max(1, 33 - (int)sw.ElapsedMilliseconds);
                    Thread.Sleep(delay);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine("Capture error: " + ex.Message);
                }
            }

            frame.Dispose();
        }

        private void StopCapture()
        {
            _isRunning = false;
            if (_captureThread != null) _captureThread.Join(2000);

            if (_capture != null)
            {
                _capture.Release();
                _capture.Dispose();
                _capture = null;
            }

            _btnStart.Enabled = true;
            _btnStop.Enabled = false;
            _cbCameraIndex.Enabled = true;
            _lblStatus.Text = "Status: Stopped";
            _lblFps.Text = "FPS: --";
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            StopCapture();
            if (_detector != null) _detector.Dispose();
            base.OnFormClosing(e);
        }
    }
}

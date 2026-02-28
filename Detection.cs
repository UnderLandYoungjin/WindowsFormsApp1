using System;

namespace WindowsFormsApp1
{
    public class Detection
    {
        public int ClassId { get; set; }
        public string ClassName { get; set; }
        public float Confidence { get; set; }
        public float X { get; set; }
        public float Y { get; set; }
        public float Width { get; set; }
        public float Height { get; set; }

        public Detection()
        {
            ClassName = string.Empty;
        }

        public override string ToString()
        {
            return string.Format("{0} ({1:P1}) [{2:F0}, {3:F0}, {4:F0}, {5:F0}]",
                ClassName, Confidence, X, Y, Width, Height);
        }
    }
}

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace BK_Masker
{
    internal class ReviewLogManager
    {
        private const string _logName = "reviewLog.txt";
        private List<string> _reviewLog = new List<string>();
        private readonly string _logPath; // TODO: make init only

        public ReviewLogManager(string folderPath)
        {
            if (string.IsNullOrEmpty(folderPath)) throw new ArgumentNullException(nameof(folderPath));
            if (!Directory.Exists(folderPath)) throw new ArgumentException($"Log folder path does not exist: [{folderPath}]");
            _logPath = Path.Combine(folderPath, _logName);
            lock(_logPath)
            {
                _reviewLog = new FileManager().ReadFileNames(_logPath);
            }
        }


        async public void AddImage(string imagePath)
        {
            string imageName = Path.GetFileName(imagePath);
            if (!_reviewLog.Contains(imageName))
            {
                _reviewLog.Add(imageName);

                try
                {
                    await Task.Run(() => WriteToLog(_reviewLog));
                }
                catch (Exception ex)
                {
                    // TODO: Notify user
                    Debug.WriteLine($"Failed to save the review log: [{ex.Message}]");
                }
            }
        }

        async public void RemoveImage(string imagePath)
        {
            string imageName = Path.GetFileName(imagePath);
            if (_reviewLog.Contains(imageName))
            {
                _reviewLog.Remove(imageName);

                try
                {
                    await Task.Run(() => WriteToLog(_reviewLog));
                }
                catch (Exception ex)
                {
                    // TODO: Notify user
                    Debug.WriteLine($"Failed to save the review log: [{ex.Message}]");
                }
            }
        }

        public bool IsInLog(string imagePath)=>
            _reviewLog.Contains(Path.GetFileName(imagePath));

        private async Task? WriteToLog(List<string> ReviewLog)
        {
            lock (_logPath)
            {
                new FileManager().WriteFileNames(_logPath, ReviewLog);
            }
        }
    }
}

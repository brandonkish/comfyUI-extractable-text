using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows;
using Microsoft.Win32;

namespace AI_Toolkit_log_parser
{
    public partial class MainWindow : Window
    {
        // This list will hold the parsed safetensor data
        private List<SafetensorData> safetensors = new List<SafetensorData>();

        public MainWindow()
        {
            InitializeComponent();
        }

        // Struct to hold the safetensor's name and loss value
        public struct SafetensorData
        {
            public string Name { get; set; }
            public double Loss { get; set; }

            public SafetensorData(string name, double loss)
            {
                Name = name;
                Loss = loss;
            }

            public override string ToString()
            {
                return $"{Name} - {Loss:F6}";
            }
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            // Open a file dialog to select the log file
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*",
                Title = "Select Log File"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                // Parse the selected log file
                ParseLog(openFileDialog.FileName);
                // Display the parsed data
                DisplayResults();
            }
        }

        private void ParseLog(string logFilePath)
        {
            // Clear the existing data before parsing the new log file
            safetensors.Clear();

            // Regex patterns for extracting loss values and safetensor file names
            string lossPattern = @"loss:\s([0-9\.e\-]+)";
            string safetensorPattern = @"Saved checkpoint to .+\\(.+\.safetensors)";

            // Read the log file line by line
            var lines = File.ReadAllLines(logFilePath);
            SafetensorData currentData = default(SafetensorData);

            foreach (var line in lines)
            {
                // Check for the loss value in the line
                var lossMatch = Regex.Match(line, lossPattern);
                if (lossMatch.Success)
                {
                    double lossValue = double.Parse(lossMatch.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
                    currentData.Loss = lossValue; // Update loss value
                }

                // Check for safetensor save entries
                var safetensorMatch = Regex.Match(line, safetensorPattern);
                if (safetensorMatch.Success)
                {
                    string safetensorName = safetensorMatch.Groups[1].Value;
                    safetensors.Add(new SafetensorData(safetensorName, currentData.Loss));
                }
            }

            // Sort the safetensor list by the loss value (ascending)
            safetensors = safetensors.OrderBy(s => s.Loss).ToList();
        }

        private void DisplayResults()
        {
            // Bind the results to the DataGrid
            ResultsDataGrid.ItemsSource = safetensors;
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            // Open a SaveFileDialog to save the results
            SaveFileDialog saveFileDialog = new SaveFileDialog
            {
                Filter = "Text Files (*.txt)|*.txt",
                FileName = "SafetensorList.txt"
            };

            if (saveFileDialog.ShowDialog() == true)
            {
                // Save the safetensor data to a text file
                File.WriteAllLines(saveFileDialog.FileName, safetensors.Select(s => s.ToString()));
                MessageBox.Show("List saved successfully!");
            }
        }
    }
}

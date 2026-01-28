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
        private List<SafetensorData> safetensors = new List<SafetensorData>();
        private List<SafetensorData> filteredSafetensors = new List<SafetensorData>();

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

        private void DisplayResults()
        {
            // Set the filtered list initially to display all safetensors
            filteredSafetensors = new List<SafetensorData>(safetensors);
            ResultsDataGrid.ItemsSource = filteredSafetensors;

            // Update the statistics
            UpdateStatistics();
        }

        private void UpdateStatistics()
        {
            if (safetensors.Count > 0)
            {
                int count = safetensors.Count;
                double maxLoss = safetensors.Max(s => s.Loss);
                double minLoss = safetensors.Min(s => s.Loss);

                StatsTextBlock.Text = $"Entries: {count} | Max Loss: {maxLoss:F6} | Min Loss: {minLoss:F6}";
            }
            else
            {
                StatsTextBlock.Text = "No data loaded.";
            }
        }


        // Event handler for the Load Button click
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

        // Event handler for drag enter
        private void Window_DragEnter(object sender, System.Windows.DragEventArgs e)
        {
            // Check if the data being dragged is a file
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = System.Windows.DragDropEffects.Copy;
            }
            else
            {
                e.Effects = System.Windows.DragDropEffects.None;
            }
        }

        // Event handler for file drop
        // Event handler for file/folder drop
        private void Window_Drop(object sender, System.Windows.DragEventArgs e)
        {
            // Retrieve the dropped item paths
            string[] droppedItems = (string[])e.Data.GetData(DataFormats.FileDrop);

            if (droppedItems.Length > 0)
            {
                // Reset the lists and DataGrid before loading the new log(s)
                safetensors.Clear();
                filteredSafetensors.Clear();
                ResultsDataGrid.ItemsSource = null; // Clear the DataGrid

                // Check if the dropped item is a directory (folder)
                var firstItem = droppedItems[0];
                if (Directory.Exists(firstItem))
                {
                    // If it's a folder, scan the folder and subfolders for "log.txt" files
                    var logFiles = Directory.GetFiles(firstItem, "log.txt", SearchOption.AllDirectories);

                    foreach (var logFile in logFiles)
                    {
                        // Parse each log file found in the folder and subfolders
                        ParseLog(logFile);
                    }
                }
                else
                {
                    // If it's a single file, just parse that one log file
                    ParseLog(firstItem);
                }

                // Display the parsed data
                DisplayResults();
            }
        }


        private void ParseLog(string logFilePath)
        {
            // Clear the existing data before parsing the new log file
            safetensors.Clear();

            string lossPattern = @"loss:\s([0-9\.e\-]+)";
            string safetensorPattern = @"Saved checkpoint to .+\\(.+\.safetensors)";

            var lines = File.ReadAllLines(logFilePath);
            SafetensorData currentData = default(SafetensorData);
            var isSaftensorsMatchFound = false;

            foreach (var line in lines)
            {
                var safetensorMatch = Regex.Match(line, safetensorPattern);
                if (safetensorMatch.Success)
                {
                    isSaftensorsMatchFound = true;
                    string safetensorName = safetensorMatch.Groups[1].Value;
                    currentData.Name = safetensorName;


                }

                // Check for the loss value in the line
                var lossMatch = Regex.Match(line, lossPattern);
                if (lossMatch.Success && isSaftensorsMatchFound)
                {
                    double lossValue = double.Parse(lossMatch.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
                    currentData.Loss = lossValue; // Update loss value
                    safetensors.Add(new SafetensorData(currentData.Name, currentData.Loss));
                    isSaftensorsMatchFound = false;
                }

                // Check for safetensor save entries

            }

            // Sort the safetensor list by the loss value (ascending)
            safetensors = safetensors.OrderBy(s => s.Loss).ToList();
        }


        // Event for TextBox to filter results as the user types
        private void SearchBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
        {
            string searchText = SearchBox.Text.ToLower();

            // Filter the list of safetensors by matching the name
            filteredSafetensors = safetensors.Where(s => s.Name.ToLower().Contains(searchText)).ToList();

            // Update the DataGrid to display the filtered results
            ResultsDataGrid.ItemsSource = filteredSafetensors;
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
                File.WriteAllLines(saveFileDialog.FileName, filteredSafetensors.Select(s => s.ToString()));
                MessageBox.Show("List saved successfully!");
            
            }
        }
    }
}

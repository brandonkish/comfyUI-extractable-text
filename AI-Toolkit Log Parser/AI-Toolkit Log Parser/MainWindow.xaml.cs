using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Data;

namespace AI_Toolkit_Log_Parser
{
    public partial class MainWindow : Window
    {
        private ObservableCollection<LogEntry> logEntries;
        private ListCollectionView sortedLogEntries;

        public MainWindow()
        {
            InitializeComponent();
            logEntries = new ObservableCollection<LogEntry>();
            sortedLogEntries = new ListCollectionView(logEntries);
            LogListView.ItemsSource = sortedLogEntries;  // Binding to ListCollectionView
        }

        private void LoadLogButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "Log Files (*.log)|*.log|All Files (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                // Parse the log file
                var parser = new LogParser();
                var parsedEntries = parser.ParseLog(openFileDialog.FileName);

                // Clear the existing entries and add the new ones
                logEntries.Clear();
                foreach (var entry in parsedEntries)
                {
                    logEntries.Add(entry);
                }

                // Automatically sort the log by Loss after loading the log
                SortLog("Loss");
            }
        }

        private void SaveListButton_Click(object sender, RoutedEventArgs e)
        {
            var saveFileDialog = new SaveFileDialog
            {
                Filter = "Text Files (*.txt)|*.txt|All Files (*.*)|*.*"
            };

            if (saveFileDialog.ShowDialog() == true)
            {
                var filePath = saveFileDialog.FileName;

                // Save the list to a file
                var lines = new List<string>();
                foreach (var entry in logEntries)
                {
                    lines.Add($"{entry.FileName} - Loss: {entry.Loss} - Step: {entry.Step}");
                }

                File.WriteAllLines(filePath, lines);
            }
        }

        private void SortLog(string sortBy)
        {
            // Handle sorting by Loss or Step
            if (sortBy == "Loss")
            {
                sortedLogEntries.SortDescriptions.Clear();
                sortedLogEntries.SortDescriptions.Add(new SortDescription("Loss", ListSortDirection.Ascending));
            }
            else if (sortBy == "Step")
            {
                sortedLogEntries.SortDescriptions.Clear();
                sortedLogEntries.SortDescriptions.Add(new SortDescription("Step", ListSortDirection.Ascending));
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace BK_Caption_Parser
{
    public partial class MainWindow : Window
    {
        private string currentFolder;
        private List<ImageSet> imageSets;
        private Dictionary<string, (string originalImage, string captionText, string captionImage)> fileHashes;

        public MainWindow()
        {
            InitializeComponent();
            imageSets = new List<ImageSet>();
            fileHashes = new Dictionary<string, (string, string, string)>();
        }

        // Handle drag-and-drop for the window
        private void Window_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] droppedFiles = (string[])e.Data.GetData(DataFormats.FileDrop);
                string droppedFile = droppedFiles.FirstOrDefault();

                if (!string.IsNullOrEmpty(droppedFile))
                {
                    string folderPath = Path.GetDirectoryName(droppedFile);
                    currentFolder = folderPath;
                    ParseFolder(folderPath);
                }
            }
        }

        // Parse folder for image sets and display in ListView
        private void ParseFolder(string folderPath)
        {
            var textFiles = Directory.GetFiles(folderPath, "*.txt");
            var allFiles = Directory.GetFiles(folderPath);

            imageSets.Clear(); // Clear previous data
            fileHashes.Clear();

            foreach (var textFile in textFiles)
            {
                string baseName = Path.GetFileNameWithoutExtension(textFile);
                string originalImage = allFiles.FirstOrDefault(f => Path.GetFileNameWithoutExtension(f) == baseName && f.EndsWith(".png"));
                string captionImage = allFiles.FirstOrDefault(f => Path.GetFileNameWithoutExtension(f).StartsWith(baseName) && f.EndsWith(".png"));

                if (originalImage != null && captionImage != null)
                {
                    string captionText = File.Exists(textFile) ? File.ReadAllText(textFile) : "No caption available";

                    // Add the image set to the list
                    imageSets.Add(new ImageSet
                    {
                        OriginalImage = new BitmapImage(new Uri(originalImage)),
                        CaptionImage = new BitmapImage(new Uri(captionImage)),
                        CaptionText = captionText
                    });

                    // Store hashes for future processing
                    fileHashes[baseName] = (originalImage, textFile, captionImage);
                }
            }

            // Display in ListView
            SetsListView.ItemsSource = imageSets;
        }

        // Click to expand a row in the list to "enlarged" view
        private void SetRow_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (SetsListView.SelectedItem is ImageSet selectedSet)
            {
                EnlargedViewPanel.Visibility = Visibility.Visible;

                OriginalImageEnlarged.Source = selectedSet.OriginalImage;
                CaptionImageEnlarged.Source = selectedSet.CaptionImage;
                CaptionTextBoxEnlarged.Text = selectedSet.CaptionText;
            }
        }

        // Delete caption (just removes the caption image and text)
        private void DeleteCaptionButton_Click(object sender, RoutedEventArgs e)
        {
            string baseName = Path.GetFileNameWithoutExtension(OriginalImageEnlarged.Source.ToString());
            string captionImage = fileHashes[baseName].captionImage;
            string captionTextFile = fileHashes[baseName].captionText;

            var result = MessageBox.Show("Are you sure you want to delete the caption?", "Delete Caption", MessageBoxButton.YesNo);
            if (result == MessageBoxResult.Yes)
            {
                // Delete caption files
                if (File.Exists(captionImage)) File.Delete(captionImage);
                if (File.Exists(captionTextFile)) File.Delete(captionTextFile);

                // Update the UI (set to "No Caption")
                CaptionImageEnlarged.Source = null;
                CaptionTextBoxEnlarged.Text = "No Caption Found";
            }
        }

        // Delete all (deletes the original, caption, and text files)
        private void DeleteAllButton_Click(object sender, RoutedEventArgs e)
        {
            string baseName = Path.GetFileNameWithoutExtension(OriginalImageEnlarged.Source.ToString());
            string originalImage = fileHashes[baseName].originalImage;
            string captionImage = fileHashes[baseName].captionImage;
            string captionTextFile = fileHashes[baseName].captionText;

            var result = MessageBox.Show("Are you sure you want to delete all files?", "Delete All", MessageBoxButton.YesNo);
            if (result == MessageBoxResult.Yes)
            {
                // Delete all files in the set
                if (File.Exists(originalImage)) File.Delete(originalImage);
                if (File.Exists(captionImage)) File.Delete(captionImage);
                if (File.Exists(captionTextFile)) File.Delete(captionTextFile);

                // Remove from dictionary and ListView
                fileHashes.Remove(baseName);
                imageSets.RemoveAll(x => Path.GetFileNameWithoutExtension(x.OriginalImage.UriSource.AbsolutePath) == baseName);

                // Refresh ListView
                SetsListView.ItemsSource = null;
                SetsListView.ItemsSource = imageSets;
            }
        }

        // Save caption text
        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            string baseName = Path.GetFileNameWithoutExtension(OriginalImageEnlarged.Source.ToString());
            string captionTextFile = fileHashes[baseName].captionText;

            // Allow user to edit caption text
            string newCaption = CaptionTextBoxEnlarged.Text;
            File.WriteAllText(captionTextFile, newCaption);

            // Update the caption in memory
            var selectedSet = imageSets.FirstOrDefault(x => Path.GetFileNameWithoutExtension(x.OriginalImage.UriSource.AbsolutePath) == baseName);
            if (selectedSet != null)
            {
                selectedSet.CaptionText = newCaption;
            }

            MessageBox.Show("Caption saved successfully!", "Save Caption", MessageBoxButton.OK);
        }
    }

    public class ImageSet
    {
        public BitmapImage OriginalImage { get; set; }
        public BitmapImage CaptionImage { get; set; }
        public string CaptionText { get; set; }
    }
}

using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;

namespace BK_Caption_Parser
{
    public partial class MainWindow : Window
    {
        public ObservableCollection<ImageSet> ImageSets { get; } = new();
        public ImageSet? SelectedSet { get; set; }
        string? currentFolder;

        public MainWindow()
        {
            InitializeComponent();
            DataContext = this;
        }

        private void Window_Drop(object sender, DragEventArgs e)
        {
            if (!e.Data.GetDataPresent(DataFormats.FileDrop)) return;

            var files = (string[])e.Data.GetData(DataFormats.FileDrop);
            currentFolder = Path.GetDirectoryName(files[0]);

            ReloadFolder();
        }

        void ReloadFolder()
        {
            ImageSets.Clear();
            foreach (var set in FolderParser.Parse(currentFolder))
            {
                // Check if an .updatecap file exists and load its content if available
                var baseName = Path.GetFileNameWithoutExtension(set.CaptionTextPath);
                var updateCapFilePath = Path.Combine(Path.GetDirectoryName(set.CaptionTextPath), baseName + ".updatecap");

                if (File.Exists(updateCapFilePath))
                {
                    // Load the .updatecap file content into the CaptionChangeText property
                    set.CaptionChangeText = File.ReadAllText(updateCapFilePath);
                }
                else
                {
                    // If no .updatecap file, set the text to empty
                    set.CaptionChangeText = string.Empty;
                }

                ImageSets.Add(set);
            }
        }

        private void Delete_Click(object sender, RoutedEventArgs e)
        {
            // Get the selected ImageSet associated with the current button click
            var button = sender as Button;
            var imageSet = button?.DataContext as ImageSet;

            if (imageSet == null) return;

            // Confirm delete action
            var confirmResult = MessageBox.Show("Are you sure you want to delete all files for this set?",
                                                 "Confirm Delete",
                                                 MessageBoxButton.YesNo,
                                                 MessageBoxImage.Warning);
            if (confirmResult == MessageBoxResult.No)
                return;

            try
            {
                // Clear the ImageSource bindings before deletion to avoid file locks
                ClearImageSourceBindings(imageSet);

                // Delete associated files
                DeleteFileIfExists(imageSet.CaptionTextPath);
                DeleteFileIfExists(imageSet.OriginalImagePath);
                DeleteFileIfExists(imageSet.CaptionImagePath);

                // Delete the .updatecap file if exists
                var updateCapFilePath = Path.ChangeExtension(imageSet.CaptionTextPath, ".updatecap");
                DeleteFileIfExists(updateCapFilePath);

                // Remove the image set from the collection and return to the list
                ImageSets.Remove(imageSet);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error deleting files: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }

            // Renumber last set to deleted set number
            var last = ImageSets[^1];
            if (last != SelectedSet)
            {
                RenameSet(last, SelectedSet.SetNumber);
            }
        }

        private void ClearImageSourceBindings(ImageSet imageSet)
        {
            // Clear the ImageSource references to avoid file locks
            if (imageSet.OriginalImage != null)
            {
                // Nullifying the image source to release the file lock
                imageSet.OriginalImage = null;
            }

            if (imageSet.CaptionImage != null)
            {
                // Nullifying the caption image source to release the file lock
                imageSet.CaptionImage = null;
            }

            Debug.WriteLine("Cleared image references.");
        }


        void RenameSet(ImageSet set, int newNumber)
        {
            string newBase = Regex.Replace(
                Path.GetFileNameWithoutExtension(set.CaptionTextPath),
                @"\d+$", newNumber.ToString("D4"));

            foreach (var path in Directory.GetFiles(set.FolderPath)
                         .Where(p => p.Contains(set.SetNumber.ToString())))
            {
                File.Move(path,
                    Path.Combine(set.FolderPath,
                    newBase + Path.GetExtension(path)));
            }
        }

        private void DeleteFileIfExists(string filePath)
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
                Debug.WriteLine($"Deleted: {filePath}");
            }
        }





        private void CaptionTextBox_LostFocus(object sender, RoutedEventArgs e)
        {
            var textBox = sender as TextBox;
            var imageSet = (textBox?.DataContext as ImageSet);  // Bind to ImageSet in DataContext

            if (imageSet == null) return;

            var baseName = Path.GetFileNameWithoutExtension(imageSet.CaptionTextPath);
            var updateCapFilePath = Path.Combine(Path.GetDirectoryName(imageSet.CaptionTextPath), baseName + ".updatecap");

            if (string.IsNullOrWhiteSpace(textBox.Text))
            {
                // If the TextBox is empty, delete the .updatecap file if it exists
                if (File.Exists(updateCapFilePath))
                {
                    File.Delete(updateCapFilePath);
                    Debug.WriteLine($"Deleted {updateCapFilePath}");
                }
            }
            else
            {
                // Otherwise, save the text to the .updatecap file
                File.WriteAllText(updateCapFilePath, textBox.Text, Encoding.UTF8);
                Debug.WriteLine($"Saved {updateCapFilePath} with text: {textBox.Text}");
            }
        }

    }
}
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Text.RegularExpressions;
using System.Windows;

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

        private void ReloadFolder()
        {

            ImageSets.Clear();
            foreach (var set in FolderParser.Parse(currentFolder))
            {
                // Use ImageLoader to load images for the set
                set.OriginalImage = ImageLoader.LoadImage(set.OriginalImagePath);
                set.CaptionImage = ImageLoader.LoadImage(set.CaptionImagePath);

                // Add the set to the collection
                ImageSets.Add(set);
            }
        }

        private void ImageSet_Clicked(object sender, RoutedEventArgs e)
        {
            SelectedSet = (sender as FrameworkElement)?.DataContext as ImageSet;
            ListViewGrid.Visibility = Visibility.Collapsed;
            EditViewGrid.Visibility = Visibility.Visible;
            DataContext = this;
        }

        private void Back_Click(object? sender, RoutedEventArgs? e)
        {
            SaveCaptionChange();
            EditViewGrid.Visibility = Visibility.Collapsed;
            ListViewGrid.Visibility = Visibility.Visible;
            ReloadFolder();
        }

        private void SaveCaptionChange()
        {
            if (!string.IsNullOrWhiteSpace(SelectedSet?.CaptionChangeText))
            {
                var baseName = Path.GetFileNameWithoutExtension(SelectedSet.CaptionTextPath);

                if(SelectedSet.FolderPath is null)
                {
                    Debug.WriteLine("SelectedSet.FolderPath is null when trying to SaveCaptionChange. Failed to save the Caption Change.");
                    return;
                }

                File.WriteAllText(
                    Path.Combine(SelectedSet.FolderPath, baseName + ".capchange"),
                    SelectedSet.CaptionChangeText);
            }
        }

        private void Delete_Click(object sender, RoutedEventArgs e)
        {
            if (MessageBox.Show(
                "Are you sure you want to delete all files including the original image?",
                "Confirm",
                MessageBoxButton.YesNo) != MessageBoxResult.Yes)
                return;

            if (SelectedSet is null)
            {
                Debug.WriteLine("SelectedSet is null when trying to Delete the files. Failed to delete set.");
                return;
            }

            foreach (var path in new[] {
                SelectedSet.CaptionTextPath,
                SelectedSet.OriginalImagePath,
                SelectedSet.CaptionImagePath
            })
            {
                if (File.Exists(path)) File.Delete(path);
            }

            // Renumber last set to deleted set number
            var last = ImageSets[^1];
            if (last != SelectedSet)
            {
                RenameSet(last, SelectedSet.SetNumber);
            }

            Back_Click(null, null);
        }

        private void RenameSet(ImageSet set, int newNumber)
        {
            if(set.CaptionTextPath is null)
            {
                Debug.WriteLine($"set.CaptionTextPath is null when trying to rename the set. newNumber [{newNumber}]");
                return;
            }

            string newBase = Regex.Replace(
                Path.GetFileNameWithoutExtension(set.CaptionTextPath),
                @"\d+$", newNumber.ToString("D4"));

            if (set.FolderPath is null)
            {
                Debug.WriteLine("set.FolderPath is null when trying to Rename the set. Failed to rename the set.");
                return;
            }

            foreach (var path in Directory.GetFiles(set.FolderPath)
                         .Where(p => p.Contains(set.SetNumber.ToString())))
            {
                File.Move(path,
                    Path.Combine(set.FolderPath,
                    newBase + Path.GetExtension(path)));
            }
        }
    }
}
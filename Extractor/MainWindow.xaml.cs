//using System.Windows.Shapes;
using Newtonsoft.Json.Linq;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace METAVACE
{
    public partial class MainWindow : Window
    {

        PngImage? PngImage = null;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void Window_DragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = DragDropEffects.Copy; // Or Move, Link, etc.
            }
            else
            {
                e.Effects = DragDropEffects.None;
            }

            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
                if (files.Any(f => Path.GetExtension(f).Equals(".png", StringComparison.OrdinalIgnoreCase)))
                {
                    e.Effects = DragDropEffects.Copy;
                }
            }

            e.Handled = true;
        }


        private void WidgetValue_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            var clickedText = (sender as TextBlock)?.Text;
            if (clickedText != null)
            {
                Clipboard.SetText(clickedText);
                MessageBox.Text = $"Copied To Clipboard: {clickedText}";
            }
        }



        private void Window_Drop(object sender, DragEventArgs e)
        {
            if (!e.Data.GetDataPresent(DataFormats.FileDrop))
                return;

            string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);

            
            IEnumerable<string> allPngFiles = files.Where(f => Path.GetExtension(f).Equals(".png", StringComparison.OrdinalIgnoreCase));

            foreach(string pngFile in allPngFiles)
            {
                if (pngFile is null) continue;

                try
                {
                    PngImage = new PngImage(pngFile);

                    if (string.IsNullOrEmpty(PngImage.Description))
                        continue;

                    Description.Text = PngImage.Description;
                    WriteDescriptionToFile(GetTextFilePathFromPng(pngFile), PngImage.Description);

                }
                catch (Exception ex)
                {
                    MessageBox.Text = $"Error: {ex.Message}";
                }

            }

        }

        private static string GetTextFilePathFromPng(string pngFile)
        {

            if (string.IsNullOrEmpty(GetPathOfTextFile(pngFile)))
                return string.Empty;

            return Path.Combine(GetPathOfTextFile(pngFile) ?? string.Empty, GetNameOfTextFile(pngFile));
        }

        private static string? GetPathOfTextFile(string pngFile)
        {
            return Path.GetDirectoryName(pngFile);
        }

        private static string GetNameOfTextFile(string pngFile)
        {
            return Path.GetFileNameWithoutExtension(pngFile) + ".txt";
        }

        private void WriteDescriptionToFile(string textFilePath, string Descripton)
        {
            using (StreamWriter sw = File.CreateText(textFilePath))
            {
                sw.WriteLine(RemoveDescriptionWordAndNullChar(Descripton));
            }
        }

        private static string RemoveDescriptionWordAndNullChar(string description)
        {
            return description.Substring(description.IndexOf('\0') + 1);
        }
    }
}
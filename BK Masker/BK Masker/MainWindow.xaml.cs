using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace BK_Masker
{
    public partial class MainWindow : Window
    {
        //TODO: Add to options
        private const double _minMaskSize = -1;
        private const string _maskedFolderName = "masked";

        private List<string> _imagePaths = new List<string>();
        private int _currentImageIndex = -1;
        private Rectangle? _selectionRectangle;
        private Point _canvasSelectionStartPoint;
        private Point _imageSelectionStartPoint;
        private bool _isLeftClickDown = false;
        private ReviewLogManager? _reviewLog;
        

        public MainWindow()
        {
            InitializeComponent();


        }



        private void MoveToNextImageOnClick()
        {
            if (_imagePaths.Count == 0) return;

            if (_reviewLog is not null)
                _reviewLog.AddImage(_imagePaths[_currentImageIndex]);

            // Advance to the next image
            if (_currentImageIndex < (_imagePaths.Count - 1))
                _currentImageIndex = _currentImageIndex + 1;

            DisplayImage(_imagePaths[_currentImageIndex]); // Update image and title
        }

        private void MoveToPreviousImageOnClick()
        {
            if (_imagePaths.Count == 0) return;

            if (_reviewLog is not null)
                _reviewLog.AddImage(_imagePaths[_currentImageIndex]);

            // Left click: Show previous image
            if (_currentImageIndex > 0)
            {
                _currentImageIndex--;
            }

            DisplayImage(_imagePaths[_currentImageIndex]); // Update image and title
        }


        private void Window_PreviewMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (e.MiddleButton == MouseButtonState.Released && _isLeftClickDown && _selectionRectangle != null)
            {
                // NOT USED
            }
        }

        private void TransparencyDragEnd(object sender, MouseButtonEventArgs e)
        {
            
            if (ImageDisplay.Source is null)
            {
                // Clean up the selection rectangle
                ImageCanvas.Children.Remove(_selectionRectangle);
                _selectionRectangle = null;
                return;
            }

            // 1️⃣ Get the actual size of the displayed image
            Point imageSelectionEndPoint = e.GetPosition(ImageDisplay);

            var boxLeft = Math.Min(_imageSelectionStartPoint.X, imageSelectionEndPoint.X);
            var boxTop = Math.Min(_imageSelectionStartPoint.Y, imageSelectionEndPoint.Y);
            var boxWidth = Math.Abs(_imageSelectionStartPoint.X - imageSelectionEndPoint.X);
            var boxHeight = Math.Abs(_imageSelectionStartPoint.Y - imageSelectionEndPoint.Y);

            if (boxWidth < 1) boxWidth = 1;
            if (boxHeight < 1) boxHeight = 1;

            // Clamp box
            boxLeft = Math.Max(0, boxLeft);
            boxTop = Math.Max(0, boxTop);
            boxWidth = Math.Min(boxLeft + ImageDisplay.Width, boxWidth);
            boxHeight = Math.Min(ImageDisplay.Width, boxHeight);

            if(boxHeight > _minMaskSize && boxWidth > _minMaskSize)
            {
                // Create the final rectangle for saving transparency
                Rect imageRect = new Rect(boxLeft, boxTop, boxWidth, boxHeight);

                // 6️⃣ Save the selection area with transparency
                string currentImagePath = _imagePaths[_currentImageIndex];
                string directoryPath = System.IO.Path.GetDirectoryName(currentImagePath) ?? System.IO.Directory.GetCurrentDirectory();
                string reviewedFolder = System.IO.Path.Combine(directoryPath, _maskedFolderName);
                System.IO.Directory.CreateDirectory(reviewedFolder);

                string fileName = System.IO.Path.GetFileName(currentImagePath);
                string savePath = System.IO.Path.Combine(reviewedFolder, fileName);

                SaveImageWithTransparency(currentImagePath, savePath, imageRect, ImageDisplay);
            }

            // 7️⃣ Move to the next image
            //_currentImageIndex = (_currentImageIndex + 1) % _imagePaths.Count;
            MoveToNextImageOnClick();

            // Clean up the selection rectangle
            ImageCanvas.Children.Remove(_selectionRectangle);
            _selectionRectangle = null;

        }

        private void Window_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            // Resize and center the image each time the window size changes
            if (ImageDisplay.Source != null)
            {
                BitmapImage bitmapImage = (BitmapImage)ImageDisplay.Source;
                ResizeAndCenterImage(bitmapImage);
            }
        }

        private void ResizeAndCenterImage(BitmapImage bitmapImage)
        {
            // Get the dimensions of the current window
            double windowWidth = this.ActualWidth;
            double windowHeight = this.ActualHeight;

            // Maintain the aspect ratio
            double imageWidth = bitmapImage.PixelWidth;
            double imageHeight = bitmapImage.PixelHeight;

            // Calculate the scaling factors for resizing
            double scaleX = windowWidth / imageWidth;
            double scaleY = windowHeight / imageHeight;

            // Choose the smaller scale factor to maintain the aspect ratio
            double scale = Math.Min(scaleX, scaleY);

            // Set the scaled image size
            ImageDisplay.Width = imageWidth * scale;
            ImageDisplay.Height = imageHeight * scale;

            // Center the image within the window
            Canvas.SetLeft(ImageDisplay, (windowWidth - ImageDisplay.Width) / 2);
            Canvas.SetTop(ImageDisplay, (windowHeight - ImageDisplay.Height) / 2);
        }



        private void Window_Drop(object sender, DragEventArgs e)
        {
            // Get the dropped files (string array)
            var files = (string[])e.Data.GetData(DataFormats.FileDrop);

            // Ensure that at least one file is dropped
            if (files != null && files.Length > 0)
            {
                string? draggedFile = files.FirstOrDefault();

                // Check if the file's parent directory exists
                if (draggedFile != null && Directory.Exists(System.IO.Path.GetDirectoryName(draggedFile)))
                {
                    string? folderPath = System.IO.Path.GetDirectoryName(draggedFile);
                    if (folderPath is not null)
                    {
                        _reviewLog = new ReviewLogManager(folderPath);
                        LoadImagesFromFolder(folderPath);
                        AddImagePathToStartOfList(draggedFile);
                        DisplayImage(draggedFile);
                    }
                }
            }

            // Mark the event as handled
            e.Handled = true;
        }

        private void AddImagePathToStartOfList(string draggedFile)
        {
            // Add dragged image to beginning of list at idx 0
            if (_imagePaths.Contains(draggedFile))
                _imagePaths.Remove(draggedFile);

            _imagePaths.Insert(0, draggedFile);
        }

        private void DisplayImage(string imagePath)
        {
            if (File.Exists(imagePath))
            {
                // Load the image source into the ImageDisplay control
                //BitmapImage bitmapImage = new BitmapImage(new Uri(imagePath));

                using (FileStream stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read))
                {
                    // Load image
                    BitmapImage bitmapImage = new BitmapImage();
                    bitmapImage.BeginInit();
                    bitmapImage.StreamSource = stream;
                    bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                    bitmapImage.EndInit();
                    bitmapImage.Freeze();

                    // Ensure BGRA32 so we have an alpha channel to overwrite
                    BitmapSource source = bitmapImage;
                    if (source.Format != PixelFormats.Bgra32)
                    {
                        source = new FormatConvertedBitmap(bitmapImage, PixelFormats.Bgra32, null, 0);
                        source.Freeze();
                    }

                    // Copy pixels
                    int bytesPerPixel = 4;
                    int stride = source.PixelWidth * bytesPerPixel;
                    byte[] pixels = new byte[stride * source.PixelHeight];
                    source.CopyPixels(pixels, stride, 0);

                    // Force alpha = 255 for every pixel
                    for (int i = 0; i < pixels.Length; i += bytesPerPixel)
                    {
                        pixels[i + 3] = 255; // Alpha channel
                    }

                    // Create new opaque bitmap
                    WriteableBitmap opaqueBitmap = new WriteableBitmap(
                        source.PixelWidth,
                        source.PixelHeight,
                        source.DpiX,
                        source.DpiY,
                        PixelFormats.Bgra32,
                        null);

                    opaqueBitmap.WritePixels(
                        new Int32Rect(0, 0, source.PixelWidth, source.PixelHeight),
                        pixels,
                        stride,
                        0);

                    ImageDisplay.Source = opaqueBitmap;
                    ResizeAndCenterImage(ConvertToBitmapImage(opaqueBitmap));
                }

                //ImageDisplay.Source = bitmapImage;

                // Resize and center the image within the window




                // Update the window title to show the image filename
                this.Title = $"BK Masker - {System.IO.Path.GetFileName(imagePath)} ({_currentImageIndex + 1} of {_imagePaths.Count})";
            }
        }

        BitmapImage ConvertToBitmapImage(BitmapSource source)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                BitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(source));
                encoder.Save(ms);

                ms.Position = 0;

                BitmapImage bmp = new BitmapImage();
                bmp.BeginInit();
                bmp.StreamSource = ms;
                bmp.CacheOption = BitmapCacheOption.OnLoad;
                bmp.EndInit();
                bmp.Freeze();

                return bmp;
            }
        }


        private void Window_PreviewDragOver(object sender, DragEventArgs e)
        {
            // Check if the dropped data is a file (FileDrop format)
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                // Set the drag effect to 'Copy' so the drop is allowed
                e.Effects = DragDropEffects.Copy;
            }
            else
            {
                // Set the effect to None if the dragged data is not a file
                e.Effects = DragDropEffects.None;
            }

            // Mark the event as handled
            e.Handled = true;
        }
        private void LoadImagesFromFolder(string folderPath)
        {
            _reviewLog = new ReviewLogManager(folderPath);
            var validExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".gif" };
            _imagePaths = Directory.GetFiles(folderPath)
                                    .Where(f => validExtensions.Contains(System.IO.Path.GetExtension(f).ToLower()))
                                    .Where(f => !_reviewLog.IsInLog(System.IO.Path.GetFileName(f)))
                                    .ToList();
            _imagePaths.Sort();
            _currentImageIndex = 0;
        }

        private void Window_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // Only execute if the Image canvas was clicked
            if (e.OriginalSource is UIElement element && (element == ImageCanvas || ImageCanvas.IsAncestorOf(element)))
            {
                _isLeftClickDown = true;
                TransparencyDragStart(sender, e);
            }
        }

        private void Window_PreviewMouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {

            if (e.OriginalSource is UIElement element && (element == ImageCanvas || ImageCanvas.IsAncestorOf(element)))
            {
                MoveToNextImageOnClick();
            }
            
        }
            

        private void MoveToNextImageAndSaveOnClick(object sender, MouseButtonEventArgs e)
        {
            if (_imagePaths.Count == 0) return;

            // Right click: Save current image to "masked" folder, then show next image
            string currentImagePath = _imagePaths[_currentImageIndex];
            string directoryPath = System.IO.Path.GetDirectoryName(currentImagePath) ?? Directory.GetCurrentDirectory();

            // Create cropped folder
            string croppedFolder = System.IO.Path.Combine(directoryPath, "cropped");
            if (!Directory.Exists(croppedFolder))
            {
                Directory.CreateDirectory(croppedFolder);
            }

            string fileName = System.IO.Path.GetFileName(currentImagePath);
            string savePath = System.IO.Path.Combine(croppedFolder, fileName);

            SaveImageAsPng(currentImagePath, savePath);
            MoveToNextImageOnClick();
        }



        private void SaveImageAsPng(string sourceImagePath, string targetPath)
        {
            BitmapImage bitmap = new BitmapImage(new Uri(sourceImagePath));

            // Convert the image to PNG with 0 compression
            PngBitmapEncoder encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(bitmap));

            using (FileStream fs = new FileStream(targetPath, FileMode.Create))
            {
                encoder.Save(fs);
            }
        }

        private void Window_PreviewMouseDown(object sender, MouseButtonEventArgs e)
        {
            // Check for middle mouse button (button 3)
            if (e.MiddleButton == MouseButtonState.Pressed)
            {
                // Only execute if the Image canvas was clicked
                if (e.OriginalSource is UIElement element && (element == ImageCanvas || ImageCanvas.IsAncestorOf(element)))
                {
                    MoveToPreviousImageOnClick();
                }
            }
        }

        private void TransparencyDragStart(object sender, MouseButtonEventArgs e)
        {
            _canvasSelectionStartPoint = e.GetPosition(ImageCanvas);
            _imageSelectionStartPoint = e.GetPosition(ImageDisplay);

            if (_selectionRectangle == null)
            {
                _selectionRectangle = new Rectangle
                {
                    Stroke = Brushes.Red,
                    StrokeThickness = 2,
                    Fill = new SolidColorBrush(Color.FromArgb(50, 255, 0, 0))
                };
                ImageCanvas.Children.Add(_selectionRectangle);
            }
        }

        private void Window_MouseMove(object sender, MouseEventArgs e)
        {
            if (_isLeftClickDown && _selectionRectangle != null)
            {
                Point currentPoint = e.GetPosition(ImageCanvas);

                double width = Math.Abs(currentPoint.X - _canvasSelectionStartPoint.X);
                double height = Math.Abs(currentPoint.Y - _canvasSelectionStartPoint.Y);

                Canvas.SetLeft(_selectionRectangle, Math.Min(currentPoint.X, _canvasSelectionStartPoint.X));
                Canvas.SetTop(_selectionRectangle, Math.Min(currentPoint.Y, _canvasSelectionStartPoint.Y));

                _selectionRectangle.Width = width;
                _selectionRectangle.Height = height;
            }
        }

        private void SaveImageWithTransparency(string sourceImagePath, string targetPath, System.Windows.Rect selectionArea, Image image)
        {
            // Load the original image
            System.Windows.Media.Imaging.BitmapImage bitmapImage = new System.Windows.Media.Imaging.BitmapImage(new Uri(sourceImagePath));

            // Convert the BitmapImage to a System.Drawing.Bitmap (with proper pixel format for transparency)
            using (System.IO.MemoryStream memoryStream = new System.IO.MemoryStream())
            {



                // Save BitmapImage to MemoryStream
                System.Windows.Media.Imaging.PngBitmapEncoder encoder = new System.Windows.Media.Imaging.PngBitmapEncoder();
                encoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(bitmapImage));
                encoder.Save(memoryStream);

                // Re-load as System.Drawing.Bitmap
                memoryStream.Seek(0, System.IO.SeekOrigin.Begin);
                System.Drawing.Bitmap originalBitmap = new System.Drawing.Bitmap(memoryStream);

                                // Convert the image to 32bpp ARGB (with alpha channel for transparency)
                System.Drawing.Bitmap convertedBitmap = originalBitmap.Clone(new System.Drawing.Rectangle(0, 0, originalBitmap.Width, originalBitmap.Height),
                    System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                // Get the conversion factor from displayed image coordinates to image resolution
                Vector imageToBitmapRatio = new Vector();
                imageToBitmapRatio.X = convertedBitmap.Width / image.Width;
                imageToBitmapRatio.Y = convertedBitmap.Height / image.Height;

                // Now, convert to a WriteableBitmap
                System.Windows.Media.Imaging.WriteableBitmap writableBitmap = new System.Windows.Media.Imaging.WriteableBitmap(convertedBitmap.Width, convertedBitmap.Height,
                    96, 96, System.Windows.Media.PixelFormats.Bgra32, null);

                // Copy the pixel data into the WriteableBitmap
                System.Drawing.Imaging.BitmapData bitmapData = convertedBitmap.LockBits(new System.Drawing.Rectangle(0, 0, convertedBitmap.Width, convertedBitmap.Height),
                    System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                IntPtr ptr = bitmapData.Scan0;
                int bytes = Math.Abs(bitmapData.Stride) * convertedBitmap.Height;
                byte[] pixels = new byte[bytes];
                System.Runtime.InteropServices.Marshal.Copy(ptr, pixels, 0, bytes);

                // Apply 100% transparency to the selected rectangle area (Rect)
                int startX = (int)(selectionArea.X * imageToBitmapRatio.X);
                int startY = (int)(selectionArea.Y * imageToBitmapRatio.Y);
                int endX = (int)((selectionArea.X + selectionArea.Width) * imageToBitmapRatio.X);
                int endY = (int)((selectionArea.Y + selectionArea.Height) * imageToBitmapRatio.Y);

                // Loop through all pixels and modify the ones inside the rectangle for transparency
                for (int y = 0; y < convertedBitmap.Height; y++)
                {
                    for (int x = 0; x < convertedBitmap.Width; x++)
                    {
                        int i = (y * convertedBitmap.Width + x) * 4; // Calculate index in the pixel array

                        // Check if the pixel is within the selection area
                        if (x >= startX && x <= endX && y >= startY && y <= endY)
                        {
                            // Apply 100% transparency (alpha = 0)
                            pixels[i + 3] = 0;  // Set alpha to 0 (100% transparent)
                        }
                        else
                        {
                            // Ensure alpha channel is set to 100% (100% transparency outside the selection area)
                            pixels[i + 3] = 255;  // set all other pixels to opaque
                        }
                    }
                }

                // Write the modified pixel data back to the WriteableBitmap
                writableBitmap.WritePixels(new System.Windows.Int32Rect(0, 0, writableBitmap.PixelWidth, writableBitmap.PixelHeight), pixels,
                    writableBitmap.PixelWidth * 4, 0);

                // Encode the modified WriteableBitmap as a PNG
                System.Windows.Media.Imaging.PngBitmapEncoder finalEncoder = new System.Windows.Media.Imaging.PngBitmapEncoder();
                finalEncoder.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(writableBitmap));

                // Save the modified image with transparency as PNG
                using (System.IO.FileStream fs = new System.IO.FileStream(targetPath, System.IO.FileMode.Create))
                {
                    finalEncoder.Save(fs);
                }

                // Clean up
                convertedBitmap.UnlockBits(bitmapData);
                originalBitmap.Dispose();
                convertedBitmap.Dispose();
            }
        }

        
        private void Window_PreviewMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            // Only execute if the Image canvas was clicked
            if (e.OriginalSource is UIElement element && (element == ImageCanvas || ImageCanvas.IsAncestorOf(element)))
            {
                _isLeftClickDown = true;
                TransparencyDragEnd(sender, e);
            }

        }

        private void DeleteBtn_Click(object sender, RoutedEventArgs e)
        {

            try
            {
                if(File.Exists(_imagePaths[_currentImageIndex]))
                    File.Delete(_imagePaths[_currentImageIndex]);
                _imagePaths.RemoveAt(_currentImageIndex);
                MoveToNextImageOnClick();
            }
            catch (Exception ex)
            {

                Debug.WriteLine($"Failed to delete image: [{_imagePaths[_currentImageIndex]}]:\n\t{ex.Message}");
            }

            

        }
    }
}

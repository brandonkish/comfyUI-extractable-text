using System;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace BK_Caption_Parser;
public class ImageLoader
{
    public static BitmapImage LoadImage(string? path)
    {
        if (path is null)
        {
            Debug.WriteLine("Path was null when trying to load an image.");
        }    

        // Check if the file exists
        if (File.Exists(path))
        {
            try
            {
                // Create a BitmapImage object
                BitmapImage bitmapImage = new BitmapImage();

                // Initialize the image from the file path
                bitmapImage.BeginInit();
                bitmapImage.UriSource = new Uri(path);  // Pass the file path as a Uri
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;  // Fully load the image into memory
                bitmapImage.EndInit();  // End initialization

                return bitmapImage;
            }
            catch (Exception ex)
            {
                // Log the error if loading the image fails
                Console.WriteLine($"Error loading image: {ex.Message}");
                return GetPlaceholderImage();  // Return a placeholder image on error
            }
        }
        else
        {
            // If the file does not exist, return a placeholder image
            return GetPlaceholderImage();
        }
    }

    // Placeholder image provider in case the image loading fails
    private static BitmapImage GetPlaceholderImage()
    {
        // Create a DrawingVisual object
        DrawingVisual drawingVisual = new DrawingVisual();

        // Create a DrawingContext for drawing on the DrawingVisual
        using (DrawingContext drawingContext = drawingVisual.RenderOpen())
        {
            // Draw a light gray rectangle as background
            drawingContext.DrawRectangle(Brushes.LightGray, new Pen(Brushes.Black, 2), new Rect(0, 0, 200, 200));

            // Draw a red "X"
            drawingContext.DrawLine(new Pen(Brushes.Red, 10), new Point(50, 50), new Point(150, 150));  // First diagonal line
            drawingContext.DrawLine(new Pen(Brushes.Red, 10), new Point(150, 50), new Point(50, 150));  // Second diagonal line
        }

        // Create a RenderTargetBitmap with the appropriate size
        RenderTargetBitmap renderTargetBitmap = new RenderTargetBitmap(200, 200, 96, 96, PixelFormats.Pbgra32);

        // Render the DrawingVisual to the RenderTargetBitmap
        renderTargetBitmap.Render(drawingVisual);

        // Convert RenderTargetBitmap to BitmapImage
        BitmapImage bitmapImage = new BitmapImage();
        using (MemoryStream memoryStream = new MemoryStream())
        {
            // Save the RenderTargetBitmap as PNG into a MemoryStream
            PngBitmapEncoder encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(renderTargetBitmap));
            encoder.Save(memoryStream);

            // Reset the position to the start of the stream
            memoryStream.Position = 0;

            // Initialize BitmapImage from MemoryStream
            bitmapImage.BeginInit();
            bitmapImage.StreamSource = memoryStream;
            bitmapImage.EndInit();
        }

        return bitmapImage;
    }


}
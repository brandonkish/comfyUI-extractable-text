using System;
using System.Diagnostics;
using System.IO;
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
        BitmapImage placeholderImage = new BitmapImage();
        placeholderImage.BeginInit();
        placeholderImage.UriSource = new Uri("pack://application:,,,/Images/placeholder.png");  // Specify a valid path for placeholder
        placeholderImage.EndInit();
        return placeholderImage;
    }
}
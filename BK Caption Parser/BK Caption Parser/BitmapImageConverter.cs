using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media.Imaging;

namespace BK_Caption_Parser
{
    public class BitmapImageConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value == null) return null;

            // If the value is already a BitmapImage, return it directly
            if (value is BitmapImage bitmapImage)
            {
                return bitmapImage;
            }

            var path = value.ToString();
            if (string.IsNullOrEmpty(path)) return null;

            // Now we create the BitmapImage from the file path
            bitmapImage = new BitmapImage();
            bitmapImage.BeginInit();
            bitmapImage.UriSource = new Uri(path);
            bitmapImage.CacheOption = BitmapCacheOption.OnLoad;  // This ensures the file is not locked
            bitmapImage.EndInit();
            return bitmapImage;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}

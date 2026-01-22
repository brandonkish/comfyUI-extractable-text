using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Windows.Media.Imaging;


namespace BK_Caption_Parser;
public static class FolderParser
{
    static Regex numberRegex = new Regex(@"(\d+)(?!.*\d)");

    static Regex numberCheckRegex = new Regex(@"(\d+)");

    public static bool CompareStrings(string? str1, string? str2)
    {
        if (string.IsNullOrEmpty(str1) || string.IsNullOrEmpty(str2))
            return false;

        // Find the longest common prefix
        int minLength = Math.Min(str1.Length, str2.Length);
        int commonPrefixLength = 0;

        while (commonPrefixLength < minLength && str1[commonPrefixLength] == str2[commonPrefixLength])
        {
            commonPrefixLength++;
        }

        // Remove the common prefix from both strings
        string remainingStr1 = str1.Substring(commonPrefixLength);
        string remainingStr2 = str2.Substring(commonPrefixLength);

        // If the first character of either remaining string is a number, return false
        if (remainingStr1.Length > 0 && Char.IsDigit(remainingStr1[0]))
            return false;

        if (remainingStr2.Length > 0 && Char.IsDigit(remainingStr2[0]))
            return false;

        // Otherwise, return true
        return true;
    }

    // Helper method to extract the numeric part after the prefix
    private static string GetNumberAfterPrefix(string fileName)
    {
        // Regex to capture the numeric part after the first underscore and prefix
        Regex numberRegex = new Regex(@"(?<=\D)(\d+)(?=\D|$)"); // Match digits after the first underscore

        var match = numberRegex.Match(fileName);
        return match.Success ? match.Value : string.Empty; // Return the numeric part as string
    }

    public static List<ImageSet> Parse(string? folder)
    {


        if (folder == null)
        {
            Debug.WriteLine($"folder is null when trying to parse the folder in FolderParser. Returning an empty List<ImageSet>.");
            return new List<ImageSet>();
        }

        var sets = new Dictionary<int, ImageSet>();

        foreach (var txt in Directory.GetFiles(folder, "*.txt"))
        {
            var match = numberRegex.Match(Path.GetFileNameWithoutExtension(txt));
            if (!match.Success) continue;

            int num = int.Parse(match.Value);

            sets[num] = new ImageSet
            {
                SetNumber = num,
                FolderPath = folder,
                CaptionTextPath = txt,
                CaptionText = File.ReadAllText(txt)
            };
        }

        foreach (var set in sets.Values)
        {
            string? baseName = Path.GetFileNameWithoutExtension(set.CaptionTextPath);

            if (baseName == null) continue;

            var supportedExtensions = new[] { ".png", ".jpeg", ".jpg", ".bmp", ".gif", ".tiff" };  // Add all supported image extensions
            var images = Directory.GetFiles(folder)
                .Where(f => f.StartsWith(Path.Combine(folder, baseName)) &&
                            supportedExtensions.Contains(Path.GetExtension(f).ToLower()))
                .ToList();

            Debug.WriteLine($"baseName [{baseName}]");

            // Find the Original Image (exact match based on base name with number suffix)
            set.OriginalImagePath = images.FirstOrDefault(i =>
                Path.GetFileNameWithoutExtension(i) == baseName);

            Debug.WriteLine($"OriginalImagePath [{set.OriginalImagePath}]");

            // Find the Caption Image (matching based on the same base name but a different number suffix)
            set.CaptionImagePath = images.FirstOrDefault(i =>
                i != set.OriginalImagePath &&
                Path.GetFileNameWithoutExtension(i).StartsWith(baseName) &&
                !Path.GetFileNameWithoutExtension(i).EndsWith(baseName.Split('_').Last()) &&
                CompareStrings(i, set.OriginalImagePath)); // Ensure different number suffix

            Debug.WriteLine($"CaptionImagePath [{set.CaptionImagePath}]");

            // Load the images (fallback to placeholder if not found)
            set.OriginalImage = LoadOrPlaceholder(set.OriginalImagePath);
            set.CaptionImage = LoadOrPlaceholder(set.CaptionImagePath);

            // Hashing the files
            set.OriginalImageHash = FileHasher.HashFirst2KB(set.OriginalImagePath);
            set.CaptionImageHash = FileHasher.HashFirst2KB(set.CaptionImagePath);
            set.CaptionTextHash = FileHasher.HashFirst2KB(set.CaptionTextPath);
        }

        return sets.Values.OrderBy(s => s.SetNumber).ToList();
    }

    static BitmapImage LoadOrPlaceholder(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            Debug.WriteLine($"The path was null when trying to load or set placeholder. Returning placeholder image.");
            return PlaceholderImageProvider.Image;
        }
            

        return File.Exists(path)
            ? new BitmapImage(new System.Uri(path))
            : PlaceholderImageProvider.Image;
    }

}

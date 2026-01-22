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

            set.OriginalImagePath = images.FirstOrDefault(i =>
                Path.GetFileNameWithoutExtension(i) == baseName);

            set.CaptionImagePath = images.FirstOrDefault(i =>
                i != set.OriginalImagePath);

            set.OriginalImage = LoadOrPlaceholder(set.OriginalImagePath);
            set.CaptionImage = LoadOrPlaceholder(set.CaptionImagePath);

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

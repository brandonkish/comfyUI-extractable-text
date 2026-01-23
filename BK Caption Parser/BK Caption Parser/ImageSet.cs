using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Media.Imaging;

public class ImageSet : INotifyPropertyChanged
{
    public int SetNumber { get; set; }

    public string? FolderPath { get; set; }

    public string? CaptionTextPath { get; set; }
    public string? OriginalImagePath { get; set; }
    public string? CaptionImagePath { get; set; }

    public string? CaptionText { get; set; }

    public string? CaptionChangeText { get; set; }

    public string? CaptionTextHash { get; set; }
    public string? OriginalImageHash { get; set; }
    public string? CaptionImageHash { get; set; }

    public BitmapImage? OriginalImage { get; set; }
    public BitmapImage? CaptionImage { get; set; }

    public event PropertyChangedEventHandler? PropertyChanged;
    void OnPropertyChanged([CallerMemberName] string? p = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(p));
}

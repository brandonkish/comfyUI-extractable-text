using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace BK_Caption_Parser;

public static class PlaceholderImageProvider
{
    public static BitmapImage Image { get; }

    static PlaceholderImageProvider()
    {
        var bmp = new RenderTargetBitmap(300, 300, 96, 96, PixelFormats.Pbgra32);
        var dv = new DrawingVisual();

        using (var dc = dv.RenderOpen())
        {
            dc.DrawRectangle(Brushes.White, null, new System.Windows.Rect(0, 0, 300, 300));
            var pen = new Pen(Brushes.Red, 10);
            dc.DrawLine(pen, new(0, 0), new(300, 300));
            dc.DrawLine(pen, new(300, 0), new(0, 300));
        }

        bmp.Render(dv);

        using var ms = new MemoryStream();
        BitmapEncoder enc = new PngBitmapEncoder();
        enc.Frames.Add(BitmapFrame.Create(bmp));
        enc.Save(ms);

        Image = new BitmapImage();
        Image.BeginInit();
        Image.StreamSource = new MemoryStream(ms.ToArray());
        Image.EndInit();
    }
}

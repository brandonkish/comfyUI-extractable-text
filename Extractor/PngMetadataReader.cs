using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Collections;
using static System.Net.Mime.MediaTypeNames;

namespace METAVACE;

public class PngMetadataReader
{
    private static byte[] pngSignature = new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 };
    private static int signatureLength = pngSignature.Length;

    public static Chunk GetPNGWorkflowChunk(string filePath)
    {
        Chunk result = null;

        using FileStream image = File.OpenRead(filePath);

        if (!HasPNGSignature(image)) 
            throw new ArgumentException("Valid PNG signature not found in file.");

        try
        {
            while (true)
            {
                Chunk chunk = Chunk.GetNext(image);
                if (!chunk.Type.Equals("tEXt"))
                    continue;
                if(!chunk.Data.StartsWith("description", StringComparison.OrdinalIgnoreCase))
                    continue;
                result = chunk;
                break;
            }
        }
        catch (EndOfStreamException eos)
        {
            throw new InvalidOperationException("End of file reached. No Description found in image.");
        }

        return result;
    }

    private static bool HasPNGSignature(FileStream image)
    {
        if (image == null) return false;
        byte[] fileSignature = new byte[signatureLength];
        image.ReadExactly(fileSignature);

        if (fileSignature.SequenceEqual(pngSignature)) return true;
        return false;
    }
}

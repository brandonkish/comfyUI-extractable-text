using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text;


namespace BK_Caption_Parser;

public static class FileHasher
{
    public static string? HashFirst2KB(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            Debug.WriteLine($"The path is null when trying to has the first 2KB of file.");
            return "";
        }
            

        if (!File.Exists(path)) return null;

        using var sha = SHA256.Create();
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);

        byte[] buffer = new byte[2048];
        int bytesRead = fs.Read(buffer, 0, buffer.Length);

        // Check if we read the full 2048 bytes
        if (bytesRead != buffer.Length)
        {
            // Handle case where fewer bytes were read than expected
            // Optionally, trim the buffer to the actual number of bytes read
            Array.Resize(ref buffer, bytesRead);  // Resize the buffer if not all bytes are read
        }

        return Convert.ToHexString(sha.ComputeHash(buffer));
    }
}

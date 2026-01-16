using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

public class FileManager
{
    // Method to read file names from a text file
    public List<string> ReadFileNames(string filePath)
    {

        if (!File.Exists(filePath))
        {
            // If the file doesn't exist, create it (optionally, you can add default content here)
            File.Create(filePath).Dispose(); // Dispose immediately to release the file handle
            Debug.WriteLine($"File not found. A new file has been created at {filePath}.");
        }

        return new List<string>(File.ReadAllLines(filePath));
    }

    // Method to write a list of file names to a text file
    public void WriteFileNames(string filePath, List<string> fileNames)
    {
        File.WriteAllLines(filePath, fileNames);
    }
}

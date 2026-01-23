using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;


namespace AI_Toolkit_Log_Parser;
public class LogParser
{
    public List<LogEntry> ParseLog(string logFilePath)
    {
        var logEntries = new List<LogEntry>();
        string lastFileName = null;
        float lastLoss = 0;
        int lastStep = 0;

        // Regex to match the loss line
        var lossRegex = new Regex(@"(\d+/\d+)\s+\[\d+:\d+<\d+:\d+:\d+,.*loss:\s*([0-9.e+-]+)");

        // Read log file line by line
        foreach (var line in File.ReadLines(logFilePath))
        {
            // Check for lines with 'Saved checkpoint to'
            if (line.StartsWith("Saved checkpoint to"))
            {
                // Extract the .safetensors filename
                var match = Regex.Match(line, @"Saved checkpoint to (.+\.safetensors)");
                if (match.Success)
                {
                    lastFileName = match.Groups[1].Value;
                }
            }

            // Check for lines with 'loss:'
            var lossMatch = lossRegex.Match(line);
            if (lossMatch.Success && lastFileName != null)
            {
                var stepMatch = Regex.Match(lossMatch.Groups[1].Value, @"(\d+)/\d+");
                if (stepMatch.Success)
                {
                    lastStep = int.Parse(stepMatch.Groups[1].Value);
                }

                lastLoss = float.Parse(lossMatch.Groups[2].Value);

                // Add the entry to the list
                logEntries.Add(new LogEntry
                {
                    FileName = lastFileName,
                    Loss = lastLoss,
                    Step = lastStep
                });
            }
        }

        return logEntries;
    }
}

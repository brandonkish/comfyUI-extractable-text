public class LogEntry
{
    public string? FileName { get; set; }  // Make FileName nullable
    public float Loss { get; set; }
    public int Step { get; set; }

    public string Display => $"{FileName} - Loss: {Loss} - Step: {Step}";
}

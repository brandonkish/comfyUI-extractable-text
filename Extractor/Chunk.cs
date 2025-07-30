using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace METAVACE;
public class Chunk
{
    private int _length;
    private string? _type;
    private string? _data;
    private int _crc;

    private static int chunkByteLength = 4;
    private static int chunkTypeLength = 4;

    public int Length
    {
        get { return _length; }
    }


    public string Type
    {
        get
        {
            if (_type == null)
                _type = string.Empty;
            return _type;
        }
    }

    public string Data
    {
        get
        {
            if (_data == null)
                _data = string.Empty;
            return _data;
        }
    }

    public int CRC
    {
        get { return _crc; }
    }

    public Chunk(string dataString)
    {
        _data = dataString;
    }

    private Chunk()
    {
        _length = 0;
        _type = null;
        _data = null;
        _crc = 0;
    }

    public static Chunk GetNext(FileStream pngFile)
    {
        Chunk chunk = new Chunk();
        chunk._length = GetChunkLength(pngFile);
        chunk._type = GetChunkType(pngFile);
        chunk._data = GetChunkData(pngFile, chunk.Length);
        chunk._crc = GetChunkCRC(pngFile);
        return chunk;
    }

    private static int GetChunkLength(FileStream image)
    {
        if (image == null) return -1;
        byte[] chunkLength = new byte[chunkByteLength];
        image.ReadExactly(chunkLength);
        return BytesToInt(chunkLength);
    }


    private static string? GetChunkType(FileStream image)
    {
        if (image is null) return null;
        byte[] chunkType = new byte[chunkTypeLength];
        image.ReadExactly(chunkType);

        return Encoding.UTF8.GetString(chunkType);
    }

    private static string? GetChunkData(FileStream image, int chunkLength)
    {
        if (image == null) return null;
        byte[] chunkData = new byte[chunkLength];
        image.ReadExactly(chunkData);
        return Encoding.UTF8.GetString(chunkData);
    }

    private static int GetChunkCRC(FileStream image)
    {
        if (image == null) return -1;
        byte[] chunkCRC = new byte[chunkByteLength];
        image.ReadExactly(chunkCRC);
        return BytesToInt(chunkCRC);
    }

    private static int BytesToInt(byte[] byteArray)
    {
        if (byteArray.Length != 4)
        {
            throw new ArgumentException("The chunk length must be exactly 4 bytes.");
        }

        // Convert the byte array from big-endian to an integer
        int chunkLength = BitConverter.ToInt32(byteArray, 0);

        // Ensure the result is correctly interpreted as a big-endian value
        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(byteArray);
            chunkLength = BitConverter.ToInt32(byteArray, 0);
        }

        return chunkLength;
    }

    public string GetDataJSON() =>
        GetNestedJson(Data);

    private static int GetFirstCurlyBracketIdx(string str) =>
            str.IndexOf("{");

    private static int GetLength(string str, int idx) =>
        str.Length - idx;

    private static string GetNestedJson(string str) =>
        str.Substring(GetFirstCurlyBracketIdx(str), GetLength(str, GetFirstCurlyBracketIdx(str)));
}

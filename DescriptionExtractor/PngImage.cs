using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace METAVACE
{
    
    
    

    class PngImage
    {
        private Chunk _chunk;

        public string Description
        {
            get
            {
                if(_chunk is null)
                    return string.Empty;

                if(_chunk.Data is null)
                    return string.Empty;

                return _chunk.Data;
            }
        }

        public PngImage(string path)
        {
            _chunk = PngMetadataReader.GetPNGWorkflowChunk(path);
        }
    }
}

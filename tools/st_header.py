import json
import struct
import sys
import urllib.request

Header = dict[str, dict]


# reads only the json header of a remote safetensors file (two http range requests, no weights).
def read_header(url: str) -> Header:
    def fetch(start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        return urllib.request.urlopen(request).read()

    size = struct.unpack("<Q", fetch(0, 7))[0]
    return json.loads(fetch(8, 7 + size))


if __name__ == "__main__":
    header = read_header(sys.argv[1])
    header.pop("__metadata__", None)
    json.dump({k: [v["dtype"], v["shape"]] for k, v in sorted(header.items())}, sys.stdout)

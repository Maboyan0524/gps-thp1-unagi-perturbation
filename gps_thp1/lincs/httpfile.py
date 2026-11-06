"""极简 HTTP 范围读文件对象，给 h5py 用。只依赖 requests。"""
import io, os, requests

class HTTPFile(io.RawIOBase):
    def __init__(self, url, block=4*1024*1024, session=None):
        self.url = url
        self.block = block
        self.s = session or requests.Session()
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = {}          # block_idx -> bytes
        self.bytes_read = 0
        self.n_req = 0
    def readable(self): return True
    def seekable(self): return True
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos
    def tell(self): return self.pos
    def _fetch(self, bi):
        if bi in self.cache: return self.cache[bi]
        st = bi * self.block
        en = min(st + self.block, self.size) - 1
        for attempt in range(8):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={st}-{en}"}, timeout=180)
                r.raise_for_status()
                d = r.content
                break
            except Exception:
                if attempt == 7: raise
        self.bytes_read += len(d); self.n_req += 1
        if len(self.cache) > 600: self.cache.clear()
        self.cache[bi] = d
        return d
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray(); left = n
        while left > 0:
            bi = self.pos // self.block
            off = self.pos - bi * self.block
            d = self._fetch(bi)
            take = min(left, len(d) - off)
            if take <= 0: break
            out += d[off:off+take]; self.pos += take; left -= take
        return bytes(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

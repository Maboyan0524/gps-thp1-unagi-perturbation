"""从远端 gctx 按行抓取。矩阵是连续未压缩存储，直接算偏移、合并相邻行做范围请求。"""
import os
import sys, time, h5py, numpy as np, requests
sys.path.insert(0, os.environ["WORK_DIR"])
from httpfile import HTTPFile

U = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx"
NG = 12328; ITEM = 4

def row_base(url):
    fh = HTTPFile(url, block=1024*1024)
    f = h5py.File(fh, "r")
    d = f["/0/DATA/0/matrix"]
    off = d.id.get_offset()
    assert d.chunks is None and d.compression is None, "不是连续未压缩存储"
    assert off is not None, "拿不到偏移"
    return off, d.shape

def fetch(url, rows, base, gap=64, verbose=True):
    """rows: 行索引数组。gap: 相邻行号差 <= gap 就合并成一次请求"""
    rows = np.unique(np.asarray(rows))
    spans = []
    st = pv = rows[0]
    for r in rows[1:]:
        if r - pv <= gap: pv = r
        else: spans.append((st, pv)); st = pv = r
    spans.append((st, pv))
    tot = sum((b - a + 1) for a, b in spans) * NG * ITEM
    if verbose:
        print(f"  {len(rows)} 行 → 合并成 {len(spans)} 次请求，传输约 {tot/1048576:.1f} MB", flush=True)
    s = requests.Session()
    out = np.empty((len(rows), NG), dtype=np.float32)
    pos = {r: i for i, r in enumerate(rows)}
    got = 0; t0 = time.time()
    for k, (a, b) in enumerate(spans):
        st_b = base + a * NG * ITEM
        en_b = base + (b + 1) * NG * ITEM - 1
        for attempt in range(8):
            try:
                r = s.get(url, headers={"Range": f"bytes={st_b}-{en_b}"}, timeout=300)
                r.raise_for_status(); buf = r.content; break
            except Exception:
                if attempt == 7: raise
                time.sleep(2)
        arr = np.frombuffer(buf, dtype="<f4").reshape(b - a + 1, NG)
        for rr in range(a, b + 1):
            if rr in pos: out[pos[rr]] = arr[rr - a]
        got += len(buf)
        if verbose and (k + 1) % max(1, len(spans)//10) == 0:
            el = time.time() - t0
            print(f"    {k+1}/{len(spans)}  {got/1048576:.0f} MB  {got/el/1048576:.2f} MB/s", flush=True)
    return out, rows

if __name__ == "__main__":
    t = time.time()
    base, shape = row_base(U)
    print(f"矩阵 {shape}  数据起始偏移 {base}", flush=True)
    idx = np.load(f"{os.environ['WORK_DIR']}/thp1_rowidx.npy")
    ids = np.load(f"{os.environ['WORK_DIR']}/all_sigids.npy", allow_pickle=True)
    print(f"THP1 行数 {len(idx)}", flush=True)
    X, rows = fetch(U, idx, base)
    np.save(f"{os.environ['WORK_DIR']}/thp1_X.npy", X)
    np.save(f"{os.environ['WORK_DIR']}/thp1_rows_sorted.npy", rows)
    np.save(f"{os.environ['WORK_DIR']}/thp1_sigids.npy", ids[rows])
    print(f"完成 {X.shape}  耗时 {time.time()-t:.0f}s", flush=True)
    print("=== FETCH DONE ===", flush=True)

import os
import faiss

def is_cdn_mounted(mount_point: str = "./cdn_storage") -> bool:
    return os.path.ismount(mount_point)

def load_faiss_index(event_id: str, dimension: int, use_id_map: bool = True):
    from core.config import FAISS_INDEX_DIR

    if not is_cdn_mounted():
        raise RuntimeError("CDN storage is not mounted")

    if not event_id:
        raise ValueError("event_id must be provided")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")

    if os.path.exists(path):
        return faiss.read_index(path)

    base_index = faiss.IndexFlatIP(dimension)
    return faiss.IndexIDMap(base_index) if use_id_map else base_index


def save_faiss_index(index: faiss.Index, event_id: str):
    from core.config import FAISS_INDEX_DIR

    if not is_cdn_mounted():
        raise RuntimeError("CDN storage is not mounted")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")
    faiss.write_index(index, path)

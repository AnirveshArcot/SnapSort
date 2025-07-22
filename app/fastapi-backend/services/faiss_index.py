import os
import faiss

def load_faiss_index(event_id: str, dimension: int):
    from core.config import FAISS_INDEX_DIR
    if not event_id:
        raise ValueError("event_id must be provided")
    path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")
    if os.path.exists(path):
        return faiss.read_index(path)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
    return index

def save_faiss_index(faiss_index, event_id: str):
    from core.config import FAISS_INDEX_DIR
    path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")
    faiss.write_index(faiss_index, path)
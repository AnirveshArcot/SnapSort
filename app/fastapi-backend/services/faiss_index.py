import os
import faiss

def load_faiss_index(event_id: str, dimension: int):
    from core.config import FAISS_INDEX_DIR
    if event_id is None:
        raise ValueError("`event_id` must be provided")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    index_path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")
    print(index_path)
    print(os.path.exists(index_path))
    if os.path.exists(index_path):
        faiss_index = faiss.read_index(index_path)
        # Get number of vectors and dimension
        num_vectors = faiss_index.ntotal
        dimension = faiss_index.d

        # Print all vectors
        for i in range(num_vectors):
            vector = faiss_index.reconstruct(i)
            print(f"Vector {i}: {vector}")
        print(f"Loaded FAISS index for event {event_id}")
    else:
        base = faiss.IndexFlatIP(dimension)
        faiss_index = faiss.IndexIDMap(base)
        print(f"Created new FAISS index for event {event_id}")
    
    return faiss_index


def save_faiss_index(faiss_index: faiss.Index, event_id: str):
    from core.config import FAISS_INDEX_DIR
    if event_id is None:
        raise ValueError("`event_id` must be provided")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    if faiss_index is None:
        print(f"No FAISS index loaded for event {event_id}; nothing to save.")
        return

    index_path = os.path.join(FAISS_INDEX_DIR, f"{event_id}.faiss")
    faiss.write_index(faiss_index, index_path)
    print(f"Saved FAISS index for event {event_id} to '{index_path}'")

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from ..config import get_settings


def uuid_to_faiss_id(uuid_str: str) -> int:
    import uuid as _uuid

    value = _uuid.UUID(uuid_str)
    return int((value.int >> 64) & 0xFFFFFFFFFFFFFFFF)


class VectorStore:
    def __init__(self, dimension: Optional[int] = None) -> None:
        self.settings = get_settings()
        self.index_path = Path(self.settings.data_dir) / "faiss.index"
        self.meta_path = Path(self.settings.data_dir) / "faiss_meta.json"
        self.id_map: Dict[str, str] = {}
        self.dimension = dimension or self.settings.embedding_dim
        self.index = self._load_index()
        if self.index.d != self.dimension:
            self.dimension = self.index.d

    def _load_index(self) -> faiss.Index:
        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.d != self.dimension:
                # 既存インデックスと埋め込み次元が異なる場合は再初期化
                self.reset(self.dimension)
                return self.index
            self.dimension = index.d
            self.id_map = self._load_metadata()
            return index
        index = faiss.IndexFlatIP(self.dimension)
        return faiss.IndexIDMap(index)

    def _load_metadata(self) -> Dict[str, str]:
        if not self.meta_path.exists():
            return {}
        with self.meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.get("id_map", {}).items()}

    def _save_metadata(self) -> None:
        payload = {"id_map": self.id_map}
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        self._save_metadata()

    def reset(self, dimension: Optional[int] = None) -> None:
        if dimension is not None:
            self.dimension = dimension
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        self.id_map = {}
        self.save()

    def add(self, ids: Sequence[str], vectors: Sequence[np.ndarray]) -> None:
        if not ids:
            return
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors size mismatch")
        embeddings = np.stack(vectors).astype(np.float32)
        faiss_ids = []
        for case_id in ids:
            fid = uuid_to_faiss_id(case_id)
            faiss_ids.append(fid)
            self.id_map[str(fid)] = case_id
        id_array = np.array(faiss_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, id_array)
        self.save()

    def remove(self, ids: Iterable[str]) -> None:
        remove_ids: List[int] = []
        for case_id in ids:
            fid = uuid_to_faiss_id(case_id)
            remove_ids.append(fid)
            self.id_map.pop(str(fid), None)
        if remove_ids:
            id_array = np.array(remove_ids, dtype=np.int64)
            self.index.remove_ids(id_array)
            self.save()

    def rebuild(self, id_vector_pairs: Sequence[Tuple[str, np.ndarray]]) -> None:
        if not id_vector_pairs:
            self.reset(self.dimension)
            return
        dimension = id_vector_pairs[0][1].shape[-1]
        self.reset(dimension)
        ids, vectors = zip(*id_vector_pairs)
        self.add(ids, vectors)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter_ids: Optional[Sequence[str]] = None,
        oversample: int = 3,
    ) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        limit = min(top_k * oversample, self.index.ntotal)
        query = query_vector.astype(np.float32).reshape(1, -1)
        scores, labels = self.index.search(query, limit)
        allowed = set(filter_ids) if filter_ids else None
        results: List[Tuple[str, float]] = []
        for score, label in zip(scores[0], labels[0]):
            if label == -1:
                continue
            key = str(int(label))
            case_id = self.id_map.get(key)
            if not case_id:
                continue
            if allowed is not None and case_id not in allowed:
                continue
            results.append((case_id, float(score)))
            if len(results) >= top_k:
                break
        return results


__all__ = ["VectorStore", "uuid_to_faiss_id"]

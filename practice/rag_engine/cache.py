"""Embedding缓存：用内容哈希做key，避免重复编码"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(self,cache_dir:Path,model_name:str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.model_name = model_name
        self.keys_path = self.cache_dir / "keys.json"
        self.vectors_path = self.cache_dir / "vectors.npy"
        self._store = self._load()
    
    def _key(self,text:str)->str:
        raw = f"{self.model_name}::{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) ->dict:
        if not self.keys_path.exists() or not self.vectors_path.exists():
            return{}
        
        keys = json.loads(self.keys_path.read_text(encoding="utf-8"))
        vectors = np.load(self.vectors_path)
        return dict(zip(keys,vectors))

    def _save(self):
        if not self._store:
            return
        
        keys = list(self._store)
        vectors = np.stack([self._store[k] for k in keys])
        self.keys_path.write_text(json.dumps(keys),encoding="utf-8")
        np.save(self.vectors_path,vectors)

    def encode(self,texts:list[str],encoder)->np.ndarray:
        """返回和texts顺序一致的向量矩阵；只编码缓存里没有的"""
        keys = [self._key(t) for t in texts]
        missing = [i for i,k in enumerate(keys) if k not in self._store]

        if missing:
            # 同一次调用里，相同内容的 key 只编码一次
            text_by_key = {}
            for i in missing:
                text_by_key.setdefault(keys[i], texts[i])   # 每个 key 只保留一份文本

            new_keys = list(text_by_key)
            new_texts = [text_by_key[k] for k in new_keys]
            logger.info("缓存命中 %d / %d，需要编码 %d 条（去重后 %d 条）",
                        len(texts) - len(missing), len(texts), len(missing), len(new_keys))

            new_vectors = encoder.encode(new_texts)
            for k, vec in zip(new_keys, new_vectors):
                self._store[k] = vec
            self._save()


        else:
            logger.info("全部命中缓存：%d 条",len(texts))

        return np.stack([self._store[k] for k in keys])
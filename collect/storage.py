"""Storage backends for capture data, labels, and checkpoints.

Provides a Protocol-based abstraction with three implementations:
- LocalStorage: filesystem (default, backwards-compatible)
- R2Storage: Cloudflare R2 via S3-compatible API (boto3)
- CachedR2Storage: R2 with batched local disk cache for training
"""

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal storage interface shared by all backends."""

    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def put_text(self, key: str, text: str) -> None: ...
    def get_text(self, key: str) -> str: ...
    def list_keys(self, prefix: str = "") -> list[str]: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...


class LocalStorage:
    """Filesystem backend rooted at *data_root*. Keys are relative paths."""

    def __init__(self, data_root: str):
        self.root = Path(data_root)

    def put(self, key: str, data: bytes) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def put_text(self, key: str, text: str) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def get_text(self, key: str) -> str:
        return (self.root / key).read_text()

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()
        )

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)


class R2Storage:
    """Cloudflare R2 via the S3-compatible API (boto3).

    Credentials are read from constructor args or environment variables:
      R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
    """

    def __init__(
        self,
        account_id: str,
        bucket: str,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ):
        import boto3
        from botocore.config import Config as BotoConfig

        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id or os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=secret_access_key
            or os.environ["R2_SECRET_ACCESS_KEY"],
            config=BotoConfig(
                retries={"max_attempts": 3, "mode": "adaptive"},
                signature_version="s3v4",
            ),
            region_name="auto",
        )

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        resp = self._client.get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()

    def put_text(self, key: str, text: str) -> None:
        self.put(key, text.encode("utf-8"))

    def get_text(self, key: str) -> str:
        return self.get(key).decode("utf-8")

    def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self.bucket, Key=key)

    def presign(self, key: str, expires: int = 3600) -> str:
        """Return a pre-signed GET URL valid for *expires* seconds."""
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )


class CachedR2Storage:
    """Wraps R2Storage with a local disk cache for batched reads.

    Usage for batch training:
        cached = CachedR2Storage(r2, cache_dir=".r2cache")
        cached.prefetch(list_of_keys)   # parallel download
        data = cached.get(key)          # served from cache
        cached.clear_cache()            # cleanup
    """

    def __init__(self, r2: R2Storage, cache_dir: str = ".r2cache"):
        self.r2 = r2
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- Delegated writes go straight to R2 --

    def put(self, key: str, data: bytes) -> None:
        self.r2.put(key, data)

    def put_text(self, key: str, text: str) -> None:
        self.r2.put_text(key, text)

    # -- Reads check cache first --

    def get(self, key: str) -> bytes:
        cached = self.cache_dir / key
        if cached.exists():
            return cached.read_bytes()
        data = self.r2.get(key)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        return data

    def get_text(self, key: str) -> str:
        return self.get(key).decode("utf-8")

    def list_keys(self, prefix: str = "") -> list[str]:
        return self.r2.list_keys(prefix)

    def exists(self, key: str) -> bool:
        cached = self.cache_dir / key
        return cached.exists() or self.r2.exists(key)

    def delete(self, key: str) -> None:
        cached = self.cache_dir / key
        if cached.exists():
            cached.unlink()
        self.r2.delete(key)

    def presign(self, key: str, expires: int = 3600) -> str:
        return self.r2.presign(key, expires)

    # -- Batch operations --

    def prefetch(self, keys: list[str], workers: int = 8) -> None:
        """Download *keys* from R2 into the local cache in parallel.

        Skips keys already present in cache.
        """
        to_fetch = [k for k in keys if not (self.cache_dir / k).exists()]
        if not to_fetch:
            logger.info("Cache is warm — nothing to prefetch.")
            return
        logger.info(
            f"Prefetching {len(to_fetch)} files from R2 ({len(keys) - len(to_fetch)} already cached)..."
        )

        def _download(key: str) -> str | None:
            try:
                data = self.r2.get(key)
                dest = self.cache_dir / key
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return None
            except Exception as e:
                return f"{key}: {e}"

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download, k): k for k in to_fetch}
            for fut in as_completed(futures):
                err = fut.result()
                if err:
                    errors.append(err)

        if errors:
            logger.warning(
                f"Prefetch completed with {len(errors)} errors:\n  "
                + "\n  ".join(errors[:10])
            )
        else:
            logger.info(f"Prefetch complete: {len(to_fetch)} files downloaded.")

    def clear_cache(self) -> None:
        """Remove all cached files."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cache cleared: {self.cache_dir}")

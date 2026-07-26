"""Tests for collect.storage backends."""

from unittest.mock import MagicMock, patch

from collect.storage import LocalStorage, R2Storage, CachedR2Storage


# ---------- LocalStorage ----------


class TestLocalStorage:
    def test_put_get_bytes(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        store.put("a/b/file.bin", b"hello")
        assert store.get("a/b/file.bin") == b"hello"

    def test_put_text_get_text(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        store.put_text("notes.txt", "world")
        assert store.get_text("notes.txt") == "world"

    def test_exists(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        assert not store.exists("missing.txt")
        store.put_text("present.txt", "x")
        assert store.exists("present.txt")

    def test_list_keys(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        store.put_text("20260322/001/images/a.jpg", "img")
        store.put_text("20260322/001/metar/metar.txt", "metar")
        store.put_text("20260323/002/images/b.jpg", "img2")

        keys = store.list_keys("20260322")
        assert len(keys) == 2
        assert "20260322/001/images/a.jpg" in keys
        assert "20260322/001/metar/metar.txt" in keys

    def test_list_keys_empty(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        assert store.list_keys("nonexistent") == []

    def test_creates_parent_dirs(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        store.put("deep/nested/dir/file.bin", b"data")
        assert (tmp_path / "deep" / "nested" / "dir" / "file.bin").exists()


# ---------- R2Storage (mocked boto3) ----------


class TestR2Storage:
    @patch.dict(
        "sys.modules",
        {"boto3": MagicMock(), "botocore": MagicMock(), "botocore.config": MagicMock()},
    )
    def test_init_from_env(self, monkeypatch):
        import sys

        mock_boto3 = sys.modules["boto3"]
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key-id")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret")

        # Re-import to pick up mocked boto3
        import importlib
        import collect.storage

        importlib.reload(collect.storage)

        store = collect.storage.R2Storage(account_id="abc123", bucket="my-bucket")
        mock_boto3.client.assert_called_once()
        call_kwargs = mock_boto3.client.call_args
        assert (
            call_kwargs[1]["endpoint_url"] == "https://abc123.r2.cloudflarestorage.com"
        )
        assert call_kwargs[1]["aws_access_key_id"] == "test-key-id"
        assert store.bucket == "my-bucket"

    def test_put(self, monkeypatch):
        mock_client = MagicMock()
        store = R2Storage.__new__(R2Storage)
        store.bucket = "b"
        store._client = mock_client

        store.put("key.jpg", b"data")
        mock_client.put_object.assert_called_once_with(
            Bucket="b", Key="key.jpg", Body=b"data"
        )

    def test_get(self):
        mock_client = MagicMock()
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: b"img-data")
        }
        store = R2Storage.__new__(R2Storage)
        store.bucket = "b"
        store._client = mock_client

        assert store.get("key.jpg") == b"img-data"

    def test_put_text(self):
        mock_client = MagicMock()
        store = R2Storage.__new__(R2Storage)
        store.bucket = "b"
        store._client = mock_client

        store.put_text("metar.txt", "KSEA 221153Z")
        mock_client.put_object.assert_called_once_with(
            Bucket="b", Key="metar.txt", Body=b"KSEA 221153Z"
        )

    def test_presign(self):
        mock_client = MagicMock()
        mock_client.generate_presigned_url.return_value = "https://signed.url"
        store = R2Storage.__new__(R2Storage)
        store.bucket = "b"
        store._client = mock_client

        url = store.presign("img.jpg", expires=600)
        assert url == "https://signed.url"

    def test_list_keys(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "a.jpg"}, {"Key": "b.txt"}]},
        ]
        mock_client.get_paginator.return_value = paginator
        store = R2Storage.__new__(R2Storage)
        store.bucket = "b"
        store._client = mock_client

        keys = store.list_keys("prefix/")
        assert keys == ["a.jpg", "b.txt"]


# ---------- CachedR2Storage ----------


class TestCachedR2Storage:
    def test_get_caches_locally(self, tmp_path):
        mock_r2 = MagicMock(spec=R2Storage)
        mock_r2.get.return_value = b"image-bytes"

        cached = CachedR2Storage(mock_r2, cache_dir=str(tmp_path / "cache"))
        result = cached.get("20260322/img.jpg")
        assert result == b"image-bytes"

        # Second call should read from cache, not R2
        result2 = cached.get("20260322/img.jpg")
        assert result2 == b"image-bytes"
        assert mock_r2.get.call_count == 1  # only called once

    def test_prefetch_skips_cached(self, tmp_path):
        mock_r2 = MagicMock(spec=R2Storage)
        mock_r2.get.return_value = b"data"

        cache_dir = tmp_path / "cache"
        cached = CachedR2Storage(mock_r2, cache_dir=str(cache_dir))

        # Pre-populate one file in cache
        (cache_dir / "existing.jpg").parent.mkdir(parents=True, exist_ok=True)
        (cache_dir / "existing.jpg").write_bytes(b"cached")

        cached.prefetch(["existing.jpg", "new.jpg"])

        # Only "new.jpg" should be fetched from R2
        mock_r2.get.assert_called_once_with("new.jpg")

    def test_put_delegates_to_r2(self, tmp_path):
        mock_r2 = MagicMock(spec=R2Storage)
        cached = CachedR2Storage(mock_r2, cache_dir=str(tmp_path / "cache"))

        cached.put("key", b"data")
        mock_r2.put.assert_called_once_with("key", b"data")

    def test_clear_cache(self, tmp_path):
        mock_r2 = MagicMock(spec=R2Storage)
        cache_dir = tmp_path / "cache"
        cached = CachedR2Storage(mock_r2, cache_dir=str(cache_dir))
        (cache_dir / "some_file").write_bytes(b"x")

        cached.clear_cache()
        assert not cache_dir.exists()

    def test_presign_delegates(self, tmp_path):
        mock_r2 = MagicMock(spec=R2Storage)
        mock_r2.presign.return_value = "https://signed"
        cached = CachedR2Storage(mock_r2, cache_dir=str(tmp_path / "cache"))

        assert cached.presign("key") == "https://signed"

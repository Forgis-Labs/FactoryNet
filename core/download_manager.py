"""Download manager for external datasets.

Provides robust downloading with retry logic, caching, checksums,
and progress tracking.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_CHUNK_SIZE = 8192  # bytes


@dataclass
class DownloadResult:
    """Result of a download operation."""
    url: str
    local_path: Path
    success: bool
    size_bytes: int = 0
    checksum: str = ""
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


@dataclass
class DownloadConfig:
    """Configuration for download operations."""
    timeout: int = DEFAULT_TIMEOUT
    retry_count: int = DEFAULT_RETRY_COUNT
    retry_delay: float = DEFAULT_RETRY_DELAY
    chunk_size: int = DEFAULT_CHUNK_SIZE
    verify_ssl: bool = True
    user_agent: str = "FactoryNet/1.0"
    headers: Dict[str, str] = field(default_factory=dict)


class DownloadManager:
    """Manages downloads with caching, retries, and progress tracking.

    Features:
    - Automatic retry on failure
    - Resume partial downloads
    - Checksum verification
    - Progress bar display
    - Local caching to avoid re-downloads

    Example:
        manager = DownloadManager(cache_dir="./downloads")

        # Download a single file
        result = manager.download(
            url="https://example.com/data.mat",
            filename="data.mat",
            expected_checksum="abc123..."
        )

        # Download multiple files
        results = manager.download_batch([
            ("https://example.com/file1.mat", "file1.mat"),
            ("https://example.com/file2.mat", "file2.mat"),
        ])
    """

    def __init__(
        self,
        cache_dir: str | Path,
        config: Optional[DownloadConfig] = None,
    ):
        """Initialize download manager.

        Args:
            cache_dir: Directory for downloaded files
            config: Download configuration options
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or DownloadConfig()

        # Checksum cache file
        self.checksum_file = self.cache_dir / ".checksums.json"
        self._checksums: Dict[str, str] = self._load_checksums()

    def _load_checksums(self) -> Dict[str, str]:
        """Load cached checksums from file."""
        if self.checksum_file.exists():
            import json
            try:
                with open(self.checksum_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_checksums(self) -> None:
        """Save checksums to cache file."""
        import json
        with open(self.checksum_file, "w") as f:
            json.dump(self._checksums, f, indent=2)

    def _compute_checksum(
        self,
        file_path: Path,
        algorithm: str = "md5",
    ) -> str:
        """Compute checksum of a file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha256)

        Returns:
            Hex digest of checksum
        """
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(self.config.chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_session(self) -> requests.Session:
        """Create a requests session with default headers."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.config.user_agent,
            **self.config.headers,
        })
        return session

    def download(
        self,
        url: str,
        filename: Optional[str] = None,
        subdir: Optional[str] = None,
        expected_checksum: Optional[str] = None,
        force: bool = False,
        progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DownloadResult:
        """Download a file from URL.

        Args:
            url: URL to download
            filename: Local filename (defaults to URL filename)
            subdir: Subdirectory within cache_dir
            expected_checksum: Expected MD5 checksum
            force: Force re-download even if cached
            progress: Show progress bar
            progress_callback: Optional callback(downloaded, total)

        Returns:
            DownloadResult with status and file info
        """
        start_time = time.time()

        # Determine local path
        if filename is None:
            filename = Path(urlparse(url).path).name
        if subdir:
            local_dir = self.cache_dir / subdir
            local_dir.mkdir(parents=True, exist_ok=True)
        else:
            local_dir = self.cache_dir
        local_path = local_dir / filename

        # Check if already cached
        if local_path.exists() and not force:
            # Verify checksum if provided
            if expected_checksum:
                cached_checksum = self._checksums.get(str(local_path))
                if cached_checksum == expected_checksum:
                    logger.debug(f"Using cached file: {local_path}")
                    return DownloadResult(
                        url=url,
                        local_path=local_path,
                        success=True,
                        size_bytes=local_path.stat().st_size,
                        checksum=cached_checksum,
                        elapsed_seconds=time.time() - start_time,
                    )
                # Checksum mismatch - recompute
                actual_checksum = self._compute_checksum(local_path)
                if actual_checksum == expected_checksum:
                    self._checksums[str(local_path)] = actual_checksum
                    self._save_checksums()
                    return DownloadResult(
                        url=url,
                        local_path=local_path,
                        success=True,
                        size_bytes=local_path.stat().st_size,
                        checksum=actual_checksum,
                        elapsed_seconds=time.time() - start_time,
                    )
                # Checksum mismatch, need to re-download
                logger.warning(f"Checksum mismatch for {local_path}, re-downloading")
            else:
                # No checksum, assume cached file is valid
                return DownloadResult(
                    url=url,
                    local_path=local_path,
                    success=True,
                    size_bytes=local_path.stat().st_size,
                    checksum=self._checksums.get(str(local_path), ""),
                    elapsed_seconds=time.time() - start_time,
                )

        # Download with retries
        last_error = None
        for attempt in range(self.config.retry_count):
            try:
                result = self._do_download(
                    url=url,
                    local_path=local_path,
                    progress=progress,
                    progress_callback=progress_callback,
                )
                if result.success:
                    # Verify checksum
                    if expected_checksum:
                        actual_checksum = self._compute_checksum(local_path)
                        if actual_checksum != expected_checksum:
                            result.success = False
                            result.error = f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                            local_path.unlink(missing_ok=True)
                            continue
                        result.checksum = actual_checksum
                    else:
                        result.checksum = self._compute_checksum(local_path)

                    # Cache checksum
                    self._checksums[str(local_path)] = result.checksum
                    self._save_checksums()

                    result.elapsed_seconds = time.time() - start_time
                    return result

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Download attempt {attempt + 1}/{self.config.retry_count} failed: {e}"
                )
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        return DownloadResult(
            url=url,
            local_path=local_path,
            success=False,
            error=last_error or "Download failed",
            elapsed_seconds=time.time() - start_time,
        )

    def _do_download(
        self,
        url: str,
        local_path: Path,
        progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DownloadResult:
        """Perform the actual download.

        Args:
            url: URL to download
            local_path: Local file path
            progress: Show progress bar
            progress_callback: Optional callback(downloaded, total)

        Returns:
            DownloadResult
        """
        session = self._get_session()

        # Get file size
        response = session.get(
            url,
            stream=True,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        # Download to temp file first
        temp_path = local_path.with_suffix(local_path.suffix + ".tmp")

        try:
            with open(temp_path, "wb") as f:
                if progress and total_size > 0:
                    pbar = tqdm(
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        desc=local_path.name,
                    )
                else:
                    pbar = None

                downloaded = 0
                for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if pbar:
                            pbar.update(len(chunk))
                        if progress_callback:
                            progress_callback(downloaded, total_size)

                if pbar:
                    pbar.close()

            # Move temp file to final location
            shutil.move(str(temp_path), str(local_path))

            return DownloadResult(
                url=url,
                local_path=local_path,
                success=True,
                size_bytes=downloaded,
            )

        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise

    def download_batch(
        self,
        urls: List[tuple[str, str]],
        subdir: Optional[str] = None,
        progress: bool = True,
    ) -> List[DownloadResult]:
        """Download multiple files.

        Args:
            urls: List of (url, filename) tuples
            subdir: Subdirectory for all files
            progress: Show progress bar

        Returns:
            List of DownloadResult objects
        """
        results = []
        with tqdm(total=len(urls), desc="Downloading", disable=not progress) as pbar:
            for url, filename in urls:
                result = self.download(
                    url=url,
                    filename=filename,
                    subdir=subdir,
                    progress=False,
                )
                results.append(result)
                pbar.update(1)
                if not result.success:
                    logger.warning(f"Failed to download {url}: {result.error}")

        return results

    def is_cached(self, filename: str, subdir: Optional[str] = None) -> bool:
        """Check if a file is already cached.

        Args:
            filename: Filename to check
            subdir: Optional subdirectory

        Returns:
            True if file exists in cache
        """
        if subdir:
            path = self.cache_dir / subdir / filename
        else:
            path = self.cache_dir / filename
        return path.exists()

    def get_cached_path(
        self,
        filename: str,
        subdir: Optional[str] = None,
    ) -> Optional[Path]:
        """Get path to cached file if it exists.

        Args:
            filename: Filename to look up
            subdir: Optional subdirectory

        Returns:
            Path if cached, None otherwise
        """
        if subdir:
            path = self.cache_dir / subdir / filename
        else:
            path = self.cache_dir / filename
        return path if path.exists() else None

    def clear_cache(self, subdir: Optional[str] = None) -> int:
        """Clear cached files.

        Args:
            subdir: Only clear specific subdirectory

        Returns:
            Number of files removed
        """
        count = 0
        if subdir:
            target = self.cache_dir / subdir
            if target.exists():
                for f in target.iterdir():
                    if f.is_file():
                        f.unlink()
                        count += 1
        else:
            for f in self.cache_dir.iterdir():
                if f.is_file() and f.name != ".checksums.json":
                    f.unlink()
                    count += 1
        return count

    def get_cache_size(self) -> int:
        """Get total size of cached files in bytes."""
        total = 0
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

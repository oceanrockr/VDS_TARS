"""
NAS File Watcher Service

Monitors a NAS directory for new/modified documents and automatically indexes them.
Uses Python watchdog library for filesystem monitoring with debouncing and deduplication.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from app.core.config import settings
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)


class NASFileWatcher:
    """
    Monitors NAS directory for document changes and triggers indexing.
    """

    def __init__(self):
        self.observer: Optional[Observer] = None
        self.watch_path = settings.NAS_MOUNT_POINT
        self.enabled = settings.NAS_WATCH_ENABLED
        self.scan_interval = settings.NAS_SCAN_INTERVAL
        self.max_file_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes

        # Parse allowed extensions
        self.allowed_extensions = set(
            ext.strip().lower()
            for ext in settings.ALLOWED_EXTENSIONS.split(',')
        )

        # Debouncing: track pending files
        self.pending_files: Dict[str, float] = {}
        self.debounce_delay = 5.0  # seconds
        self.processing_files: Set[str] = set()
        self.indexed_hashes: Set[str] = set()

        # Statistics
        self.stats = {
            'files_detected': 0,
            'files_indexed': 0,
            'files_failed': 0,
            'files_skipped': 0,
            'last_scan': None,
            'started_at': None,
        }

    def is_valid_file(self, file_path: Path) -> bool:
        """
        Check if file should be indexed.

        Args:
            file_path: Path to file

        Returns:
            True if file is valid for indexing
        """
        # Check extension
        if file_path.suffix.lower() not in self.allowed_extensions:
            return False

        # Check if file exists and is a file
        if not file_path.is_file():
            return False

        # Check file size
        try:
            if file_path.stat().st_size > self.max_file_size:
                logger.warning(f"File too large: {file_path} ({file_path.stat().st_size} bytes)")
                return False
        except Exception as e:
            logger.error(f"Error checking file size: {e}")
            return False

        return True

    def compute_file_hash(self, file_path: Path) -> Optional[str]:
        """
        Compute SHA256 hash of file for deduplication.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash or None if error
        """
        try:
            sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return None

    async def index_file(self, file_path: str) -> bool:
        """
        Index a single file using RAG service.

        Args:
            file_path: Path to file to index

        Returns:
            True if successful
        """
        path = Path(file_path)

        # Validate file
        if not self.is_valid_file(path):
            self.stats['files_skipped'] += 1
            return False

        # Check if already processing
        if file_path in self.processing_files:
            logger.debug(f"File already being processed: {file_path}")
            return False

        # Compute hash for deduplication
        file_hash = self.compute_file_hash(path)
        if file_hash and file_hash in self.indexed_hashes:
            logger.info(f"File already indexed (hash match): {file_path}")
            self.stats['files_skipped'] += 1
            return False

        try:
            self.processing_files.add(file_path)
            logger.info(f"Indexing file: {file_path}")

            # Index via RAG service
            from app.models.rag import DocumentUploadRequest
            request = DocumentUploadRequest(
                file_path=file_path,
                force_reindex=False
            )

            result = await rag_service.index_document(request)

            if result.success:
                logger.info(
                    f"Successfully indexed {file_path}: "
                    f"{result.chunks_created} chunks, "
                    f"{result.processing_time_ms:.0f}ms"
                )

                # Store hash to prevent re-indexing
                if file_hash:
                    self.indexed_hashes.add(file_hash)

                self.stats['files_indexed'] += 1
                return True
            else:
                logger.error(f"Failed to index {file_path}")
                self.stats['files_failed'] += 1
                return False

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}", exc_info=True)
            self.stats['files_failed'] += 1
            return False
        finally:
            self.processing_files.discard(file_path)

    async def process_pending_files(self):
        """
        Process files that have been debounced.
        """
        while True:
            try:
                current_time = time.time()
                files_to_process = []

                # Find files ready to process
                for file_path, timestamp in list(self.pending_files.items()):
                    if current_time - timestamp >= self.debounce_delay:
                        files_to_process.append(file_path)
                        del self.pending_files[file_path]

                # Process ready files
                for file_path in files_to_process:
                    await self.index_file(file_path)

            except Exception as e:
                logger.error(f"Error in pending files processor: {e}", exc_info=True)

            # Check every second
            await asyncio.sleep(1.0)

    async def scan_directory(self):
        """
        Perform full directory scan for existing files.
        """
        if not Path(self.watch_path).exists():
            logger.warning(f"Watch path does not exist: {self.watch_path}")
            return

        logger.info(f"Scanning directory: {self.watch_path}")
        scan_start = time.time()
        files_found = 0

        try:
            for file_path in Path(self.watch_path).rglob('*'):
                if file_path.is_file() and self.is_valid_file(file_path):
                    files_found += 1
                    # Add to pending with immediate processing (no debounce for initial scan)
                    self.pending_files[str(file_path)] = time.time() - self.debounce_delay

            scan_time = time.time() - scan_start
            self.stats['last_scan'] = datetime.now().isoformat()
            logger.info(
                f"Directory scan complete: {files_found} files found in {scan_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error scanning directory: {e}", exc_info=True)

    async def periodic_scan(self):
        """
        Perform periodic directory scans.
        """
        while True:
            try:
                await self.scan_directory()
            except Exception as e:
                logger.error(f"Error in periodic scan: {e}", exc_info=True)

            # Wait for next scan
            await asyncio.sleep(self.scan_interval)

    def start(self):
        """
        Start the file watcher service.
        """
        if not self.enabled:
            logger.info("NAS watcher disabled in configuration")
            return

        if not Path(self.watch_path).exists():
            logger.error(f"NAS mount point does not exist: {self.watch_path}")
            return

        self.stats['started_at'] = datetime.now().isoformat()
        logger.info(f"Starting NAS file watcher on {self.watch_path}")

        # Create event handler
        event_handler = NASEventHandler(self)

        # Create and start observer
        self.observer = Observer()
        self.observer.schedule(event_handler, self.watch_path, recursive=True)
        self.observer.start()

        logger.info("NAS file watcher started successfully")

    def stop(self):
        """
        Stop the file watcher service.
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("NAS file watcher stopped")

    def get_stats(self) -> Dict:
        """
        Get watcher statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'pending_files': len(self.pending_files),
            'processing_files': len(self.processing_files),
            'indexed_hashes': len(self.indexed_hashes),
            'enabled': self.enabled,
            'watch_path': self.watch_path,
        }


class NASEventHandler(FileSystemEventHandler):
    """
    Handles filesystem events for NAS watcher.
    """

    def __init__(self, watcher: NASFileWatcher):
        super().__init__()
        self.watcher = watcher

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = event.src_path
        logger.debug(f"File created: {file_path}")

        self.watcher.stats['files_detected'] += 1
        # Add to pending files with debounce
        self.watcher.pending_files[file_path] = time.time()

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = event.src_path
        logger.debug(f"File modified: {file_path}")

        # Update timestamp to restart debounce
        if file_path in self.watcher.pending_files:
            self.watcher.pending_files[file_path] = time.time()


# Global watcher instance
nas_watcher = NASFileWatcher()

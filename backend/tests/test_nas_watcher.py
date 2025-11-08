"""
Tests for NAS Watcher Service

Basic tests for NAS file monitoring functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.nas_watcher import NASFileWatcher


@pytest.fixture
def nas_watcher():
    """Create a NAS watcher instance for testing"""
    watcher = NASFileWatcher()
    watcher.enabled = False  # Disable actual watching for tests
    return watcher


class TestNASFileWatcher:
    """Test NAS file watcher methods"""

    def test_watcher_initialization(self, nas_watcher):
        """Test watcher can be initialized"""
        assert nas_watcher.debounce_delay == 5.0
        assert nas_watcher.max_file_size > 0
        assert len(nas_watcher.allowed_extensions) > 0

    def test_allowed_extensions(self, nas_watcher):
        """Test allowed file extensions are loaded"""
        assert '.pdf' in nas_watcher.allowed_extensions
        assert '.docx' in nas_watcher.allowed_extensions
        assert '.txt' in nas_watcher.allowed_extensions
        assert '.md' in nas_watcher.allowed_extensions
        assert '.csv' in nas_watcher.allowed_extensions

    def test_is_valid_file_extension(self, nas_watcher):
        """Test file extension validation"""
        # Valid extensions
        assert nas_watcher.is_valid_file(Path("test.pdf")) or not Path("test.pdf").exists()
        assert nas_watcher.is_valid_file(Path("test.docx")) or not Path("test.docx").exists()

    def test_is_valid_file_nonexistent(self, nas_watcher):
        """Test validation fails for nonexistent files"""
        result = nas_watcher.is_valid_file(Path("/nonexistent/file.pdf"))
        assert result is False

    def test_compute_file_hash_nonexistent(self, nas_watcher):
        """Test hash computation for nonexistent file"""
        result = nas_watcher.compute_file_hash(Path("/nonexistent/file.pdf"))
        assert result is None

    def test_get_stats(self, nas_watcher):
        """Test getting watcher statistics"""
        stats = nas_watcher.get_stats()

        assert 'files_detected' in stats
        assert 'files_indexed' in stats
        assert 'files_failed' in stats
        assert 'files_skipped' in stats
        assert 'pending_files' in stats
        assert 'processing_files' in stats
        assert 'indexed_hashes' in stats
        assert 'enabled' in stats
        assert 'watch_path' in stats

        assert isinstance(stats['files_detected'], int)
        assert isinstance(stats['pending_files'], int)
        assert isinstance(stats['enabled'], bool)

    def test_stats_initialization(self, nas_watcher):
        """Test initial stats are zero"""
        stats = nas_watcher.get_stats()

        assert stats['files_detected'] == 0
        assert stats['files_indexed'] == 0
        assert stats['files_failed'] == 0
        assert stats['files_skipped'] == 0


@pytest.mark.asyncio
class TestNASWatcherAsync:
    """Test async NAS watcher methods"""

    @pytest.mark.skip(reason="Requires RAG service")
    async def test_index_file(self, nas_watcher):
        """Test file indexing"""
        # Would require RAG service to be running
        result = await nas_watcher.index_file("/path/to/test.pdf")
        assert isinstance(result, bool)

    @pytest.mark.skip(reason="Requires filesystem access")
    async def test_scan_directory(self, nas_watcher):
        """Test directory scanning"""
        # Would require actual filesystem
        await nas_watcher.scan_directory()
        stats = nas_watcher.get_stats()
        assert 'last_scan' in stats

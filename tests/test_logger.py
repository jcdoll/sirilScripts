"""Tests for logger module."""

import tempfile
import time
from pathlib import Path

from siril_job_runner.logger import JobLogger, create_logger


def test_elapsed_time_format():
    """Test elapsed time formatting."""
    logger = JobLogger()
    # Just started, should be [00:00]
    elapsed = logger._elapsed()
    assert elapsed.startswith("[00:0")


def test_elapsed_time_increments():
    """Test that elapsed time increments."""
    logger = JobLogger()
    time.sleep(1.1)
    elapsed = logger._elapsed()
    # Should be at least 1 second
    assert elapsed in ["[00:01]", "[00:02]"]


def test_info_output(capsys):
    """Test info message output."""
    logger = JobLogger()
    logger.info("Test message")
    captured = capsys.readouterr()
    assert "Test message" in captured.out


def test_substep_indentation(capsys):
    """Test substep indentation."""
    logger = JobLogger()
    logger.substep("Indented message")
    captured = capsys.readouterr()
    # Should have leading spaces for indentation
    assert "  Indented message" in captured.out


def test_log_file_creation():
    """Test log file creation in output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        logger = JobLogger(output_dir, job_name="test_job")
        logger.info("Test log entry")
        logger.close()

        # Check log file was created
        log_files = list(output_dir.glob("job_log_test_job_*.txt"))
        assert len(log_files) == 1

        # Check content
        content = log_files[0].read_text()
        assert "Test log entry" in content


def test_context_manager():
    """Test logger as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with JobLogger(Path(tmpdir), "ctx_test") as logger:
            logger.info("Inside context")

        # Log file should be closed and contain completion message
        log_files = list(Path(tmpdir).glob("job_log_*.txt"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "Job completed" in content


def test_timed_operation(capsys):
    """Test timed operation context manager."""
    logger = JobLogger()
    with logger.timed_operation("Test operation"):
        time.sleep(0.1)

    captured = capsys.readouterr()
    assert "Test operation" in captured.out
    assert "completed in" in captured.out


def test_warning_prefix(capsys):
    """Test warning message has prefix."""
    logger = JobLogger()
    logger.warning("Something bad")
    captured = capsys.readouterr()
    assert "WARNING:" in captured.out


def test_error_prefix(capsys):
    """Test error message has prefix."""
    logger = JobLogger()
    logger.error("Something failed")
    captured = capsys.readouterr()
    assert "ERROR:" in captured.out


def test_create_logger_convenience():
    """Test create_logger convenience function."""
    logger = create_logger()
    assert isinstance(logger, JobLogger)


def test_table_output(capsys):
    """Test table formatting."""
    logger = JobLogger()
    logger.table(["Filter", "Count"], [["L", "45"], ["R", "20"]])
    captured = capsys.readouterr()
    assert "Filter" in captured.out
    assert "Count" in captured.out
    assert "L" in captured.out
    assert "45" in captured.out

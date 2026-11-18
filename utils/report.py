"""
utils/report.py — Legacy compatibility wrapper for report utilities.

Delegates directly to utils.report_utils to avoid code duplication.
"""

from utils.report_utils import (
    generate_markdown_report as write_markdown,
    print_results_table as print_table,
    console,
    _fmt,
)

__all__ = ["write_markdown", "print_table", "console", "_fmt"]

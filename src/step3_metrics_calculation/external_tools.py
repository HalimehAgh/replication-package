"""
External Tool Integration for Metrics Analysis

Integrates external static analysis tools like PMD-CPD for metrics
that cannot be calculated with Python AST alone.

PMD Copy/Paste Detector (CPD) is used for code clone detection.
"""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from src import config

logger = logging.getLogger(__name__)


class PMDCPDRunner:
    """
    Run PMD Copy/Paste Detector for code clone detection.

    PMD-CPD identifies duplicated code blocks by tokenizing source code
    and finding sequences of identical tokens.

    Installation:
        macOS: brew install pmd
        Linux: Download from https://pmd.github.io/

    Environment variable PMD_CPD_PATH can be set to specify the path.
    """

    def __init__(self, pmd_path: Optional[str] = None):
        """
        Initialize PMD-CPD runner.

        Args:
            pmd_path: Path to PMD installation (uses config/PATH if None)
        """
        self.pmd_path = pmd_path or config.PMD_CPD_PATH
        self._available = None  # Lazy validation

    def is_available(self) -> bool:
        """
        Check if PMD-CPD is available on the system.

        Returns:
            True if PMD-CPD can be executed
        """
        if self._available is not None:
            return self._available

        try:
            # Try to run pmd with --version
            result = subprocess.run(
                [self.pmd_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            self._available = result.returncode == 0
            if self._available:
                logger.debug(f"PMD-CPD available: {result.stdout.strip()}")
            else:
                logger.debug(f"PMD-CPD not available: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug(f"PMD-CPD not found: {e}")
            self._available = False

        return self._available

    def calculate_clone_ratio(
        self,
        source_code: str,
        min_tokens: Optional[int] = None
    ) -> float:
        """
        Calculate code clone ratio for source code.

        Clone ratio = duplicated_tokens / total_tokens

        Args:
            source_code: Python source code content
            min_tokens: Minimum tokens for duplicate detection (default from config)

        Returns:
            Clone ratio as float (0.0-1.0), or 0.0 if analysis fails
        """
        if not self.is_available():
            logger.debug("PMD-CPD not available, returning 0.0 for clone ratio")
            return 0.0

        if not source_code or not source_code.strip():
            return 0.0

        min_tokens = min_tokens or config.PMD_CPD_MIN_TOKENS

        # Create isolated temporary directory with single file for analysis
        with tempfile.TemporaryDirectory(prefix="pmd_cpd_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_file = temp_dir_path / "source.py"
            temp_file.write_text(source_code, encoding='utf-8')

            try:
                # Run PMD-CPD on the isolated directory
                xml_output = self._run_cpd(temp_dir_path, min_tokens)
                if xml_output is None:
                    return 0.0

                # Parse results
                clone_info = self._parse_cpd_output(xml_output)
                if clone_info is None:
                    return 0.0

                # Calculate ratio
                total_tokens = self._estimate_tokens(source_code)
                if total_tokens == 0:
                    return 0.0

                duplicated_tokens = clone_info.get('total_duplicated_tokens', 0)
                return min(1.0, duplicated_tokens / total_tokens)

            except Exception as e:
                logger.warning(f"Error calculating clone ratio: {e}")
                return 0.0

    def _run_cpd(
        self,
        directory: Path,
        min_tokens: int
    ) -> Optional[str]:
        """
        Run PMD-CPD on a directory.

        Args:
            directory: Directory containing files to analyze
            min_tokens: Minimum token threshold

        Returns:
            XML output string or None if failed
        """
        try:
            # PMD 7.x syntax: use --dir to scan all Python files in directory
            cmd = [
                self.pmd_path, "cpd",
                "--minimum-tokens", str(min_tokens),
                "--language", config.PMD_CPD_LANGUAGE,
                "--format", "xml",
                "--dir", str(directory)
            ]

            # Try PMD 7.x first
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.PMD_CPD_TIMEOUT
            )

            # If that fails, try PMD 6.x syntax
            if result.returncode != 0 and "Unknown option" in result.stderr:
                cmd = [
                    self.pmd_path, "cpd",
                    "--minimum-tokens", str(min_tokens),
                    "--language", config.PMD_CPD_LANGUAGE,
                    "--format", "xml",
                    "--files", str(directory)
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.PMD_CPD_TIMEOUT
                )

            # Return code 4 means duplications found, which is fine
            if result.returncode in (0, 4):
                return result.stdout
            else:
                logger.debug(f"PMD-CPD failed with code {result.returncode}: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"PMD-CPD timed out after {config.PMD_CPD_TIMEOUT}s")
            return None
        except Exception as e:
            logger.warning(f"Error running PMD-CPD: {e}")
            return None

    def _parse_cpd_output(self, xml_output: str) -> Optional[Dict[str, Any]]:
        """
        Parse PMD-CPD XML output.

        Returns:
            Dictionary with clone information or None if parsing fails
        """
        try:
            # Handle empty output
            if not xml_output or not xml_output.strip():
                return {"duplications": [], "total_duplicated_tokens": 0}

            root = ET.fromstring(xml_output)

            duplications = []
            total_duplicated_tokens = 0

            # PMD 7.x uses XML namespace
            ns = {'cpd': 'https://pmd-code.org/schema/cpd-report'}
            dup_elements = root.findall('.//cpd:duplication', ns)

            # Fallback to no namespace (PMD 6.x or earlier)
            if not dup_elements:
                dup_elements = root.findall('.//duplication')

            for dup in dup_elements:
                lines = int(dup.get('lines', 0))
                tokens = int(dup.get('tokens', 0))

                files = []
                file_elements = dup.findall('cpd:file', ns)
                if not file_elements:
                    file_elements = dup.findall('file')

                for file_elem in file_elements:
                    files.append({
                        'path': file_elem.get('path'),
                        'line': int(file_elem.get('line', 0)),
                        'endline': int(file_elem.get('endline', 0)),
                    })

                duplications.append({
                    'lines': lines,
                    'tokens': tokens,
                    'files': files,
                })

                # Count tokens (multiply by occurrences - 1 to avoid double counting)
                if len(files) > 1:
                    total_duplicated_tokens += tokens * (len(files) - 1)

            return {
                'duplications': duplications,
                'total_duplicated_tokens': total_duplicated_tokens,
            }

        except ET.ParseError as e:
            logger.debug(f"Error parsing CPD XML: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error parsing CPD output: {e}")
            return None

    def _estimate_tokens(self, source_code: str) -> int:
        """
        Estimate total tokens in source code.

        This is a rough estimate based on whitespace-separated words
        and punctuation. CPD uses a more sophisticated tokenizer.

        Args:
            source_code: Source code string

        Returns:
            Estimated token count
        """
        # Simple estimation: count words and significant punctuation
        # Remove comments and strings for more accurate count
        code = re.sub(r'#.*$', '', source_code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)

        # Count identifiers and keywords
        identifiers = len(re.findall(r'\b\w+\b', code))

        # Count operators and punctuation
        operators = len(re.findall(r'[+\-*/=<>!&|^%@~]+|[(){}\[\],.:;]', code))

        return identifiers + operators


# =============================================================================
# Convenience function for integration
# =============================================================================

# Singleton instance
_pmd_runner: Optional[PMDCPDRunner] = None


def get_pmd_runner() -> PMDCPDRunner:
    """Get or create singleton PMD-CPD runner instance."""
    global _pmd_runner
    if _pmd_runner is None:
        _pmd_runner = PMDCPDRunner()
    return _pmd_runner


def calculate_code_clone_ratio(source_code: str) -> float:
    """
    Calculate code clone ratio using PMD-CPD.

    Convenience function for use in analyze_metrics.py.

    Args:
        source_code: Python source code content

    Returns:
        Clone ratio as float (0.0-1.0)
    """
    runner = get_pmd_runner()
    return runner.calculate_clone_ratio(source_code)

# type: ignore
"""
Generic Document Parser Utility

This module provides functionality for parsing PDF and image documents using MinerU 2.0 library,
and converts the parsing results into markdown and JSON formats

Note: MinerU 2.0 no longer includes LibreOffice document conversion module.
For Office documents (.doc, .docx, .ppt, .pptx), please convert them to PDF format first.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import (
    Any,
    TypeVar,
)

T = TypeVar("T")


class Parser:
    """
    Base class for document parsing utilities.

    Defines common functionality and constants for parsing different document types.
    """

    # Define common file formats
    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS = {".txt", ".md"}

    # Class-level logger
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialize the base parser."""
        pass

    @staticmethod
    def convert_office_to_pdf(
        doc_path: str | Path, output_dir: str | None = None
    ) -> Path:
        """
        Convert Office document (.doc, .docx, .ppt, .pptx, .xls, .xlsx) to PDF.
        Requires LibreOffice to be installed.

        Args:
            doc_path: Path to the Office document file
            output_dir: Output directory for the PDF file

        Returns:
            Path to the generated PDF file
        """
        try:
            # Convert to Path object for easier handling
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Office document does not exist: {doc_path}")

            name_without_suff = doc_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "libreoffice_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary directory for PDF conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Convert to PDF using LibreOffice
                logging.info(f"Converting {doc_path.name} to PDF using LibreOffice...")

                # Prepare subprocess parameters to hide console window on Windows
                import platform

                # Try LibreOffice commands in order of preference
                commands_to_try = ["libreoffice", "soffice"]

                conversion_successful = False
                for cmd in commands_to_try:
                    try:
                        convert_cmd = [
                            cmd,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(temp_path),
                            str(doc_path),
                        ]

                        # Prepare conversion subprocess parameters
                        convert_subprocess_kwargs = {
                            "capture_output": True,
                            "text": True,
                            "timeout": 60,  # 60 second timeout
                            "encoding": "utf-8",
                            "errors": "ignore",
                        }

                        # Hide console window on Windows
                        if platform.system() == "Windows":
                            convert_subprocess_kwargs["creationflags"] = (
                                subprocess.CREATE_NO_WINDOW
                            )

                        result = subprocess.run(
                            convert_cmd, **convert_subprocess_kwargs
                        )

                        if result.returncode == 0:
                            conversion_successful = True
                            logging.info(
                                f"Successfully converted {doc_path.name} to PDF using {cmd}"
                            )
                            break
                        else:
                            logging.warning(
                                f"LibreOffice command '{cmd}' failed: {result.stderr}"
                            )
                    except FileNotFoundError:
                        logging.warning(f"LibreOffice command '{cmd}' not found")
                    except subprocess.TimeoutExpired:
                        logging.warning(f"LibreOffice command '{cmd}' timed out")
                    except Exception as e:
                        logging.error(
                            f"LibreOffice command '{cmd}' failed with exception: {e}"
                        )

                if not conversion_successful:
                    raise RuntimeError(
                        f"LibreOffice conversion failed for {doc_path.name}. "
                        f"Please ensure LibreOffice is installed:\n"
                        "- Windows: Download from https://www.libreoffice.org/download/download/\n"
                        "- macOS: brew install --cask libreoffice\n"
                        "- Ubuntu/Debian: sudo apt-get install libreoffice\n"
                        "- CentOS/RHEL: sudo yum install libreoffice\n"
                        "Alternatively, convert the document to PDF manually."
                    )

                # Find the generated PDF
                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    raise RuntimeError(
                        f"PDF conversion failed for {doc_path.name} - no PDF file generated. "
                        f"Please check LibreOffice installation or try manual conversion."
                    )

                pdf_path = pdf_files[0]
                logging.info(
                    f"Generated PDF: {pdf_path.name} ({pdf_path.stat().st_size} bytes)"
                )

                # Validate the generated PDF
                if pdf_path.stat().st_size < 100:  # Very small file, likely empty
                    raise RuntimeError(
                        "Generated PDF appears to be empty or corrupted. "
                        "Original file may have issues or LibreOffice conversion failed."
                    )

                # Copy PDF to final output directory
                final_pdf_path = base_output_dir / f"{name_without_suff}.pdf"
                import shutil

                shutil.copy2(pdf_path, final_pdf_path)

                return final_pdf_path

        except Exception as e:
            logging.error(f"Error in convert_office_to_pdf: {str(e)}")
            raise

    @staticmethod
    def convert_text_to_pdf(
        text_path: str | Path, output_dir: str | None = None
    ) -> Path:
        """
        Convert text file (.txt, .md) to PDF using ReportLab with full markdown support.

        Args:
            text_path: Path to the text file
            output_dir: Output directory for the PDF file

        Returns:
            Path to the generated PDF file
        """
        try:
            text_path = Path(text_path)
            if not text_path.exists():
                raise FileNotFoundError(f"Text file does not exist: {text_path}")

            # Supported text formats
            supported_text_formats = {".txt", ".md"}
            if text_path.suffix.lower() not in supported_text_formats:
                raise ValueError(f"Unsupported text format: {text_path.suffix}")

            # Read the text content
            try:
                with open(text_path, encoding="utf-8") as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ["gbk", "latin-1", "cp1252"]:
                    try:
                        with open(text_path, encoding=encoding) as f:
                            text_content = f.read()
                        logging.info(f"Successfully read file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(
                        f"Could not decode text file {text_path.name} with any supported encoding"
                    )

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = text_path.parent / "reportlab_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = base_output_dir / f"{text_path.stem}.pdf"

            # Convert text to PDF
            logging.info(f"Converting {text_path.name} to PDF...")

            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
                from reportlab.lib.units import inch
                from reportlab.pdfbase import pdfmetrics
                from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

                # Create PDF document
                doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=A4,
                    leftMargin=inch,
                    rightMargin=inch,
                    topMargin=inch,
                    bottomMargin=inch,
                )

                # Get styles
                styles = getSampleStyleSheet()
                normal_style = styles["Normal"]
                heading_style = styles["Heading1"]

                # Try to register a font that supports Chinese characters
                try:
                    # Try to use system fonts that support Chinese
                    import platform

                    system = platform.system()
                    if system == "Windows":
                        # Try common Windows fonts
                        for font_name in ["SimSun", "SimHei", "Microsoft YaHei"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                    elif system == "Darwin":  # macOS
                        for font_name in ["STSong-Light", "STHeiti"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                except Exception:
                    pass  # Use default fonts if Chinese font setup fails

                # Build content
                story = []

                # Handle markdown or plain text
                if text_path.suffix.lower() == ".md":
                    # Handle markdown content - simplified implementation
                    lines = text_content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            story.append(Spacer(1, 12))
                            continue

                        # Headers
                        if line.startswith("#"):
                            level = len(line) - len(line.lstrip("#"))
                            header_text = line.lstrip("#").strip()
                            if header_text:
                                header_style = ParagraphStyle(
                                    name=f"Heading{level}",
                                    parent=heading_style,
                                    fontSize=max(16 - level, 10),
                                    spaceAfter=8,
                                    spaceBefore=16 if level <= 2 else 12,
                                )
                                story.append(Paragraph(header_text, header_style))
                        else:
                            # Regular text
                            story.append(Paragraph(line, normal_style))
                            story.append(Spacer(1, 6))
                else:
                    # Handle plain text files (.txt)
                    logging.info(
                        f"Processing plain text file with {len(text_content)} characters..."
                    )

                    # Split text into lines and process each line
                    lines = text_content.split("\n")
                    line_count = 0

                    for line in lines:
                        line = line.rstrip()
                        line_count += 1

                        # Empty lines
                        if not line.strip():
                            story.append(Spacer(1, 6))
                            continue

                        # Regular text lines
                        # Escape special characters for ReportLab
                        safe_line = (
                            line.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )

                        # Create paragraph
                        story.append(Paragraph(safe_line, normal_style))
                        story.append(Spacer(1, 3))

                    logging.info(f"Added {line_count} lines to PDF")

                    # If no content was added, add a placeholder
                    if not story:
                        story.append(Paragraph("(Empty text file)", normal_style))

                # Build PDF
                doc.build(story)
                logging.info(
                    f"Successfully converted {text_path.name} to PDF ({pdf_path.stat().st_size / 1024:.1f} KB)"
                )

            except ImportError:
                raise RuntimeError(
                    "reportlab is required for text-to-PDF conversion. "
                    "Please install it using: pip install reportlab"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert text file {text_path.name} to PDF: {str(e)}"
                )

            # Validate the generated PDF
            if not pdf_path.exists() or pdf_path.stat().st_size < 100:
                raise RuntimeError(
                    f"PDF conversion failed for {text_path.name} - generated PDF is empty or corrupted."
                )

            return pdf_path

        except Exception as e:
            logging.error(f"Error in convert_text_to_pdf: {str(e)}")
            raise

    @staticmethod
    def _process_inline_markdown(text: str) -> str:
        """
        Process inline markdown formatting (bold, italic, code, links)

        Args:
            text: Raw text with markdown formatting

        Returns:
            Text with ReportLab markup
        """
        import re

        # Escape special characters for ReportLab
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Bold text: **text** or __text__
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)

        # Italic text: *text* or _text_ (but not in the middle of words)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

        # Inline code: `code`
        text = re.sub(
            r"`([^`]+?)`",
            r'<font name="Courier" size="9" color="darkred">\1</font>',
            text,
        )

        # Links: [text](url) - convert to text with URL annotation
        def link_replacer(match):
            link_text = match.group(1)
            url = match.group(2)
            return f'<link href="{url}" color="blue"><u>{link_text}</u></link>'

        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", link_replacer, text)

        # Strikethrough: ~~text~~
        text = re.sub(r"~~(.*?)~~", r"<strike>\1</strike>", text)

        return text

    def parse_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | None = None,
        method: str = "auto",
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Abstract method to parse PDF document.
        Must be implemented by subclasses.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_pdf must be implemented by subclasses")

    def parse_image(
        self,
        image_path: str | Path,
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Abstract method to parse image document.
        Must be implemented by subclasses.

        Note: Different parsers may support different image formats.
        Check the specific parser's documentation for supported formats.

        Args:
            image_path: Path to the image file
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_image must be implemented by subclasses")

    def parse_document(
        self,
        file_path: str | Path,
        method: str = "auto",
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Abstract method to parse a document.
        Must be implemented by subclasses.

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method (auto, txt, ocr)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_document must be implemented by subclasses")

    def check_installation(self) -> bool:
        """
        Abstract method to check if the parser is properly installed.
        Must be implemented by subclasses.

        Returns:
            bool: True if installation is valid, False otherwise
        """
        raise NotImplementedError(
            "check_installation must be implemented by subclasses"
        )


class MineruParser(Parser):
    """
    MinerU 2.0 document parsing utility class

    Supports parsing PDF and image documents, converting the content into structured data
    and generating markdown and JSON output.

    Note: Office documents are no longer directly supported. Please convert them to PDF first.
    """

    __slots__ = ()

    # Class-level logger
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialize MineruParser"""
        super().__init__()

    @staticmethod
    def _run_mineru_command(
        input_path: str | Path,
        output_dir: str | Path,
        method: str = "auto",
        lang: str | None = None,
        backend: str | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        formula: bool = True,
        table: bool = True,
        device: str | None = None,
        source: str | None = None,
        vlm_url: str | None = None,
    ) -> None:
        """
        Run mineru command line tool

        Args:
            input_path: Path to input file or directory
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            backend: Parsing backend
            start_page: Starting page number (0-based)
            end_page: Ending page number (0-based)
            formula: Enable formula parsing
            table: Enable table parsing
            device: Inference device
            source: Model source
            vlm_url: When the backend is `vlm-sglang-client`, you need to specify the server_url
        """
        cmd = [
            "mineru",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-m",
            method,
        ]

        if backend:
            cmd.extend(["-b", backend])
        if source:
            cmd.extend(["--source", source])
        if lang:
            cmd.extend(["-l", lang])
        if start_page is not None:
            cmd.extend(["-s", str(start_page)])
        if end_page is not None:
            cmd.extend(["-e", str(end_page)])
        if not formula:
            cmd.extend(["-f", "false"])
        if not table:
            cmd.extend(["-t", "false"])
        if device:
            cmd.extend(["-d", device])
        if vlm_url:
            cmd.extend(["-u", vlm_url])

        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform
            import threading
            from queue import Empty, Queue

            # Log the command being executed
            logging.info(f"Executing mineru command: {' '.join(cmd)}")

            subprocess_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "ignore",
                "bufsize": 1,  # Line buffered
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            # Function to read output from subprocess and add to queue
            def enqueue_output(pipe, queue, prefix):
                try:
                    for line in iter(pipe.readline, ""):
                        if line.strip():  # Only add non-empty lines
                            queue.put((prefix, line.strip()))
                    pipe.close()
                except Exception as e:
                    queue.put((prefix, f"Error reading {prefix}: {e}"))

            # Start subprocess
            process = subprocess.Popen(cmd, **subprocess_kwargs)
            # Register process for external lifecycle control if a registry is present
            try:
                # Optional global registry set by host application for graceful shutdown
                from raganything.utils import (
                    register_external_process,
                    unregister_external_process,
                )  # type: ignore

                try:
                    register_external_process(process)
                except Exception:
                    pass
            except Exception:
                # registry not available; continue without registration
                pass

            # Create queues for stdout and stderr
            stdout_queue = Queue()
            stderr_queue = Queue()

            # Start threads to read output
            stdout_thread = threading.Thread(
                target=enqueue_output, args=(process.stdout, stdout_queue, "STDOUT")
            )
            stderr_thread = threading.Thread(
                target=enqueue_output, args=(process.stderr, stderr_queue, "STDERR")
            )

            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Process output in real time
            while process.poll() is None:
                # Check stdout queue
                try:
                    while True:
                        prefix, line = stdout_queue.get_nowait()
                        # Log mineru output with INFO level, prefixed with [MinerU]
                        logging.info(f"[MinerU] {line}")
                except Empty:
                    pass

                # Check stderr queue
                try:
                    while True:
                        prefix, line = stderr_queue.get_nowait()
                        # Log mineru errors with WARNING level
                        if "warning" in line.lower():
                            logging.warning(f"[MinerU] {line}")
                        elif "error" in line.lower():
                            logging.error(f"[MinerU] {line}")
                        else:
                            logging.info(f"[MinerU] {line}")
                except Empty:
                    pass

                # Small delay to prevent busy waiting
                import time

                time.sleep(0.1)

            # Process any remaining output after process completion
            try:
                while True:
                    prefix, line = stdout_queue.get_nowait()
                    logging.info(f"[MinerU] {line}")
            except Empty:
                pass

            try:
                while True:
                    prefix, line = stderr_queue.get_nowait()
                    if "warning" in line.lower():
                        logging.warning(f"[MinerU] {line}")
                    elif "error" in line.lower():
                        logging.error(f"[MinerU] {line}")
                    else:
                        logging.info(f"[MinerU] {line}")
            except Empty:
                pass

            # Wait for process to complete and get return code
            return_code = process.wait()
            try:
                unregister_external_process(process)
            except Exception:
                pass

            # Wait for threads to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            if return_code == 0:
                logging.info("[MinerU] Command executed successfully")
            else:
                raise subprocess.CalledProcessError(return_code, cmd)

        except subprocess.CalledProcessError as e:
            logging.error(f"Error running mineru command: {e}")
            logging.error(f"Command: {' '.join(cmd)}")
            logging.error(f"Return code: {e.returncode}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "mineru command not found. Please ensure MinerU 2.0 is properly installed:\n"
                "pip install -U 'mineru[core]' or uv pip install -U 'mineru[core]'"
            )
        except Exception as e:
            logging.error(f"Unexpected error running mineru command: {e}")
            raise

    @staticmethod
    def _read_output_files(
        output_dir: Path, file_stem: str, method: str = "auto"
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Read the output files generated by mineru

        Args:
            output_dir: Output directory
            file_stem: File name without extension

        Returns:
            Tuple containing (content list JSON, Markdown text)
        """
        # Look for the generated files
        md_file = output_dir / f"{file_stem}.md"
        json_file = output_dir / f"{file_stem}_content_list.json"
        images_base_dir = output_dir  # Base directory for images

        file_stem_subdir = output_dir / file_stem
        if file_stem_subdir.exists():
            md_file = file_stem_subdir / method / f"{file_stem}.md"
            json_file = file_stem_subdir / method / f"{file_stem}_content_list.json"
            images_base_dir = file_stem_subdir / method

        # Read markdown content
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                logging.warning(f"Could not read markdown file {md_file}: {e}")

        # Read JSON content list
        content_list = []
        if json_file.exists():
            try:
                with open(json_file, encoding="utf-8") as f:
                    content_list = json.load(f)

                # Always fix relative paths in content_list to absolute paths
                logging.info(
                    f"Fixing image paths in {json_file} with base directory: {images_base_dir}"
                )
                for item in content_list:
                    if isinstance(item, dict):
                        for field_name in [
                            "img_path",
                            "table_img_path",
                            "equation_img_path",
                        ]:
                            if field_name in item and item[field_name]:
                                img_path = item[field_name]
                                absolute_img_path = (
                                    images_base_dir / img_path
                                ).resolve()
                                item[field_name] = str(absolute_img_path)
                                logging.debug(
                                    f"Updated {field_name}: {img_path} -> {item[field_name]}"
                                )

            except Exception as e:
                logging.warning(f"Could not read JSON file {json_file}: {e}")

        return content_list, md_content

    def parse_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | None = None,
        method: str = "auto",
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse PDF document using MinerU 2.0

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object for easier handling
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            name_without_suff = pdf_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = pdf_path.parent / "mineru_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run mineru command
            self._run_mineru_command(
                input_path=pdf_path,
                output_dir=base_output_dir,
                method=method,
                lang=lang,
                **kwargs,
            )

            # Read the generated output files
            backend = kwargs.get("backend", "")
            if backend.startswith("vlm-"):
                method = "vlm"

            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff, method=method
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_pdf: {str(e)}")
            raise

    def parse_image(
        self,
        image_path: str | Path,
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse image document using MinerU 2.0

        Note: MinerU 2.0 natively supports .png, .jpeg, .jpg formats.
        Other formats (.bmp, .tiff, .tif, etc.) will be automatically converted to .png.

        Args:
            image_path: Path to the image file
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object for easier handling
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")

            # Supported image formats by MinerU 2.0
            mineru_supported_formats = {".png", ".jpeg", ".jpg"}

            # All supported image formats (including those we can convert)
            all_supported_formats = {
                ".png",
                ".jpeg",
                ".jpg",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            }

            ext = image_path.suffix.lower()
            if ext not in all_supported_formats:
                raise ValueError(
                    f"Unsupported image format: {ext}. Supported formats: {', '.join(all_supported_formats)}"
                )

            # Determine the actual image file to process
            actual_image_path = image_path
            temp_converted_file = None

            # If format is not natively supported by MinerU, convert it
            if ext not in mineru_supported_formats:
                logging.info(
                    f"Converting {ext} image to PNG for MinerU compatibility..."
                )

                try:
                    from PIL import Image
                except ImportError:
                    raise RuntimeError(
                        "PIL/Pillow is required for image format conversion. "
                        "Please install it using: pip install Pillow"
                    )

                # Create temporary directory for conversion
                temp_dir = Path(tempfile.mkdtemp())
                temp_converted_file = temp_dir / f"{image_path.stem}_converted.png"

                try:
                    # Open and convert image
                    with Image.open(image_path) as img:
                        # Handle different image modes
                        if img.mode in ("RGBA", "LA", "P"):
                            # For images with transparency or palette, convert to RGB first
                            if img.mode == "P":
                                img = img.convert("RGBA")

                            # Create white background for transparent images
                            background = Image.new("RGB", img.size, (255, 255, 255))
                            if img.mode == "RGBA":
                                background.paste(
                                    img, mask=img.split()[-1]
                                )  # Use alpha channel as mask
                            else:
                                background.paste(img)
                            img = background
                        elif img.mode not in ("RGB", "L"):
                            # Convert other modes to RGB
                            img = img.convert("RGB")

                        # Save as PNG
                        img.save(temp_converted_file, "PNG", optimize=True)
                        logging.info(
                            f"Successfully converted {image_path.name} to PNG ({temp_converted_file.stat().st_size / 1024:.1f} KB)"
                        )

                        actual_image_path = temp_converted_file

                except Exception as e:
                    if temp_converted_file and temp_converted_file.exists():
                        temp_converted_file.unlink()
                    raise RuntimeError(
                        f"Failed to convert image {image_path.name}: {str(e)}"
                    )

            name_without_suff = image_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = image_path.parent / "mineru_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Run mineru command (images are processed with OCR method)
                self._run_mineru_command(
                    input_path=actual_image_path,
                    output_dir=base_output_dir,
                    method="ocr",  # Images require OCR method
                    lang=lang,
                    **kwargs,
                )

                # Read the generated output files
                content_list, _ = self._read_output_files(
                    base_output_dir, name_without_suff, method="ocr"
                )
                return content_list

            finally:
                # Clean up temporary converted file if it was created
                if temp_converted_file and temp_converted_file.exists():
                    try:
                        temp_converted_file.unlink()
                        temp_converted_file.parent.rmdir()  # Remove temp directory if empty
                    except Exception:
                        pass  # Ignore cleanup errors

        except Exception as e:
            logging.error(f"Error in parse_image: {str(e)}")
            raise

    def parse_office_doc(
        self,
        doc_path: str | Path,
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse office document by first converting to PDF, then parsing with MinerU 2.0

        Note: This method requires LibreOffice to be installed separately for PDF conversion.
        MinerU 2.0 no longer includes built-in Office document conversion.

        Supported formats: .doc, .docx, .ppt, .pptx, .xls, .xlsx

        Args:
            doc_path: Path to the document file (.doc, .docx, .ppt, .pptx, .xls, .xlsx)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert Office document to PDF using base class method
            pdf_path = self.convert_office_to_pdf(doc_path, output_dir)

            # Parse the converted PDF
            return self.parse_pdf(
                pdf_path=pdf_path, output_dir=output_dir, lang=lang, **kwargs
            )

        except Exception as e:
            logging.error(f"Error in parse_office_doc: {str(e)}")
            raise

    def parse_text_file(
        self,
        text_path: str | Path,
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse text file by first converting to PDF, then parsing with MinerU 2.0

        Supported formats: .txt, .md

        Args:
            text_path: Path to the text file (.txt, .md)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert text file to PDF using base class method
            pdf_path = self.convert_text_to_pdf(text_path, output_dir)

            # Parse the converted PDF
            return self.parse_pdf(
                pdf_path=pdf_path, output_dir=output_dir, lang=lang, **kwargs
            )

        except Exception as e:
            logging.error(f"Error in parse_text_file: {str(e)}")
            raise

    def parse_document(
        self,
        file_path: str | Path,
        method: str = "auto",
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse document using MinerU 2.0 based on file extension

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method (auto, txt, ocr)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for mineru command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        # Convert to Path object
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Get file extension
        ext = file_path.suffix.lower()

        # Choose appropriate parser based on file type
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)
        elif ext in self.IMAGE_FORMATS:
            return self.parse_image(file_path, output_dir, lang, **kwargs)
        elif ext in self.OFFICE_FORMATS:
            logging.warning(
                f"Warning: Office document detected ({ext}). "
                f"MinerU 2.0 requires conversion to PDF first."
            )
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        elif ext in self.TEXT_FORMATS:
            return self.parse_text_file(file_path, output_dir, lang, **kwargs)
        else:
            # For unsupported file types, try as PDF
            logging.warning(
                f"Warning: Unsupported file extension '{ext}', "
                f"attempting to parse as PDF"
            )
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)

    def check_installation(self) -> bool:
        """
        Check if MinerU 2.0 is properly installed

        Returns:
            bool: True if installation is valid, False otherwise
        """
        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform

            subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(["mineru", "--version"], **subprocess_kwargs)
            logging.debug(f"MinerU version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.debug(
                "MinerU 2.0 is not properly installed. "
                "Please install it using: pip install -U 'mineru[core]'"
            )
            return False


class DoclingParser(Parser):
    """
    Docling document parsing utility class.

    Specialized in parsing Office documents and HTML files, converting the content
    into structured data and generating markdown and JSON output.
    """

    # Define Docling-specific formats
    HTML_FORMATS = {".html", ".htm", ".xhtml"}

    def __init__(self) -> None:
        """Initialize DoclingParser"""
        super().__init__()

    def parse_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | None = None,
        method: str = "auto",
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse PDF document using Docling

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object for easier handling
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            name_without_suff = pdf_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = pdf_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=pdf_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_pdf: {str(e)}")
            raise

    def parse_document(
        self,
        file_path: str | Path,
        method: str = "auto",
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse document using Docling based on file extension

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        # Convert to Path object
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Get file extension
        ext = file_path.suffix.lower()

        # Choose appropriate parser based on file type
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)
        elif ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        elif ext in self.HTML_FORMATS:
            return self.parse_html(file_path, output_dir, lang, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Docling only supports PDF files, Office formats ({', '.join(self.OFFICE_FORMATS)}) "
                f"and HTML formats ({', '.join(self.HTML_FORMATS)})"
            )

    def _run_docling_command(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        file_stem: str,
        **kwargs,
    ) -> None:
        """
        Run docling command line tool

        Args:
            input_path: Path to input file or directory
            output_dir: Output directory path
            file_stem: File stem for creating subdirectory
            **kwargs: Additional parameters for docling command
        """
        # Create subdirectory structure similar to MinerU
        file_output_dir = Path(output_dir) / file_stem / "docling"
        file_output_dir.mkdir(parents=True, exist_ok=True)

        cmd_json = [
            "docling",
            "--output",
            str(file_output_dir),
            "--to",
            "json",
            str(input_path),
        ]
        cmd_md = [
            "docling",
            "--output",
            str(file_output_dir),
            "--to",
            "md",
            str(input_path),
        ]

        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform

            docling_subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                docling_subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result_json = subprocess.run(cmd_json, **docling_subprocess_kwargs)
            result_md = subprocess.run(cmd_md, **docling_subprocess_kwargs)
            logging.info("Docling command executed successfully")
            if result_json.stdout:
                logging.debug(f"JSON cmd output: {result_json.stdout}")
            if result_md.stdout:
                logging.debug(f"Markdown cmd output: {result_md.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running docling command: {e}")
            if e.stderr:
                logging.error(f"Error details: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "docling command not found. Please ensure Docling is properly installed."
            )

    def _read_output_files(
        self,
        output_dir: Path,
        file_stem: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Read the output files generated by docling and convert to MinerU format

        Args:
            output_dir: Output directory
            file_stem: File name without extension

        Returns:
            Tuple containing (content list JSON, Markdown text)
        """
        # Use subdirectory structure similar to MinerU
        file_subdir = output_dir / file_stem / "docling"
        md_file = file_subdir / f"{file_stem}.md"
        json_file = file_subdir / f"{file_stem}.json"

        # Read markdown content
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                logging.warning(f"Could not read markdown file {md_file}: {e}")

        # Read JSON content and convert format
        content_list = []
        if json_file.exists():
            try:
                with open(json_file, encoding="utf-8") as f:
                    docling_content = json.load(f)
                    # Convert docling format to minerU format
                    content_list = self.read_from_block_recursive(
                        docling_content["body"],
                        "body",
                        file_subdir,
                        0,
                        "0",
                        docling_content,
                    )
            except Exception as e:
                logging.warning(f"Could not read or convert JSON file {json_file}: {e}")
        return content_list, md_content

    def read_from_block_recursive(
        self,
        block,
        type: str,
        output_dir: Path,
        cnt: int,
        num: str,
        docling_content: dict[str, Any],
    ) -> list[dict[str, Any]]:
        content_list = []
        if not block.get("children"):
            cnt += 1
            content_list.append(self.read_from_block(block, type, output_dir, cnt, num))
        else:
            if type not in ["groups", "body"]:
                cnt += 1
                content_list.append(
                    self.read_from_block(block, type, output_dir, cnt, num)
                )
            members = block["children"]
            for member in members:
                cnt += 1
                member_tag = member["$ref"]
                member_type = member_tag.split("/")[1]
                member_num = member_tag.split("/")[2]
                member_block = docling_content[member_type][int(member_num)]
                content_list.extend(
                    self.read_from_block_recursive(
                        member_block,
                        member_type,
                        output_dir,
                        cnt,
                        member_num,
                        docling_content,
                    )
                )
        return content_list

    def read_from_block(
        self, block, type: str, output_dir: Path, cnt: int, num: str
    ) -> dict[str, Any]:
        if type == "texts":
            if block["label"] == "formula":
                return {
                    "type": "equation",
                    "img_path": "",
                    "text": block["orig"],
                    "text_format": "unkown",
                    "page_idx": cnt // 10,
                }
            else:
                return {
                    "type": "text",
                    "text": block["orig"],
                    "page_idx": cnt // 10,
                }
        elif type == "pictures":
            try:
                base64_uri = block["image"]["uri"]
                base64_str = base64_uri.split(",")[1]
                # Create images directory within the docling subdirectory
                image_dir = output_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                image_path = image_dir / f"image_{num}.png"
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(base64_str))
                return {
                    "type": "image",
                    "img_path": str(image_path.resolve()),  # Convert to absolute path
                    "image_caption": block.get("caption", ""),
                    "image_footnote": block.get("footnote", ""),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                logging.warning(f"Failed to process image {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Image processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }
        else:
            try:
                return {
                    "type": "table",
                    "img_path": "",
                    "table_caption": block.get("caption", ""),
                    "table_footnote": block.get("footnote", ""),
                    "table_body": block.get("data", []),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                logging.warning(f"Failed to process table {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Table processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }

    def parse_office_doc(
        self,
        doc_path: str | Path,
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse office document directly using Docling

        Supported formats: .doc, .docx, .ppt, .pptx, .xls, .xlsx

        Args:
            doc_path: Path to the document file
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Document file does not exist: {doc_path}")

            if doc_path.suffix.lower() not in self.OFFICE_FORMATS:
                raise ValueError(f"Unsupported office format: {doc_path.suffix}")

            name_without_suff = doc_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=doc_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_office_doc: {str(e)}")
            raise

    def parse_html(
        self,
        html_path: str | Path,
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse HTML document using Docling

        Supported formats: .html, .htm, .xhtml

        Args:
            html_path: Path to the HTML file
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object
            html_path = Path(html_path)
            if not html_path.exists():
                raise FileNotFoundError(f"HTML file does not exist: {html_path}")

            if html_path.suffix.lower() not in self.HTML_FORMATS:
                raise ValueError(f"Unsupported HTML format: {html_path.suffix}")

            name_without_suff = html_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = html_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=html_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_html: {str(e)}")
            raise

    def check_installation(self) -> bool:
        """
        Check if Docling is properly installed

        Returns:
            bool: True if installation is valid, False otherwise
        """
        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform

            subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(["docling", "--version"], **subprocess_kwargs)
            logging.debug(f"Docling version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.debug(
                "Docling is not properly installed. "
                "Please ensure it is installed correctly."
            )
            return False


class SimpleTextParser(Parser):
    """
    Fast text parser for plain text and markdown files (.txt, .md).

    Processes text files directly without conversion to PDF, providing
    significantly faster processing for simple text documents.
    """

    def __init__(self) -> None:
        """Initialize SimpleTextParser."""
        super().__init__()

    def parse_document(
        self,
        file_path: str | Path,
        method: str = "auto",
        output_dir: str | None = None,
        lang: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse document using SimpleTextParser based on file extension.

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method (ignored for text files)
            output_dir: Output directory path (optional)
            lang: Document language (ignored for text files)
            **kwargs: Additional parameters

        Returns:
            List[Dict[str, Any]]: List of content blocks

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file extension is not supported
        """
        # Convert to Path object
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Get file extension
        ext = file_path.suffix.lower()

        # Route to appropriate parser based on file type
        if ext == ".md":
            return self._parse_markdown_direct(file_path, output_dir, **kwargs)
        elif ext == ".txt":
            return self._parse_text_direct(file_path, output_dir, **kwargs)
        else:
            raise ValueError(
                f"SimpleTextParser only supports .txt and .md files, got: {ext}"
            )

    def _parse_markdown_direct(
        self,
        md_path: Path,
        output_dir: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse markdown file directly without PDF conversion.

        Args:
            md_path: Path to the markdown file
            output_dir: Output directory (optional, for compatibility)
            **kwargs: Additional parameters

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            logging.info(f"Parsing markdown file directly: {md_path.name}")

            # Read the markdown content
            content = self._read_text_file(md_path)

            # Parse markdown structure
            content_blocks = self._parse_markdown_content(content, md_path.stem)

            logging.info(
                f"Successfully parsed {len(content_blocks)} content blocks from markdown"
            )
            return content_blocks

        except Exception as e:
            logging.error(f"Error parsing markdown file {md_path}: {str(e)}")
            raise

    def _parse_text_direct(
        self,
        txt_path: Path,
        output_dir: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse plain text file directly.

        Args:
            txt_path: Path to the text file
            output_dir: Output directory (optional, for compatibility)
            **kwargs: Additional parameters

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            logging.info(f"Parsing text file directly: {txt_path.name}")

            # Read the text content
            content = self._read_text_file(txt_path)

            # Parse text content into blocks
            content_blocks = self._parse_text_content(content, txt_path.stem)

            logging.info(
                f"Successfully parsed {len(content_blocks)} content blocks from text"
            )
            return content_blocks

        except Exception as e:
            logging.error(f"Error parsing text file {txt_path}: {str(e)}")
            raise

    def _read_text_file(self, file_path: Path) -> str:
        """
        Read text file with multiple encoding fallbacks.

        Args:
            file_path: Path to the text file

        Returns:
            str: File content

        Raises:
            RuntimeError: If file cannot be decoded with any supported encoding
        """
        encodings_to_try = ["utf-8", "utf-8-sig", "gbk", "latin-1", "cp1252"]

        for encoding in encodings_to_try:
            try:
                with open(file_path, encoding=encoding) as f:
                    content = f.read()

                if encoding != "utf-8":
                    logging.info(f"Successfully read file with {encoding} encoding")

                return content

            except UnicodeDecodeError:
                continue

        # If all encodings fail
        raise RuntimeError(
            f"Could not decode text file {file_path.name} with any supported encoding: "
            f"{', '.join(encodings_to_try)}"
        )

    def _parse_markdown_content(
        self, content: str, file_stem: str
    ) -> list[dict[str, Any]]:
        """
        Parse markdown content into structured blocks.

        Args:
            content: Raw markdown content
            file_stem: File name without extension (for metadata)

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        blocks = []
        lines = content.split("\n")
        current_block = []
        current_type = "text"
        block_index = 0
        line_number = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            line_number += 1
            stripped_line = line.strip()

            # Skip empty lines but track them for paragraph breaks
            if not stripped_line:
                if current_block:
                    # Finish current block
                    block_content = "\n".join(current_block).strip()
                    if block_content:
                        blocks.append(
                            self._create_content_block(
                                current_type,
                                block_content,
                                block_index,
                                file_stem,
                                line_number - len(current_block),
                            )
                        )
                        block_index += 1
                    current_block = []
                    current_type = "text"
                i += 1
                continue

            # Detect headers
            if stripped_line.startswith("#"):
                # Finish previous block if any
                if current_block:
                    block_content = "\n".join(current_block).strip()
                    if block_content:
                        blocks.append(
                            self._create_content_block(
                                current_type,
                                block_content,
                                block_index,
                                file_stem,
                                line_number - len(current_block),
                            )
                        )
                        block_index += 1

                # Parse header
                level = len(stripped_line) - len(stripped_line.lstrip("#"))
                header_text = stripped_line.lstrip("#").strip()

                if header_text:
                    blocks.append(
                        self._create_header_block(
                            header_text, level, block_index, file_stem, line_number
                        )
                    )
                    block_index += 1

                current_block = []
                current_type = "text"

            # Detect code blocks
            elif stripped_line.startswith("```"):
                # Finish previous block if any
                if current_block:
                    block_content = "\n".join(current_block).strip()
                    if block_content:
                        blocks.append(
                            self._create_content_block(
                                current_type,
                                block_content,
                                block_index,
                                file_stem,
                                line_number - len(current_block),
                            )
                        )
                        block_index += 1

                # Parse code block
                code_lang = stripped_line[3:].strip()
                code_lines = []
                i += 1
                code_start_line = line_number + 1

                while i < len(lines):
                    line_number += 1
                    if lines[i].strip().startswith("```"):
                        break
                    code_lines.append(lines[i])
                    i += 1

                code_content = "\n".join(code_lines)
                if code_content.strip():
                    blocks.append(
                        self._create_code_block(
                            code_content,
                            code_lang,
                            block_index,
                            file_stem,
                            code_start_line,
                        )
                    )
                    block_index += 1

                current_block = []
                current_type = "text"

            # Detect list items
            elif stripped_line.startswith(("- ", "* ", "+ ")) or (
                stripped_line
                and stripped_line[0].isdigit()
                and ". " in stripped_line[:5]
            ):
                if current_type != "list":
                    # Finish previous block if it's not a list
                    if current_block:
                        block_content = "\n".join(current_block).strip()
                        if block_content:
                            blocks.append(
                                self._create_content_block(
                                    current_type,
                                    block_content,
                                    block_index,
                                    file_stem,
                                    line_number - len(current_block),
                                )
                            )
                            block_index += 1
                        current_block = []
                    current_type = "list"

                current_block.append(line)

            else:
                # Regular text
                if current_type != "text":
                    # Finish previous block if it's not text
                    if current_block:
                        block_content = "\n".join(current_block).strip()
                        if block_content:
                            blocks.append(
                                self._create_content_block(
                                    current_type,
                                    block_content,
                                    block_index,
                                    file_stem,
                                    line_number - len(current_block),
                                )
                            )
                            block_index += 1
                        current_block = []
                    current_type = "text"

                current_block.append(line)

            i += 1

        # Handle remaining content
        if current_block:
            block_content = "\n".join(current_block).strip()
            if block_content:
                blocks.append(
                    self._create_content_block(
                        current_type,
                        block_content,
                        block_index,
                        file_stem,
                        line_number - len(current_block),
                    )
                )

        return blocks

    def _parse_text_content(self, content: str, file_stem: str) -> list[dict[str, Any]]:
        """
        Parse plain text content into structured blocks.

        Args:
            content: Raw text content
            file_stem: File name without extension (for metadata)

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        blocks = []

        # Split content into paragraphs (separated by empty lines)
        paragraphs = []
        current_paragraph = []

        for line in content.split("\n"):
            stripped_line = line.strip()
            if stripped_line:
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []

        # Don't forget the last paragraph
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # Convert paragraphs to blocks
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                blocks.append(
                    self._create_content_block(
                        "text", paragraph.strip(), i, file_stem, 0
                    )
                )

        return blocks

    def _create_content_block(
        self,
        block_type: str,
        content: str,
        index: int,
        file_stem: str,
        line_number: int,
    ) -> dict[str, Any]:
        """Create a standard content block in LightRAG-compatible format."""
        # Use the correct field name for LightRAG compatibility
        if block_type == "text":
            return {
                "type": "text",
                "text": content,  # LightRAG expects 'text' field for text content
                "page_idx": 0,  # Default page index
                "metadata": {
                    "block_index": index,
                    "source_file": file_stem,
                    "line_start": line_number,
                    "parser": "SimpleTextParser",
                },
            }
        elif block_type == "list":
            return {
                "type": "text",  # Lists are treated as text in LightRAG
                "text": content,
                "page_idx": 0,
                "metadata": {
                    "block_index": index,
                    "source_file": file_stem,
                    "line_start": line_number,
                    "content_type": "list",
                    "parser": "SimpleTextParser",
                },
            }
        else:
            # For code and other types, also use text format but preserve type info
            return {
                "type": "text",
                "text": content,
                "page_idx": 0,
                "metadata": {
                    "block_index": index,
                    "source_file": file_stem,
                    "line_start": line_number,
                    "content_type": block_type,
                    "parser": "SimpleTextParser",
                },
            }

    def _create_header_block(
        self, content: str, level: int, index: int, file_stem: str, line_number: int
    ) -> dict[str, Any]:
        """Create a header block with level metadata in LightRAG-compatible format."""
        return {
            "type": "text",  # Headers are treated as text in LightRAG
            "text": content,
            "page_idx": 0,
            "metadata": {
                "block_index": index,
                "source_file": file_stem,
                "line_start": line_number,
                "content_type": "header",
                "header_level": level,
                "parser": "SimpleTextParser",
            },
        }

    def _create_code_block(
        self, content: str, language: str, index: int, file_stem: str, line_number: int
    ) -> dict[str, Any]:
        """Create a code block with language metadata in LightRAG-compatible format."""
        return {
            "type": "text",  # Code blocks are treated as text in LightRAG
            "text": content,
            "page_idx": 0,
            "metadata": {
                "block_index": index,
                "source_file": file_stem,
                "line_start": line_number,
                "content_type": "code",
                "language": language or "text",
                "parser": "SimpleTextParser",
            },
        }

    def check_installation(self) -> bool:
        """
        Check if SimpleTextParser can be used.

        Since SimpleTextParser only uses standard library,
        it's always available.

        Returns:
            bool: Always True for SimpleTextParser
        """
        return True


def main():
    """
    Main function to run the document parser from command line
    """
    parser = argparse.ArgumentParser(
        description="Parse documents using MinerU 2.0 or Docling"
    )
    parser.add_argument("file_path", help="Path to the document to parse")
    parser.add_argument("--output", "-o", help="Output directory path")
    parser.add_argument(
        "--method",
        "-m",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="Parsing method (auto, txt, ocr)",
    )
    parser.add_argument(
        "--lang",
        "-l",
        help="Document language for OCR optimization (e.g., ch, en, ja)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=[
            "pipeline",
            "vlm-transformers",
            "vlm-sglang-engine",
            "vlm-sglang-client",
        ],
        default="pipeline",
        help="Parsing backend",
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Inference device (e.g., cpu, cuda, cuda:0, npu, mps)",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope", "local"],
        default="huggingface",
        help="Model source",
    )
    parser.add_argument(
        "--no-formula",
        action="store_true",
        help="Disable formula parsing",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Disable table parsing",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Display content statistics"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check parser installation",
    )
    parser.add_argument(
        "--parser",
        choices=["mineru", "docling"],
        default="mineru",
        help="Parser selection",
    )
    parser.add_argument(
        "--vlm_url",
        help="When the backend is `vlm-sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`",
    )

    args = parser.parse_args()

    # Check installation if requested
    if args.check:
        doc_parser = DoclingParser() if args.parser == "docling" else MineruParser()
        if doc_parser.check_installation():
            print(f"✅ {args.parser.title()} is properly installed")
            return 0
        else:
            print(f"❌ {args.parser.title()} installation check failed")
            return 1

    try:
        # Parse the document
        doc_parser = DoclingParser() if args.parser == "docling" else MineruParser()
        content_list = doc_parser.parse_document(
            file_path=args.file_path,
            method=args.method,
            output_dir=args.output,
            lang=args.lang,
            backend=args.backend,
            device=args.device,
            source=args.source,
            formula=not args.no_formula,
            table=not args.no_table,
            vlm_url=args.vlm_url,
        )

        print(f"✅ Successfully parsed: {args.file_path}")
        print(f"📊 Extracted {len(content_list)} content blocks")

        # Display statistics if requested
        if args.stats:
            print("\n📈 Document Statistics:")
            print(f"Total content blocks: {len(content_list)}")

            # Count different types of content
            content_types = {}
            for item in content_list:
                if isinstance(item, dict):
                    content_type = item.get("type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1

            if content_types:
                print("\n📋 Content Type Distribution:")
                for content_type, count in sorted(content_types.items()):
                    print(f"  • {content_type}: {count}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

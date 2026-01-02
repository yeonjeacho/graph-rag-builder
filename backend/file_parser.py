"""
File Parser Module - Extract text from various file formats
Supports: PDF, TXT, MD, DOCX
"""
import io
from typing import Optional


def parse_pdf(file_content: bytes) -> str:
    """Extract text from PDF file using PyMuPDF (fitz)"""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(stream=file_content, filetype="pdf")
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())
        
        doc.close()
        return "\n\n".join(text_parts).strip()
    except Exception as e:
        raise Exception(f"PDF 파싱 실패: {str(e)}")


def parse_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        from docx import Document
        
        doc = Document(io.BytesIO(file_content))
        text_parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        return "\n\n".join(text_parts).strip()
    except Exception as e:
        raise Exception(f"DOCX 파싱 실패: {str(e)}")


def parse_text(file_content: bytes, encoding: str = "utf-8") -> str:
    """Extract text from plain text files (TXT, MD)"""
    try:
        # Try UTF-8 first, then fall back to other encodings
        encodings = [encoding, "utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]
        
        for enc in encodings:
            try:
                return file_content.decode(enc).strip()
            except UnicodeDecodeError:
                continue
        
        # Last resort: decode with errors ignored
        return file_content.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        raise Exception(f"텍스트 파싱 실패: {str(e)}")


def parse_file(filename: str, file_content: bytes) -> str:
    """
    Parse file and extract text based on file extension
    
    Args:
        filename: Original filename with extension
        file_content: Raw file bytes
        
    Returns:
        Extracted text content
    """
    if not filename or not file_content:
        raise ValueError("파일명과 파일 내용이 필요합니다")
    
    # Get file extension
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    
    if ext == "pdf":
        return parse_pdf(file_content)
    elif ext == "docx":
        return parse_docx(file_content)
    elif ext in ["txt", "md", "markdown", "text"]:
        return parse_text(file_content)
    else:
        # Try to parse as plain text for unknown extensions
        try:
            return parse_text(file_content)
        except:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")


def get_supported_extensions() -> list:
    """Return list of supported file extensions"""
    return ["pdf", "txt", "md", "docx"]

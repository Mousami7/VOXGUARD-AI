from pathlib import Path

from pypdf import PdfReader


def extract_text(file_path: str) -> str:
    """Read text from a local .txt, .md, or .pdf file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace").strip()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()

    raise ValueError(f"Unsupported document type: {suffix}")

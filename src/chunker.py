import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pdf_structured_recursive(
    pdf_path: str,
    chunk_size: int = 1000,
) -> list[dict]:
    """
    1. Structured step: Extracts text blocks page-by-page using PyMuPDF (fitz).
    2. Recursive step: Splits oversized blocks into bounded chunks with zero overlap
       to minimize duplicate triplet extraction.
    """
    doc = fitz.open(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    chunk_id = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        # block tuple: (x0, y0, x1, y1, text, block_no, block_type, block_bbox)
        # block_type == 0 indicates a text block (excludes images)
        for block in page.get_text("blocks"):
            if block[6] != 0:
                continue

            raw_text = block[4].strip()
            if not raw_text:
                continue

            # Normalize internal line wraps inside a block into clean spaces/paragraphs
            normalized_block = " ".join(line.strip() for line in raw_text.splitlines() if line.strip())

            # Recursively split only if the block exceeds chunk_size
            sub_chunks = splitter.split_text(normalized_block)
            for sub in sub_chunks:
                cleaned = sub.strip()
                if cleaned:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "page": page_num + 1,
                        "text": cleaned,
                    })
                    chunk_id += 1

    doc.close()
    return chunks


if __name__ == "__main__":
    sample_pdf = "data/sample_paper.pdf"
    try:
        chunks = chunk_pdf_structured_recursive(sample_pdf, chunk_size=1000)
        print(f"[SUCCESS] Generated {len(chunks)} chunks.")
        for c in chunks[:3]:
            print(f"Chunk {c['chunk_id']} (Page {c['page']}): {c['text'][:140]}...")
    except Exception as e:
        print(f"[ERROR] {e}")

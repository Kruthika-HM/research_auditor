import pymupdf4llm
from pathlib import Path

def convert_pdf_to_markdown(pdf_path: str, output_path: str) -> str:
    """
    Parses a research PDF file into clean Markdown format using pymupdf4llm
    and saves it to the specified output path.
    """
    print(f"Parsing PDF: {pdf_path}...")

    # Convert PDF pages directly to markdown text
    md_text = pymupdf4llm.to_markdown(r"C:\projects\research auditor\data\raw_pdfs\bridge_inspec RP.pdf")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save markdown text to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Successfully saved Markdown to: {output_path}")
    return md_text

if __name__ == "__main__":
    sample_pdf = "data/raw_pdfs/sample_paper.pdf"
    output_md = "data/processed_markdown/sample_paper.md"

    convert_pdf_to_markdown(sample_pdf, output_md)
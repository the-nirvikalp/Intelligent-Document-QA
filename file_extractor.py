
import io
import csv
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook


# =====================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH, override=True)


# =====================================================
# 2. PDF EXTRACTION
# =====================================================

def extract_pdf(file_bytes):
    """Extract text page by page from a PDF."""

    reader = PdfReader(io.BytesIO(file_bytes))
    extracted = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            extracted.append({
                "location": f"Page {page_number}",
                "text": text
            })

    return extracted


# =====================================================
# 3. WORD DOCUMENT EXTRACTION
# =====================================================

def extract_docx(file_bytes):
    """Extract paragraphs and tables from DOCX."""

    document = Document(io.BytesIO(file_bytes))
    content = []

    # Extract paragraphs
    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            content.append(paragraph.text)

    # Extract tables
    for table_number, table in enumerate(document.tables, start=1):

        content.append(f"\nTable {table_number}:")

        for row in table.rows:

            row_data = [
                cell.text.strip()
                for cell in row.cells
            ]

            content.append(" | ".join(row_data))

    text = "\n".join(content)

    if text.strip():
        return [{
            "location": "Word Document",
            "text": text
        }]

    return []


# =====================================================
# 4. CSV EXTRACTION
# =====================================================

def extract_csv(file_bytes):
    """Extract CSV content."""

    decoded = file_bytes.decode("utf-8-sig")

    reader = csv.reader(io.StringIO(decoded))

    rows = list(reader)

    text = "\n".join(
        " | ".join(row)
        for row in rows
    )

    if text.strip():
        return [{
            "location": "CSV Data",
            "text": text
        }]

    return []


# =====================================================
# 5. EXCEL EXTRACTION
# =====================================================

def extract_excel(file_bytes):
    """Extract data from XLSX sheets."""

    workbook = load_workbook(
        io.BytesIO(file_bytes),
        read_only=True,
        data_only=True
    )

    extracted = []

    for sheet in workbook.worksheets:

        rows = []

        for row in sheet.iter_rows(values_only=True):

            values = [
                str(value) if value is not None else ""
                for value in row
            ]

            if any(values):
                rows.append(" | ".join(values))

        text = "\n".join(rows)

        if text.strip():
            extracted.append({
                "location": f"Sheet: {sheet.title}",
                "text": text
            })

    workbook.close()

    return extracted


# =====================================================
# 6. POWERPOINT EXTRACTION
# =====================================================

def extract_pptx(file_bytes):
    """Extract text slide by slide from PPTX."""

    presentation = Presentation(io.BytesIO(file_bytes))

    extracted = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):

        content = []

        for shape in slide.shapes:

            # Extract normal text
            if shape.has_text_frame and shape.text.strip():
                content.append(shape.text)

            # Extract table content
            if shape.has_table:

                for row in shape.table.rows:

                    row_data = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    content.append(" | ".join(row_data))

        text = "\n".join(content)

        if text.strip():
            extracted.append({
                "location": f"Slide {slide_number}",
                "text": text
            })

    return extracted


# =====================================================
# 7. TEXT FILE EXTRACTION
# =====================================================

def extract_text_file(file_bytes, extension):
    """Extract TXT, Markdown and JSON content."""

    text = file_bytes.decode("utf-8-sig")

    if extension == ".json":

        try:
            data = json.loads(text)

            text = json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            )

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON file.")

    if text.strip():
        return [{
            "location": "Text Document",
            "text": text
        }]

    return []


# =====================================================
# 8. IMAGE EXTRACTION USING GEMINI VISION
# =====================================================

def extract_image(file_bytes, extension):
    """
    Extract printed and handwritten text from images
    using Gemini Vision.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file."
        )

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff"
    }

    mime_type = mime_types.get(extension)

    if not mime_type:
        raise ValueError("Unsupported image format.")

    client = genai.Client(api_key=api_key)

    prompt = """
    Extract all readable text from this image.

    Instructions:

    1. Extract both printed and handwritten text.
    2. Preserve headings and paragraphs.
    3. Preserve lists and numbering.
    4. Extract table content in a structured format.
    5. Do not invent missing or unclear words.
    6. Mark unreadable sections as [Unclear text].
    7. Preserve the original language.
    8. Return only the extracted content.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            types.Part.from_bytes(
                data=file_bytes,
                mime_type=mime_type
            ),
            prompt
        ]
    )

    extracted_text = response.text or ""

    if not extracted_text.strip():
        raise ValueError(
            "No readable text extracted from image."
        )

    return [{
        "location": "Image",
        "text": extracted_text
    }]


# =====================================================
# 9. MAIN FILE EXTRACTION FUNCTION
# =====================================================

def extract_file_content(uploaded_file):
    """
    Detect file extension and call the appropriate
    extraction function.

    Returns:
        List of dictionaries containing:
        - location
        - text
    """

    filename = uploaded_file.name

    extension = "." + filename.lower().rsplit(".", 1)[-1]

    file_bytes = uploaded_file.getvalue()

    # PDF
    if extension == ".pdf":

        extracted = extract_pdf(file_bytes)

    # Word
    elif extension == ".docx":

        extracted = extract_docx(file_bytes)

    # CSV
    elif extension == ".csv":

        extracted = extract_csv(file_bytes)

    # Excel
    elif extension == ".xlsx":

        extracted = extract_excel(file_bytes)

    # PowerPoint
    elif extension == ".pptx":

        extracted = extract_pptx(file_bytes)

    # Text files
    elif extension in [".txt", ".md", ".json"]:

        extracted = extract_text_file(
            file_bytes,
            extension
        )

    # Images
    elif extension in [
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif"
    ]:

        extracted = extract_image(
            file_bytes,
            extension
        )

    else:

        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    # Validate extraction
    if not extracted:

        raise ValueError(
            "No readable text found in this file."
        )

    return extracted

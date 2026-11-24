
import sqlite3
import os

# -----------------------------------
# DATABASE CONFIGURATION
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "document_qa.db"
)


# -----------------------------------
# DATABASE CONNECTION
# -----------------------------------

def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# -----------------------------------
# INITIALIZE DATABASE
# -----------------------------------

def initialize_database():

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                page_number INTEGER NOT NULL DEFAULT 0,
                chunk_text TEXT NOT NULL,

                FOREIGN KEY (document_id)
                REFERENCES documents(document_id)
                ON DELETE CASCADE,

                UNIQUE(document_id, chunk_number)
            )
        """)

        # Add source_location to older databases without
        # deleting existing records.
        cursor.execute("PRAGMA table_info(chunks)")
        columns = [row[1] for row in cursor.fetchall()]

        if "source_location" not in columns:
            cursor.execute("""
                ALTER TABLE chunks
                ADD COLUMN source_location TEXT
            """)

        # Populate source locations for older PDF chunks.
        cursor.execute("""
            UPDATE chunks
            SET source_location = 'Page ' || page_number
            WHERE source_location IS NULL
              AND page_number > 0
        """)


# -----------------------------------
# SAVE DOCUMENT AND CHUNKS
# -----------------------------------

def save_document_chunks(document_id, filename, chunks):

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO documents (document_id, filename)
            VALUES (?, ?)

            ON CONFLICT(document_id)
            DO UPDATE SET filename = excluded.filename
        """, (document_id, filename))

        cursor.execute("""
            DELETE FROM chunks
            WHERE document_id = ?
        """, (document_id,))

        chunk_data = []

        for chunk in chunks:

            location = chunk.get(
                "source_location",
                chunk.get("location", "")
            )

            # Keep the old page_number field compatible
            # with the existing database structure.
            page_number = chunk.get("page_number", 0)

            chunk_data.append((
                document_id,
                chunk["chunk_number"],
                page_number,
                chunk["text"],
                location
            ))

        cursor.executemany("""
            INSERT INTO chunks (
                document_id,
                chunk_number,
                page_number,
                chunk_text,
                source_location
            )
            VALUES (?, ?, ?, ?, ?)
        """, chunk_data)


# -----------------------------------
# RETRIEVE DOCUMENT CHUNKS
# -----------------------------------

def get_document_chunks(document_id):

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                chunk_number,
                page_number,
                chunk_text,
                source_location
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_number
        """, (document_id,))

        rows = cursor.fetchall()

    return [
        {
            "chunk_number": row[0],
            "page_number": row[1],
            "text": row[2],
            "source_location": (
                row[3] or (
                    f"Page {row[1]}" if row[1] else "Document"
                )
            )
        }
        for row in rows
    ]


# -----------------------------------
# GET DOCUMENT FILENAME
# -----------------------------------

def get_document_filename(document_id):

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT filename
            FROM documents
            WHERE document_id = ?
        """, (document_id,))

        result = cursor.fetchone()

    return result[0] if result else "Unknown document"


# -----------------------------------
# GET ALL DOCUMENTS
# -----------------------------------

def get_all_documents():

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                document_id,
                filename,
                upload_date
            FROM documents
            ORDER BY upload_date DESC
        """)

        rows = cursor.fetchall()

    return [
        {
            "document_id": row[0],
            "filename": row[1],
            "upload_date": row[2]
        }
        for row in rows
    ]


# -----------------------------------
# TEST DATABASE
# -----------------------------------

if __name__ == "__main__":

    initialize_database()

    print("Database initialized successfully!")
    print("Database location:", DATABASE_PATH)


from file_extractor import extract_file_content
import os


# Get file path
file_path = input(
    "Enter the full path of your file: "
).strip().strip('"').strip()


# Check whether file exists
if not os.path.isfile(file_path):
    print("\nERROR: File not found.")
    print("Check the file path and try again.")
    exit()


# Simulate Streamlit uploaded file
class UploadedFile:

    def __init__(self, path):
        self.name = os.path.basename(path)

        with open(path, "rb") as file:
            self.data = file.read()

    def getvalue(self):
        return self.data


uploaded_file = UploadedFile(file_path)


# Extract content
try:

    result = extract_file_content(uploaded_file)

    print("\nExtraction successful!")
    print("File:", uploaded_file.name)
    print("Number of sections:", len(result))

    for section in result:

        print("\nLocation:", section["location"])
        print("Text preview:")
        print(section["text"][:1000])
        print("-" * 50)

except Exception as error:

    print("\nExtraction failed:")
    print(type(error).__name__, error)

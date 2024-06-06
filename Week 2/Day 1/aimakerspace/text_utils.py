import os
from typing import List
import PyPDF2  # Change 1: Import PyPDF2 for PDF handling

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_text_file()
            elif self.path.endswith(".pdf"):  # Change 2: Check if the file is a PDF
                self.load_pdf_file()
            else:
                raise ValueError("Provided path is neither a valid .txt nor a .pdf file.")
        else:
            raise ValueError("Provided path is not valid.")

    def load_text_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf_file(self):  # Change 3: Add method to load PDF files
        with open(self.path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            self.documents.append(text)

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(os.path.join(root, file), "r", encoding=self.encoding) as f:
                        self.documents.append(f.read())
                elif file.endswith(".pdf"):  # Change 4: Handle PDF files in directory
                    with open(os.path.join(root, file), "rb") as f:
                        reader = PyPDF2.PdfFileReader(f)
                        text = ""
                        for page_num in range(reader.numPages):
                            page = reader.getPage(page_num)
                            text += page.extract_text()
                        self.documents.append(text)

    def load_documents(self):
        self.load()
        return self.documents

class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

""" if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1]) """

if __name__ == "__main__":
    # Create a TextFileLoader instance with a path to a sample PDF file
    loader = TextFileLoader("data/agents_whitepaper.pdf")  # Update this path to your PDF file

    # Load the documents (this should handle the PDF)
    loader.load()
    
    # Print the loaded documents to verify
    print("Loaded documents:")
    for doc in loader.documents:
        print(doc[:1000])  # Print the first 1000 characters to verify, to avoid overwhelming output
    
    # Split the loaded text into chunks
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    
    # Print some chunks to verify
    print("Number of chunks:", len(chunks))
    print("First chunk:", chunks[0])
    print("--------")
    print("Second chunk:", chunks[1])
    print("--------")
    print("Second to last chunk:", chunks[-2])
    print("--------")
    print("Last chunk:", chunks[-1])
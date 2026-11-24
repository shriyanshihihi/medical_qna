"""Step 1 of the pipeline: split long text into overlapping chunks.

CV phrase: 'recursiveTextSplitter'  (real class: RecursiveCharacterTextSplitter in LangChain).
It tries separators in order  ["\n\n", "\n", " ", ""]  : paragraph -> line -> word -> character,
so it keeps sentences/paragraphs together as long as possible.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def word_count(text: str) -> int:
    return len(text.split())


def chunk_documents(docs, chunk_size: int, overlap: int):
    """docs -> list of chunks: {'id', 'source', 'text'}"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=word_count,          # size counted in words (approx. tokens)
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for d in docs:
        for piece in splitter.split_text(d["text"]):
            text = piece.strip().lstrip(". ").strip()   # remove separator left-overs like a leading ". "
            if text:
                chunks.append({"id": len(chunks), "source": d["source"], "text": text})
    return chunks

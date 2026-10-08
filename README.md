# Financial Report Extraction & Q&A (RAG)

A small retrieval-augmented generation (RAG) pipeline that reads a company's annual report (10-K PDF), extracts key financial metrics as JSON, stores them in a database, and answers questions about the report through a simple Gradio interface.

Built as a learning project, following a tutorial by AaiTech. I replaced the tutorial's FastAPI/HTML front end with Gradio and ran everything in a Kaggle notebook.

## How it was run

Everything was developed and run in a **Kaggle notebook** (free GPU, ~15 GB). The notebook uses Kaggle-specific paths such as `/kaggle/input/...` and `/kaggle/working/...`. To run it elsewhere, change those paths. The PDF is not included in this repo.

## Pipeline

1. **PDF to Markdown** with `pymupdf4llm`
2. **Chunking** with LangChain `RecursiveCharacterTextSplitter` (1000 characters, 150 overlap). Company and year are attached to each chunk as metadata.
3. **Embeddings** with `sentence-transformers/all-MiniLM-L6-v2`, stored in a **FAISS** index
4. **Retrieval** of the most relevant chunks for a query, with a token budget so the prompt never exceeds GPU memory
5. **Extraction** with `Qwen/Qwen2.5-1.5B-Instruct` (via Hugging Face `transformers`), prompted to return JSON with revenue, net income, operating income, operating cash flow, total assets, total liabilities, top risk factors and top growth drivers
6. **Storage** in a SQLite database through SQLAlchemy (`save_metrics` / `get_metrics`)
7. **Interface**: a Gradio app with two tabs, one for asking questions (it shows the retrieved evidence next to each answer) and one for viewing the saved metrics

## Tools used

Python, Kaggle Notebooks, PyMuPDF (`pymupdf4llm`), LangChain, FAISS (`faiss-cpu`), Sentence-Transformers, Hugging Face Transformers, PyTorch, SQLAlchemy, SQLite, Gradio

## Known limitations

- The extraction model is small (1.5B parameters). On the first run, 4 of the 8 fields matched the report. Total assets and total liabilities were wrong because the right balance sheet chunk was not retrieved, and the model answered anyway instead of returning null.
- The risk factors and growth drivers have not been checked against the report text.
- Questions that need information from several places in the document often fail.
- The pipeline has only been tested on one report (Apple, fiscal year 2024).

## Planned improvements

- Retrieve separately for each field instead of one combined query
- Check that every extracted number actually appears in the retrieved text, and set it to null if not
- Measure accuracy against values checked by hand from the PDF
- Test on more companies and years

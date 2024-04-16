# Transforming a Local RAG into a BentoML Web Service

This is the second tutorial of this BentoML RAG example project. In the [last tutorial](../00-simple-local-rag/), we used a simple script to generate a documentation index from text files in a folder. In this tutorial, we will turn this script into a web service.

The script in the last tutorial includes two steps:

1. Create an index and populate the index with documents
2. Query the index for information.

When converting it to a web service, one solution is to divide the first step into two separate steps:

1. Create an empty index.
2. Enable users to populate the index by uploading files.
3. Allow users to query the populated index.

To simplify this process, we will use [BentoML](https://github.com/bentoml/BentoML).

## Define a BentoML Service

A BentoML Service is defined as a deployable and scalable unit through a Python class, annotated with the `@bentoml.service` decorator. This class manages Service states and their lifecycles. By convention, BentoML Services are defined in `service.py`.

The `service.py` file required for this tutorial already exists in the directory. Here is a breakdown of the file.

### Service initialization

In the Service's `__init__` method, we create the index and set up some global settings.

```python
@bentoml.service(
    traffic={"timeout": 600},
)
class RAGService:

    def __init__(self):
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        from llama_index.core import Settings
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        Settings.node_parser = self.text_splitter

        index = VectorStoreIndex.from_documents([])
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        self.index = load_index_from_storage(storage_context)

        ...
```

### APIs for document ingestion and querying

Each functionality (uploading text, PDFs, and querying the index) is exposed as an HTTP endpoint using the `@bentoml.api` decorator.

#### Text and PDF ingestion

Users can upload documents either as plain text or PDFs, which are then integrated into the index.

```python
    ...

    # continue of class RAGService
    @bentoml.api
    def ingest_text(self, txt: Annotated[Path, bentoml.validators.ContentType("text/plain")]) -> str:

        with open(txt) as f:
            text = f.read()

        doc = Document(text=text)
        self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=PERSIST_DIR)
        return "Successfully Loaded Document"


    @bentoml.api
    def ingest_pdf(self, pdf: Annotated[Path, bentoml.validators.ContentType("application/pdf")]) -> str:

        # Use pypdf to extract text from pdf
        import pypdf
        reader = pypdf.PdfReader(pdf)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            texts.append(text)
        all_text = "".join(texts)
        doc = Document(text=all_text)
        self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=PERSIST_DIR)
        return "Successfully Loaded Document"
```

#### Query

The `query` endpoint allows users to perform queries against the populated index.

```python
    ...

    @bentoml.api
    def query(self, query: str) -> str:
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
```

## Run the Service

With a `bentofile.yaml` file, you can start the web service by running `bentoml serve .`. Once running, the Service is accessible via http://localhost:3000. You can interact with the API through the Swagger UI or use tools like `curl` to perform requests.

## Next step

The web service works, but it still relies on OpenAI to implement embeddings and answer questions. In [the next tutorial](../02-custom-embedding/), we'll replace OpenAI's embedding function with a custom model for better control over the retrieval process.
# Simple Rag Web Service

In [last section](../00-simple-local-rag/) we made a simple script to generate a documentation index from some text files in a folder. In this section, we want to turn this script into a web service. Now remember that the script in last section has two steps:

1. create an index and populate the index with documents
2. query the index

When converting to a web service, a better way is to separate the first step into two different steps so we have:

1. create an index
2. let users populate the index by uploading files
3. let users query the index

To simplify this process, we will use [BentoML](https://github.com/bentoml/BentoML). In BentoML, a Service is a deployable and scalable unit, defined as a Python class with the `@bentoml.service` decorator. It can manage states and their lifecycles. In our service's `__init__` method, we will create the index and setup some global settings and these codes will be executed a

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

Then we define APIs that let users upload files and populate the index we set up in `__init__`. Each API within the BentoML Service is defined using the `@bentoml.api` decorator and turned into a HTTP endpoint:

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
        return f"Successfully Loaded Document"


    @bentoml.api
    def ingest_pdf(self, pdf: Annotated[Path, bentoml.validators.ContentType("application/pdf")]) -> str:

        # we use pypdf to extract text from pdf
        import pypdf
        ...
```

Finally, we define the API to let users query the index

```python
    ...

    @bentoml.api
    def query(self, query: str) -> str:
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response)
```

With a proper `bentofile.yaml`, we can start the web service by `bentoml serve .`. You can visit <http://localhost:3000>, scroll down to Service APIs, and click Try it out to try out query api. You can also use curl or any HTTP requests library to interact with the API endpoint.


Now we have a web service, but we still rely on OpenAI to do all the work (embedding and question answering). In [next section](../02-custom-embedding/) we will try to replace the embedding function using our own model.

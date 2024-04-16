# Building a Simple RAG System using LlamaIndex

This is the first tutorial of this BentoML RAG example project. In this tutorial, we will build a simple RAG system from LlamaIndex's [starter example](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/).

## Set the OpenAI API key

LlamaIndex uses OpenAI's `gpt-3.5-turbo` model by default. Set your API key as an environment variable by using the following command:

```bash
export OPENAI_API_KEY=XXXXX
```

## Run the script

Inside this directory, we have all the required code in the `starter.py` script.

The script first checks if an index already exists in a specified directory (`storage`); if not, it creates one from documents in another directory `data`, and if it does exist, it simply loads it. In this tutorial, we already have the `data` directory, which contains some files as the knowledge base.

```python
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# Check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # Load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
```

Then we can query the index when asking a question:

```python
query_engine = index.as_query_engine()
response = query_engine.query("What did Paul Graham do growing up?")
print(response)
```

Run the `starter.py` script:

```bash
python starter.py
```

Expected output:

```bash
The author worked on writing short stories and programming, starting with early programming experiences on an IBM 1401 in 9th grade using an early version of Fortran. Later, the author transitioned to working with microcomputers, building a Heathkit kit and eventually getting a TRS-80 to write simple games and programs.
```

## Next step

The script works as expected locally. However, we may want more people to use this script simultaneously. In the [next tutorial](../01-simple-rag/), we will turn this simple local example into a web service that accepts HTTP requests.

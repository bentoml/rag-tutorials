# Simple Rag System using LlamaIndex

We will start with a simple example from LlamaIndex's [starter example](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/). We use the following codes to generate and store the index from documents in `data/` if it doesn't exist in `storage/`, but load it if it does:

```python
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
```

Then we can query the index when we want LLM to answer a question:

```python
query_engine = index.as_query_engine()
response = query_engine.query("What did Paul Graham do growing up?")
print(response)
```

Here we can see there are 2 steps: create an index and populate the index with documents first, then query the index.

To run this example, you need to set your OpenAI key as an environment variable using: `export OPENAI_API_KEY=XXXXX`. Then, run the script by entering `python starter.py` in your terminal. Upon running the script, you will receive an output similar to:

```
The author worked on writing short stories and programming, starting with early programming experiences on an IBM 1401 in 9th grade using an early version of Fortran. Later, the author transitioned to working with microcomputers, building a Heathkit kit and eventually getting a TRS-80 to write simple games and programs.
```

Now we have a simple script that we can run locally. However, we may want more people to use this script simultaneously. In [next example](../01-simple-rag/), we will turn this simple local example into a web service that accepts HTTP requests.

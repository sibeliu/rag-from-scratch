# kiva-test

1. Document ingestion

The problems with document ingestion and chunking have been the object of much work over decades. PyMuPDF is an open-source library with a reasonably long history, well-maintained, with most of the functionality needed for this kind of work. It works better than all other open-source alternatives, in my experience. My suspicion is that Reducto, Unstructured, and any number of other competitors are mostly wrappers around this library. Note PyMuPDF license may require attention if used in a large scale commercial product.

For the purposes of this demo, I use Langchain. But in production I would probably suggest building our own pipeline, unless Langchain improves in the meantime. Here I'm using a semantic chunking algorithm that groups sentences by their similarity. If the documents are legal, it might make sense to use formatting features such as whitespace and section headings to split the text.

2. Embeddings and database

For this demo, I'm using a vanilla SQLite db and a small embedding model that can be used locally with low overhead and cost. If I had more time, I'd implement a graph database and use other methods for search as well as the vector embeddings and hybrid (BM25) implemented here. I'd also optimize the embedding model (but for one document it works fine).

Here, the hybrid_search first calculates the cosine similarity between the query embedding and each chunk's embedding. Then, it performs a keyword search by checking if the query string is present in the chunk's text. Finally, it combines the results from both searches and returns the top top_k results.

3. Using the top-k results to generate an answer to the question

Here we're going to use an expensive LLM to generate a good response. We will make a prompt that includes some of the context from the top retrieved chunk, and asks the question to OpenAI's 4o model. In the output, I'll print the context id number, which would allow us to recover the exact location in the text, as well as the context itself.

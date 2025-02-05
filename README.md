# rag-from-scratch 

## Usage: 
- Clone this repo locally
- Create virtual environment in repo directory, and activate it
- `pip install -r requirements.txt`
- Create a .env file containing `OPENAI_API_KEY="sk-.....your api key"`
- at the command line, type `python question-answering.py /path/to/document "question goes here" `  Note: if document path contains spaces, enclose in quotes eg. `python question-answering.py "ExampleCo - NDA - John Appleseed.pdf" "Is ExampleCo required to mark proprietary documents as confidential?"`

## Brief description of architecture:
This project is a classic implementation of a RAG system which ingests a document, chunks it, saves the chunks and their vector embeddings to a database, and then uses the chunks to provide relevant context when calling an LLM for question answering.

## Literature review
For the purposes of this demo, I refer to the seminal RAG paper, https://arxiv.org/abs/2005.11401. 

For PDF ingestion, I chose PyMuPDF based on the comparison results shown at https://pymupdf.readthedocs.io/en/latest/about.html
For chunking, I use an approach first described here: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb 

In a small implementation like the present, vector search can be done easily by brute force. However, in a larger database it would be necessary to implement approximate search algorithms to find relevant context in a reasonable length of time instead of brute-forcing it. One very recent result that details a way to face the 'curse of dimensionality' can be found here: https://arxiv.org/abs/2410.14452

A naive RAG solution such as the present is not capable of answering detailed questions over a large set of documents. Some evaluation measures are described here: https://arxiv.org/abs/2409.12941 
A more sophisticated solution could involve a graph representation of documents and several other steps in prompting so that global document structure would be saved along with the single chunks. Community detection algorithms, and other graph topology measures, could also be deployed to ensure that the context chosen by the retriever actually serves the purpose. Some documentation of recent 'graphRAG' systems can be found here: https://arxiv.org/abs/2410.20724 An off-the-shelf graphRAG application is here: https://github.com/FalkorDB/GraphRAG-SDK

One innovative design that might produce good results, especially on documents that contain images, tables, and graphs, is called ColPali, and is based on embedding an image of the entire page, instead of extracting text. See https://arxiv.org/abs/2407.01449

## Weaknesses in my design
This implementation is a vanilla RAG system. It suffers from several limitations:
- no representation of global document context, which could be achieved by coupling chunks with a short summary of the overall document they come from. It could be even more improved by using a graph model of the data, and recover subgraphs instead of chunks
- no consideration of using several chunks in the generation of a good answer to the user (currently using top_k = 1)
- embedding model is small and probably quite approximate
- no prompt optimization has been done
- it was written from scratch in 3 hours :)
- the database is not optimized for vector search, and would fail at scale
- this solution does not implement any ontologies, which are likely to improve result quality significantly

## Safeguards for production
In production, this implementation would require several layers of safeguards, including:
- prompt optimization by using a 'golden dataset' and observability tools to fine-tune the output to customer expectations. 
- many other parameters should be tuned, starting with temperature (here I chose .2 for low-hallucination) and number of top-k context chunks
- depending on the setting, it could be important to implement PII detection and masking at least in the user-interface. SpaCy and Microsoft's Presidio offer some usability in a privacy-sensitive setting; other proprietary tools could also be used

## Detailed architectural choices
### 1. Document ingestion

The problems with document ingestion and chunking have been the object of much work over decades. PyMuPDF is an open-source library with a reasonably long history, well-maintained, with most of the functionality needed for this kind of work. It works better than all other open-source alternatives, in my experience. My suspicion is that Reducto, Unstructured, and any number of other competitors are mostly wrappers around this library. Note PyMuPDF license may require attention if used in a large scale commercial product.

For the purposes of this demo, I use Langchain. But in production I would probably suggest building our own pipeline, unless Langchain improves in the meantime. Here I'm using a semantic chunking algorithm that groups sentences by their similarity. If the documents are legal, it might make sense to use formatting features such as whitespace and section headings to split the text.

### 2. Embeddings and database

For this demo, I'm using a vanilla SQLite db and a small embedding model that can be used locally with low overhead and cost. If I had more time, I'd implement a graph database and use other methods for search as well as the vector embeddings and hybrid (BM25) implemented here. I'd also optimize the embedding model (but for one document it works fine).

Here, the hybrid_search first calculates the cosine similarity between the query embedding and each chunk's embedding. Then, it performs a keyword search by checking if the query string is present in the chunk's text. Finally, it combines the results from both searches and returns the top top_k results.

### 3. Using the top-k results to generate an answer to the question

Here we're going to use an expensive LLM to generate a good response. We will make a prompt that includes some of the context from the top retrieved chunk, and asks the question to OpenAI's 4o model. In the output, I'll print the context id number, which would allow us to recover the exact location in the text, as well as the context itself.

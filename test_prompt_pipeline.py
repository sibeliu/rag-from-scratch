import os
import sys
import re
import json
from os.path import join, dirname
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableSequence

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Read the question from the command line
if len(sys.argv) != 2:
    print("Usage: python script.py '<question>'")
    sys.exit(1)

question = sys.argv[1]

context = """Charles Babbage complemented this theory with the principle of labour calculation 
(known since then as the ‘Babbage principle’) to indicate that the division of labour also allows the precise computation of labour costs. Part I of this book can be considered an exegesis 
of Babbage’s two principles of labour analysis and their influence on the common history of political economy, automated computation, and machine intelligence. Although it may sound 
anachronistic, Marx’s theory of automation and relative surplus-value extraction share common postulates with the first projects of machine intelligence.
Marx overturned the industrialist perspective – ‘the eye of the master’ – that was inherent in Babbage’s principles. In Capital, he argued that the social relations of production 
(the division of labour within the wage system) drive the development of the means of production (tooling machines, steam engines, etc.) and not the other way around, as 
techno- deterministic readings have been claiming then and now by centring the Industrial Revolution around technological innovation only. Of these principles of labour analysis Marx 
made also something else: he consid- ered the cooperation of labour not only as a principle to explain the design of machines but also to define the political centrality of what he 
called the Gesamtarbeiter, the general worker. The figure of the general worker was a way of acknowledging the machinic dimension of living labour and confronting the ‘vast automaton’ 
of the industrial factory on the same scale of complexity. Eventually, it was also a necessary figure to ground, on a more solid politics, 
the ambivalent idea of the general intel- lect that Ricardian socialists such as <NAME> and <NAME> pursued, as seen in chapter <NUMBER>.
"""
# Get the model name and API key from environment variables
model_name = os.getenv("HUGGINGFACE_MODEL", "")
api_key = os.getenv("HUGGINGFACE_API_KEY", "")

if not model_name:
    print("Please set the environment variable HUGGINGFACE_MODEL.")
    sys.exit(1)

if not api_key:
    print("Please set the environment variable HUGGINGFACE_API_KEY.")
    sys.exit(1)

# Initialize the Hugging Face Endpoint client
try:
    llm = HuggingFaceEndpoint(endpoint_url=f"https://api-inference.huggingface.co/models/{model_name}", model_kwargs={"api_key": api_key})
except Exception as e:
    print(f"Failed to initialize HuggingFaceEndpoint: {str(e)}")
    sys.exit(1)

# Define a few prompt templates
template = """Question: {question}\nContext: {context}\nAnswer:"""
prompt_template = PromptTemplate.from_template(template)

classification = """Question: {question}\nCategories: {categories}\n Classify this question into one of the given categories. Respond with exactly one word as your answer."""
classification_template = PromptTemplate.from_template(classification)

# Initialize the RunnableSequence
class CustomRunnableSequence(Runnable):
    def __init__(self, steps):
        self.steps = steps

    def invoke(self, inputs):
        output = inputs
        for step in self.steps:
            output = step.invoke(output)
        return output

classification_chain = CustomRunnableSequence([
    RunnablePassthrough(),
    classification_template,
    llm
])

rag_chain = CustomRunnableSequence([
    RunnablePassthrough(),  # To pass through the inputs
    prompt_template,
    llm
])

def post_process_response(response):
    """
    Post-process the model's response to ensure it ends with a complete sentence.
    @parameter response: The raw response from the model.
    @returns The processed response ending with a complete sentence.
    """
    if isinstance(response, dict) and 'text' in response:
        response = response['text']
    
    # Regular expression to match the end of a sentence
    sentence_endings = re.compile(r'([.!?])')

    # Find all the positions of sentence endings
    matches = list(sentence_endings.finditer(response))

    if not matches:
        return response  # No sentence endings found, return the response as is

    # Find the last complete sentence
    last_ending = matches[-1].end()

    # Return the response up to the last sentence ending
    return response[:last_ending]

def classify_question(question, categories):
    """use the LLM to determine which of an existing set of categories best fits the question.
    @parameter question: The user-generated query about the knowledge base
    @parameter categories: A list of strings containing possible categories 
    @returns matching_categories, the list of categories the question could be classified in as determined by the LLM """
    try:
        # Perform inference using langchain
        result = classification_chain.invoke({"question": question, "categories": categories})
        results = result.split() #get answer broken into single words. QUite often the LLM gives more than one-word answers.

    except Exception as e:
        print(f"An error occurred during the API call or processing: {e}")
        sys.exit(1)

    print (categories, results)
    matching_categories = []
    for c in categories: 
        if c.lower() in [r.lower() for r in results]: #make sure we aren't fooled by capitalization
            matching_categories.append(c)
    return matching_categories if matching_categories else None


def main():
    categories = ["Economics", "Anime", "Medicine"]  #define a group of categories for the knowledge database in the backend
    print(f"Input question: {question}")
    category = classify_question(question, categories)
    print(f"This question is categorized as: {category}")
    print(f"...retrieving context related to {category}") #implement this filter in the rag db, return top-k context with tag category

    print(f"Answering question using context: {context}")

    try:
        # Perform inference using langchain
        result = rag_chain.invoke({"question": question, "context": context})

        # Ensure result is in a string format
        if isinstance(result, list):
            result = result[0]

        # Post-process the response to ensure it ends with a complete sentence
        processed_response = post_process_response(result)
        
        # Print the final processed response
        pretty_response = json.dumps(processed_response, indent=4)
        print(f"Response: {pretty_response}")

    except Exception as e:
        print(f"An error occurred during the API call or processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


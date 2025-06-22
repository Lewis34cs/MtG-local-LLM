from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_database import retriever

model = OllamaLLM(model="llama3.2", temperature=0.1)

template = """
You are an expert in Magic: The Gathering (MtG) and you are helping a player with their deck.
You will be given a question about MtG and you will answer it using the information in the
retriever. If you do not know the answer, say "I don't know". 

You will be given the following information from the MtG database: {rules}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    question = input("Ask a question about MtG (q to quit): ")
    if question.lower() == 'q':
        print("Exiting the program.")
        break
    print("Running model...")
    # Use the retriever to get relevant information
    rules = retriever.invoke(question)
    print("Generating response...")
    # Run the chain with the question and retrieved rules
    response = chain.invoke({"question": question, 
                             "rules": rules})

    print(f"Response: {response}")
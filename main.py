from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv


load_dotenv()

llm = ChatMistralAI(
    model="mistral-small-2506"
)

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG system is ready")
print("Press 0 to exit")

while True:
    try:
        query = input("You ask a question: ").strip()
        
        if query == "0":
            print("Exiting...")
            break
        
        if not query:
            print("Please enter a valid question.\n")
            continue

        # Retrieve documents
        docs = retriever.invoke(query)
        
        if not docs:
            print("No documents found in the knowledge base for your query.\n")
            continue
        
        print(f"Found {len(docs)} relevant documents.")
        
        context = "\n".join([doc.page_content for doc in docs])

        # Create and invoke prompt
        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        # Get response
        response = llm.invoke(final_prompt)
        print(f"AI answer: {response.content}\n")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
        break
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please try again.\n")
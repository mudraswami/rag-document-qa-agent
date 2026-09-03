import os
import gradio as gr

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# NVIDIA API Key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY is not set.")


# Embedding model
embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    api_key=NVIDIA_API_KEY
)


# LLM
llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=NVIDIA_API_KEY,
    temperature=0.2
)


# Prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful document question-answering assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)


def answer_question(document, question):

    if not document.strip():
        return "Please provide a document."

    if not question.strip():
        return "Please enter a question."

    # 1. Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.create_documents([document])

    # 2. Create vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # 3. Retrieve relevant chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    documents = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    # 4. Generate answer
    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(formatted_prompt)

    return response.content


# Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(
            label="Document",
            lines=15,
            placeholder="Paste your document here..."
        ),
        gr.Textbox(
            label="Question",
            placeholder="Ask a question about the document..."
        )
    ],
    outputs=gr.Textbox(
        label="Answer",
        lines=8
    ),
    title="RAG Document Q&A Agent",
    description=(
        "Ask questions about a document using "
        "Retrieval-Augmented Generation (RAG)."
    )
)


if __name__ == "__main__":
    demo.launch()

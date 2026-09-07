import os
import gradio as gr

from pypdf import PdfReader
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


def answer_question(pdf_file, question):

    try:
        if pdf_file is None:
            return "Please upload a PDF first.", ""

        if not question.strip():
            return "Please enter a question.", ""

        reader = PdfReader(pdf_file)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():
            return "Could not extract text from this PDF.", ""

        # Split document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.create_documents([text])

        if not chunks:
            return "No usable content was found in the PDF.", ""

        # Create vector store
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Retrieve relevant chunks
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 2}
        )

        documents = retriever.invoke(question)

        if not documents:
            return "I could not find relevant information in the document.", ""

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        # Generate answer
        formatted_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = llm.invoke(formatted_prompt)

        return response.content, context

    except Exception as e:
        return f"Something went wrong: {str(e)}", ""


# Gradio interface
with gr.Blocks() as demo:

    gr.Markdown("# RAG Document Q&A Agent")
    gr.Markdown(
        "Upload a PDF and ask questions about its content."
    )

    with gr.Row():

        with gr.Column():

            pdf_file = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )

            question = gr.Textbox(
                label="Ask a question",
                placeholder="Enter your question here..."
            )

            submit_btn = gr.Button(
                "Submit",
                variant="primary"
            )

            clear_btn = gr.ClearButton()

        with gr.Column():

            answer = gr.Textbox(
                label="Answer",
                lines=6
            )

            context = gr.Textbox(
                label="Retrieved Context",
                lines=10
            )

    submit_btn.click(
        fn=answer_question,
        inputs=[pdf_file, question],
        outputs=[answer, context]
    )

    clear_btn.add(
        [pdf_file, question, answer, context]
    )


demo.launch()

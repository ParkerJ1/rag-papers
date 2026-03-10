import logging
import gradio as gr
from pathlib import Path

from ingest import ingest_folder, ingest_pdf
from query import query 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def handle_query(question: str) -> tuple[str, str]:
    if not question.strip():
        return "Please enter a question", ""
    
    result = query(question)
    answer = result["answer"]
    sources = "\n".join(result["source"])
    return answer, sources

def handle_ingest(files) -> str:
    if not files:
        return "No files uploaded."
    
    count = 0
    for file in files:
        path = Path(file.name)
        ingest_pdf(path)
        count += 1

    return f"{count} files ingested."

with gr.Blocks() as app:
    gr.Markdown("# Research Paper RAG")
    gr.Markdown("Query your local library of research papers using LFM2.")

    with gr.Tab("Query"):
        question_input = gr.Textbox(
            label="Question",
            placeholder="Enter your question here...",
            lines=2
        )
        submit_btn = gr.Button("Submit", variant="primary")
        answer_output = gr.Textbox(
            label="Answer",
            lines=8,
            interactive=False
        )
        sources_output = gr.Textbox(
            label="Sources",
            lines=4,
            interactive=False
        )
        submit_btn.click(
            fn=handle_query,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )

    with gr.Tab("Ingest"):
        file_input = gr.File(
            label="Upload PDFs",
            file_types=["pdf"],
            file_count="multiple"
        )
        ingest_btn = gr.Button("Ingest", variant="primary")
        ingest_output = gr.Textbox(
            label=" Status",
            interactive=False
        )
        ingest_btn.click(
            fn=handle_ingest,
            inputs=file_input,
            outputs=ingest_output
        )

if __name__ == "__main__":
    app.launch()
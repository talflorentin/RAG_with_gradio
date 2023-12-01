import subprocess

subprocess.run(["pip", "install", "--upgrade", "transformers[torch,sentencepiece]==4.34.1"])

from sentence_transformers import SentenceTransformer
import cohere
from os import getenv
import numpy as np

import logging
from pathlib import Path
from time import perf_counter

import gradio as gr
from jinja2 import Environment, FileSystemLoader

from backend.query_llm import generate_hf, generate_openai
from backend.semantic_search import table

from constants import (VECTOR_COLUMN_NAME, TEXT_COLUMN_NAME)
from sentence_transformers import CrossEncoder


cohere_embedding_dimensions = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "embed-english-v2.0": 4096,
    "embed-english-light-v2.0": 1024,
    "embed-multilingual-v2.0": 768,
}
EMB_MODEL_NAME = "embed-english-v3.0"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
# EMB_MODEL_NAME = "all-mpnet-base-v2"


proj_dir = Path(__file__).parent
# Setting up the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the template environment with the templates directory
env = Environment(loader=FileSystemLoader(proj_dir / 'templates'))

# Load the templates directly from the environment
template = env.get_template('template.j2')
template_html = env.get_template('template_html.j2')

# Examples
examples = ['What is MusicGen? Explain its architecture',
            'How to use the trainer module?',
            'What is wav2vec']


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def bot(history, api_kind):
    top_k_rank = 10
    query = history[-1][0]

    if not query:
         gr.Warning("Please submit a non-empty string as a prompt")
         raise ValueError("Empty string was submitted")

    logger.warning('Retrieving documents...')
    # Retrieve documents relevant to query
    document_start = perf_counter()

    if EMB_MODEL_NAME in ["paraphrase-albert-small-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
        retriever = SentenceTransformer(EMB_MODEL_NAME)
        query_vec = retriever.encode(query)

    elif EMB_MODEL_NAME in list(cohere_embedding_dimensions.keys()):
        co = cohere.Client(getenv('COHERE_API_KEY'))
        query_vec = np.array(co.embed([query], input_type="search_document", model=EMB_MODEL_NAME).embeddings[0])
    else:
        query_vec = None

    documents = table.search(query_vec, vector_column_name=VECTOR_COLUMN_NAME).limit(top_k_rank).to_list()
    documents = [doc[TEXT_COLUMN_NAME] for doc in documents]

    if top_k_rank > 4:
        all_docs = documents
        model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512)
        zipped_list = [(query, paragraph) for paragraph in all_docs]
        scores = model.predict(zipped_list)

        top_indices = np.argsort(scores)[:4]
        documents = np.array(all_docs)[top_indices].tolist()


    document_time = perf_counter() - document_start
    logger.warning(f'Finished Retrieving documents in {round(document_time, 2)} seconds...')

    # Create Prompt
    prompt = template.render(documents=documents, query=query)
    prompt_html = template_html.render(documents=documents, query=query)

    if api_kind == "HuggingFace":
         generate_fn = generate_hf
    elif api_kind == "OpenAI":
         generate_fn = generate_openai
    elif api_kind is None:
         gr.Warning("API name was not provided")
         raise ValueError("API name was not provided")
    else:
         gr.Warning(f"API {api_kind} is not supported")
         raise ValueError(f"API {api_kind} is not supported")

    history[-1][1] = ""
    for character in generate_fn(prompt, history[:-1]):
        history[-1][1] = character
        yield history, prompt_html


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            avatar_images=('https://aui.atlassian.com/aui/8.8/docs/images/avatar-person.svg',
                           'https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg'),
            bubble_full_width=False,
            show_copy_button=True,
            show_share_button=True,
            )

    with gr.Row():
        txt = gr.Textbox(
                scale=3,
                show_label=False,
                placeholder="Enter text and press enter",
                container=False,
                )
        txt_btn = gr.Button(value="Submit text", scale=1)

    api_kind = gr.Radio(choices=["HuggingFace", "OpenAI"], value="HuggingFace")

    prompt_html = gr.HTML()
    # Turn off interactivity while generating if you click
    txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, [chatbot, api_kind], [chatbot, prompt_html])

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    # Turn off interactivity while generating if you hit enter
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
            bot, [chatbot, api_kind], [chatbot, prompt_html])

    # Turn it back on
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

    # Examples
    gr.Examples(examples, txt)

demo.queue()
demo.launch(debug=True)

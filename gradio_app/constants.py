# TODO: Make sure to rerun lancedb after changing embedding model!
EMB_MODEL_NAME = "paraphrase-albert-small-v2"  # Winner of MTEB
# EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight
# EMB_MODEL_NAME = "thenlper/gte-large"  # heavy, more powerful

DB_TABLE_NAME = "split_files_db"
VECTOR_COLUMN_NAME = "vctr"
TEXT_COLUMN_NAME = "txt"
FILES_DUMP_FOLDER = "split_files_dump"

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
# HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"
# HF_MODEL = "meta-llama/Llama-2-7b-hf"
HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"

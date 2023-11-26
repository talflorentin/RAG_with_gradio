import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


directory_path = r'C:\Users\Tal\PycharmProjects\rag-gradio-sample-project\prep_scripts\docs_dump'

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

files_splits = {}

for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
    # if filename == 'main_classes_trainer.txt':

        # Construct the full path of the text file
        file_path = os.path.join(directory_path, filename)

        with open(file_path, 'r') as file:
            file_contents = file.read()
            # print(f"Contents of {filename}:\n{file_contents}\n")

            texts = text_splitter.create_documents([file_contents])

            files_splits[filename[:-4]] = texts

output_path = r'C:\Users\Tal\PycharmProjects\rag-gradio-sample-project\prep_scripts\split_files_dump'
os.makedirs(output_path, exist_ok=True)

for key, string_list in files_splits.items():
    for index, doc in enumerate(string_list):
        file_path = f"{output_path}\\{key}_{index}.txt"
        with open(file_path, 'w') as file:
            file.write(doc.page_content)
print('Finished')
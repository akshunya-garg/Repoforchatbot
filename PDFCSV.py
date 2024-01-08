import openai
import os
# edited
import streamlit as st
import tempfile
import time
# import pandasai

from pdf_csv.pdf import pdf_logic, get_docs_with_scores
from pdf_csv.csv import csv_logic


def file_status():

    # status for files
    if st.session_state.pre_file_type != file_type:

        status = ""
        if file_type == "application/pdf":
            status = "Ask queries for uploaded PDF files, you can add multiple files with same extension."
        elif file_type == "text/csv":
            status = "Ask queries for uploaded CSV files, you can add multiple files with same extension."

        st.session_state.messages.append({"role": "status", "content": status})
        st.session_state.pre_file_type = file_type


st.set_page_config(page_title="Ask to PDF & CSV")

css = """
    <style>
        .title {
            position: fixed;
            top: 45px;
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            height: 70px;
            z-index: 90;
            background-color: #0E1117;
        }
        /* Adjust styles for dark mode */
        @media (prefers-color-scheme: light) {
            .title {
            position: fixed;
            top: 45px;
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            height: 70px;
            z-index: 90;
            background-color: #ffff;
        }
        }
        .stAlert{
            position: fixed;
            top: 20%;
            background-color: #352F44;
            z-index: 999999;
            width: 46%;
            border-radius: 16px;
        }
        .stAlert div{
            background-color: #352F44;
            padding: 10px;
            font-size: 3rem;
        }
    </style>
"""
st.markdown(css, unsafe_allow_html=True)


def clear_files():
    st.session_state["file_uploader_key"] += 1
    st.session_state.up_files = []
    # st.experimental_rerun()


def create_temp_file(uploaded_file):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    # Write the uploaded content to the temporary file
    temp_file.write(uploaded_file.read())
    temp_file.close()

    return temp_file.name


# openai.api_key = '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'
# os.environ['OPENAI_API_KEY'] = openai.api_key

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "up_files" not in st.session_state:
    st.session_state.up_files = []

st.sidebar.header('Upload files')
uploaded_files = st.sidebar.file_uploader(
    "Upload files with similar extension", accept_multiple_files=True, key=st.session_state["file_uploader_key"],)


st.sidebar.button("Remove uploaded files",
                  use_container_width=True, on_click=clear_files)

# Chat bot
st.markdown("<h1 class='title'>Chat Bot for PDF and CSV</h1>",
            unsafe_allow_html=True)
# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# for previous uploded file
if "pre_file_type" not in st.session_state:
    st.session_state.pre_file_type = None

if "pdf_qa" not in st.session_state:
    st.session_state.pdf_qa = None
if "csv_qa" not in st.session_state:
    st.session_state.csv_qa = None
if "error_file" not in st.session_state:
    st.session_state.error_file = ""
# clear_history


def clear_history():
    st.session_state.messages = []
    st.session_state.pre_file_type = None


# display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# user input
chat_history = []
if st.session_state.up_files == uploaded_files and uploaded_files and (st.session_state.pdf_qa or st.session_state.csv_qa):
    if prompt := st.chat_input("Type..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        # add msg to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Generating the response..."):
                if st.session_state.pdf_qa:
                    docs_and_scores = get_docs_with_scores(prompt)
                    # print(docs_and_scores[0][1])
                    print(docs_and_scores)
                    if docs_and_scores[0][1] > 0.55:
                        response = "No Context Available On This Subject"
                    else:
                        result = st.session_state.pdf_qa(
                            {"question": prompt, "chat_history": chat_history})
                        chat_history.extend([(prompt, result["answer"])])
                        response = result["answer"]

                if st.session_state.csv_qa:
                    dataframes, pandas_ai = st.session_state.csv_qa
                    prompt = f"Given the question: Question: '''{prompt}''' . Strictly, Provide a Python code snippet to extract specific data from a CSV file using the pandas library. If the question is unrelated to data extraction from CSV files or if thee information provided is insufficient, the response should be 'I don't know'. Rigidly in case of any error or out of context question response should be 'I don't know'."
                    response = pandas_ai(dataframes, prompt)
                st.markdown(response)

        # add msg to history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


if uploaded_files != st.session_state.up_files and uploaded_files:
    overlay = st.warning('⭕ Please wait for the process to complete')
    # make session state and variable equal to run chat bot UI
    st.session_state.up_files = uploaded_files
    file_type = st.session_state.up_files[0].type
    type_check_flag = False
    other_type_files = False
    file_name = ""

    for file in st.session_state.up_files:
        if file.type != 'application/pdf' and file.type != "text/csv":
            other_type_files = True
            file_name = file.name
            break
    for file in st.session_state.up_files:
        if file.type != file_type:
            type_check_flag = True
            file_name = file.name
            break
    if other_type_files:
        st.error(f'Upload supported files. Only PDF or CSV files are supported. "{file_name}" is Invalid.')
    elif type_check_flag:
        st.error(f'Please upload the same file type. "{file_name}" is not same as rest of the files.')
    else:
        try:
            if file_type == 'application/pdf':
                st.session_state.pdf_qa = pdf_logic(st.session_state.up_files)
            if file_type == "text/csv":
                st.session_state.csv_qa = csv_logic(st.session_state.up_files)
                print(type(st.session_state.csv_qa))
                print(st.session_state.csv_qa)
            file_status()
            st.experimental_rerun()
        except Exception as e:
            if str(e) == 'EmptyFileError':
                st.error(
                    f'Unable to process the uploaded files. "{st.session_state.error_file}" might be empty. Please check and try again.')
            else:
                st.error(
                    f'Unable to Process the Uploaded files. Please check "{st.session_state.error_file}" and try again with valid files.')

            st.session_state.error_file = ""
        overlay.empty()


if len(st.session_state.messages) > 1:
    st.button("Clear History", on_click=clear_history)

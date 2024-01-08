import tempfile
import openai
import pandas as pd
# import pandasai
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import pandas as pd
from llm.llm import get_llm
# from pandasai.llm import LangchainLLM
import csv
from reportlab.pdfgen import canvas

def csv_to_text(csv_path, text_path):
    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file, open(text_path, mode='w', encoding='utf-8') as text_file:
            csv_reader = csv.reader(csv_file)
            
            for row in csv_reader:
                # Convert each row to a string and write it to the text file
                for i in row:
                   print(i)
                   text_file.write(i + '\n')

        print(f"Text file '{text_path}' has been created.")

    except Exception as e:
        print(f"Error: {e}")

def csv_to_pdf(csv_path, pdf_path):
    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Create a PDF file
            pdf_canvas = canvas.Canvas(pdf_path)

            # Set font and size for the PDF
            pdf_canvas.setFont("Helvetica", 10)

            # Set the initial y-coordinate for writing text in the PDF
            y_coordinate = 750

            for row in csv_reader:
                # Convert each row to a string and write it to the PDF
                pdf_canvas.drawString(50, y_coordinate, '\t'.join(row))
                y_coordinate -= 12  # Adjust the y-coordinate for the next line

            pdf_canvas.save()

        print(f"PDF file '{pdf_path}' has been created.")

    except Exception as e:
        print(f"Error: {e}")

# Replace 'your_file.csv', 'output.txt', and 'output.pdf' with the actual paths
# csv_to_text('output.csv', 'output.txt')
# # csv_to_pdf('output.csv', 'output.pdf')

def check_data_type(csv_path):
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)

        # Check the data types of each column
        data_types = df.dtypes

        # Iterate through the columns and determine if they contain textual or numerical data
        t=0
        for column, dtype in data_types.items():
            if dtype == 'object':
                print(f"{column} contains textual data.")
            else:
                t=1
                print(f"{column} contains numerical data.")
        if t==1:
            return "Numerical"
        else :
            return "Textual"

    except Exception as e:
        return f"Error: {e}"

# Replace 'your_file.csv' with the path to your CSV file
# a=check_data_type('input.csv')
# print(a)
def create_temp_file(uploaded_file):
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    # Write the uploaded content to the temporary file
    temp_file.write(uploaded_file.read())
    temp_file.close()

    return temp_file.name
 
def csv_logic(uploaded_files):
    dataframes = [pd.read_csv(create_temp_file(file)) for file in uploaded_files]
    API_KEY="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"
    llm = OpenAI(api_token=f"{API_KEY}")
    # OpenAI()

    pandas_ai = PandasAI(llm, verbose=True, conversational=True, enforce_privacy=True, enable_cache=False)
    
    return [dataframes, pandas_ai]

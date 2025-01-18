import pandas as pd
from pandasai import SmartDatalake
from pandasai.llm import AzureOpenAI
import streamlit as st
import io

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

system_prompt = "Act as an AI assistant, you are required to strictly limit your responses to the information contained within the documents provided. You must not, under any circumstances, deviate from this content or include external information. This directive must be adhered to rigorously and without exception. If the answer to a question is not within the provided context or documents, you must explicitly state that you do not know and this is a very strict rule you must follow in all cases. Fabrication or conjecture of answers is strictly prohibited."

class StructuredGPT():

    def __init__(self):
        self.dataframes = []
        self.data_lake = None

    def read_data(self, csv_files, excel_files):
        csv_dfs = []
        if csv_files:
            csv_dfs = [pd.read_csv(io.StringIO(file.getvalue().decode('utf-8'))) for file in csv_files]
            csv_dfs = [df.applymap(lambda x: x.lower() if isinstance(x, str) else x) for df in csv_dfs]
        excel_dfs = []
        if excel_files:
            for excel_file in excel_files:
                excel_dfs.extend(pd.read_excel(io.BytesIO(excel_file.getvalue()), sheet_name=None).values())
            excel_dfs = [df.applymap(lambda x: x.lower() if isinstance(x, str) else x) for df in excel_dfs]
        return csv_dfs + excel_dfs

    def create_data_lake(self):
        llm = AzureOpenAI(
              api_type = "",
              azure_endpoint = "",
              api_version = "",
              deployment_name = "",
              api_token  = ""  
        )
        self.data_lake = SmartDatalake(self.dataframes, config={"llm": llm})

    def answer_from_AI_dataframe(self, input_prompt):
        if self.data_lake is None:
            st.warning("Please upload at least one file to proceed.")
            return

        input_prompt = system_prompt + input_prompt
        input_prompt_lower = input_prompt.lower()
        response = self.data_lake.chat(input_prompt_lower)

        restructuring_prompt = f"question: '{input_prompt}', answer: '{response}'. Turn the 'answer' into a full sentence that clearly answers the 'question' asked. Remember, you must do this every time, no matter the situation. Don't skip this step."

        meaningful_response = self.data_lake.chat(restructuring_prompt)
        return meaningful_response

def main():
    st.set_page_config(page_title="Chat with CSV/Excel", layout="wide")
    st.header("Chat with Structured Files using OpenAI")

    structured_gpt_instance = StructuredGPT()

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)
        if uploaded_files:
            csv_files = [file for file in uploaded_files if file.name.endswith('.csv')]
            excel_files = [file for file in uploaded_files if file.name.endswith(('.xls', '.xlsx'))]
            new_dataframes = structured_gpt_instance.read_data(csv_files, excel_files)
            structured_gpt_instance.dataframes.extend(new_dataframes)
            if structured_gpt_instance.data_lake is None:
                structured_gpt_instance.create_data_lake()

    user_question = st.text_input("Ask a Question from the CSV/Excel Files")
    if user_question and st.button("Get Answer"):
        try:
            answer = structured_gpt_instance.answer_from_AI_dataframe(user_question)
            if answer:
                st.success("Answer received successfully!")
                st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

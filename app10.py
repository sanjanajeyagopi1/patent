import fitz  # PyMuPDF for PDF extraction
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import re  # For parsing structured output
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Set up Azure OpenAI API credentials from .env
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # Pull from environment
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  # Pull from environment
    api_version=os.getenv("OPENAI_API_VERSION"),  # Pull from environment
)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""

    # Extract text from each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text("text")

    doc.close()
    return full_text

# Function to check for conflicts using OpenAI API
def check_for_conflicts(action_document_text, referenced_documents_text):
    prompt = f"""
    Analyze the following action document text and referenced documents to extract foundational claims:

    Action Document:
    {action_document_text}

    Referenced Documents:
    {referenced_documents_text}

    Step 1: Extract the key claims from the action document and name them as 'Key_claims'.
    Step 2: From the 'Key_claims', extract the foundational claim and store it in a variable called "foundational_claim".
    Step 3: Extract all referenced documents under U.S.C. 102 and/or 103 mentioned in the action document, specified only in the "foundational_claim".
    Step 4: For each referenced document, create a variable that stores the document name.
    Step 5: If the foundational claim refers to the referenced documents, extract the entire technical content with its specified paragraph location and image reference. Map the claim with the conflicting document name.
    Step 6: Return the output as:
        FOUNDATIONAL CLAIM:
        DOCUMENTS REFERENCED:
        FIG:
        TEXT:
    """

    messages = [
        {
            "role": "system",
            "content": "You are a patent attorney analyzing the document for foundational claims and conflicts."
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = client.chat.completions.create(
        model="GPT-4-Omni", messages=messages, temperature=0.6
    )

    output = response.choices[0].message.content
    return output

# Function to extract and analyze figure-related details from the `check_for_conflicts` function output
def extract_figures_and_text(conflict_results):
    fig_section = re.search(r"FIG:(.*?)(?=TEXT:)", conflict_results, re.DOTALL)
    text_section = re.search(r"TEXT:(.*)", conflict_results, re.DOTALL)

    fig_details = fig_section.group(1).strip() if fig_section else "No figure details found."
    text_details = text_section.group(1).strip() if text_section else "No technical text found."

    # Prepare a structured prompt for figure analysis
    figure_analysis_prompt = f"""
    Analyze the following figures and technical text. For each figure, extract:
    1. The figure number and title.
    2. The technical details related to the figure, as referenced in the text, do not miss any technical details.
    3. Explain the figure's importance in relation to the foundational claim.
    4. Extract the text from the paragraphs mentioned in the foundational claim and store it in separate variable.

    Figures:
    {fig_details}

    Text:
    {text_details}
    """

    messages = [
        {
            "role": "system",
            "content": "You are a technical expert analyzing figures in a document."
        },
        {
            "role": "user",
            "content": figure_analysis_prompt,
        },
    ]

    # Call OpenAI API for figure analysis
    response = client.chat.completions.create(
        model="GPT-4-Omni", messages=messages, temperature=0.6
    )

    analysis_output = response.choices[0].message.content
    return analysis_output

# Streamlit app interface
st.title("Patent Conflict Analyzer")

# Initialize session state for managing uploaded documents and analysis
if 'referenced_documents' not in st.session_state:
    st.session_state.referenced_documents = []

if 'conflict_results' not in st.session_state:
    st.session_state.conflict_results = []

# Upload main action PDF file
uploaded_action_file = st.file_uploader("Upload Main Action Document", type="pdf")
if uploaded_action_file:
    # Extract text from the main action PDF
    action_file_path = "temp_action.pdf"
    with open(action_file_path, "wb") as f:
        f.write(uploaded_action_file.read())
    extracted_action_text = extract_text_from_pdf(action_file_path)

    # Upload referenced documents
    uploaded_ref_files = st.file_uploader("Upload Referenced Documents", type="pdf", accept_multiple_files=True, key="referenced")

    # When referenced documents are uploaded
    if uploaded_ref_files:
        referenced_documents_text = ""
        for uploaded_file in uploaded_ref_files:
            # Save each referenced document temporarily
            temp_path = f"temp_referenced_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Extract text from the referenced document
            extracted_ref_text = extract_text_from_pdf(temp_path)
            referenced_documents_text += f"{uploaded_file.name}:\n{extracted_ref_text}\n\n"

            # Add the uploaded file to the session state
            st.session_state.referenced_documents.append({
                'file_name': uploaded_file.name,
                'text': extracted_ref_text
            })

        # Perform conflict analysis when the button is clicked
        if st.button("Check for Conflicts"):
            conflict_results_raw = check_for_conflicts(extracted_action_text, referenced_documents_text)
            st.session_state.conflict_results.append(conflict_results_raw)

            # Display foundational claim
            foundational_claim = re.search(r"FOUNDATIONAL CLAIM:(.*?)(?=DOCUMENTS REFERENCED:)", conflict_results_raw, re.DOTALL)
            if foundational_claim:
                st.subheader("Foundational Claim")
                st.text(foundational_claim.group(1).strip())

            # Button to analyze figures after conflict results are obtained
            if st.button("Analyze Figures"):
                ref_fig_results = extract_figures_and_text(conflict_results_raw)
                st.subheader("Figure Analysis")
                st.text_area("Analysis Result", value=ref_fig_results, height=300)

# Display all analysis results saved in session state
if st.session_state.conflict_results:
    st.subheader("Stored Conflict Analysis Results")
    for i, result in enumerate(st.session_state.conflict_results):
        st.text_area(f"Analysis Result {i + 1}", value=result, height=300)

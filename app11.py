import fitz  # PyMuPDF for PDF extraction
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import re  # For parsing structured output
import streamlit as st
from io import BytesIO
import docx

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
    try:  
        doc = fitz.open(pdf_path)  
        full_text = ""  
  
        # Extract text from each page  
        for page_num in range(len(doc)):  
            page = doc.load_page(page_num)  
            full_text += page.get_text("text")  
  
        doc.close()  
        return full_text  
    except Exception as e:  
        st.error(f"Error extracting text from PDF: {e}")  
        return ""  
  
# Function to check for conflicts using OpenAI API  
def check_for_conflicts(action_document_text):  
    """  
    Analyzes the action document and extracts:  
    - Foundational claim  
    - Referenced documents  
    - Figures and technical text related to them  
    """  
    prompt = f"""  
    Analyze the following action document text and extract the foundational claim:  
  
    {action_document_text}  
  
    Step 1: Extract the key claims from the document and name it as 'Key_claims'.  
    Step 2: From the 'Key_claims' extract the foundational claim and store it in a variable called "foundational_claim" (Note: method claims and system claims are not considered independent claims and only one claim can be the foundational claim).  
    Step 3: From the foundational claim, extract the information under U.S.C 102 and/or 103.  
    Step 4: Extract all referenced documents under U.S.C. 102 and/or 103 mentioned in the action document specified only in the "foundational_claim".  
    Step 5: For each referenced document, create a variable that stores the document name.  
    Step 6: If the foundational claim refers to the referenced documents, extract the entire technical content with its specified paragraph location and image reference. Map the claim with the conflicting document name.  
    Step 7: Do not extract any referenced document data that is not related to the foundational claim.  
    Step 8: Return the output as:  
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
  
    # Call OpenAI API for conflict checking  
    try:  
        response = client.chat.completions.create(
        model="GPT-4-Omni", messages=messages, temperature=0.6
    )
        output = response.choices[0].message.content  
        return output  
    except Exception as e:  
        st.error(f"Error during conflict checking: {e}")  
        return None  
  
# Function to extract and analyze figure-related details  
def extract_figures_and_text(conflict_results, ref_document_text):  
    """  
    Extract figures and related technical text from the 'check_for_conflicts' function's output.  
    """  
    # Extract the 'FIG' and 'TEXT' sections using regex or another parsing method  
    fig_section = re.search(r"FIG:(.*?)(?=TEXT:)", conflict_results, re.DOTALL)  
    text_section = re.search(r"TEXT:(.*)", conflict_results, re.DOTALL)  
  
    fig_details = fig_section.group(1).strip() if fig_section else "No figures referenced."  
    text_details = text_section.group(1).strip() if text_section else "No technical text found."  
  
    # Prepare a structured prompt for figure analysis  
    figure_analysis_prompt = f"""  
    Analyze the figures and technical text from the referenced document in relation to the foundational claim.  
  
    Instructions:  
  
    1. Identify Figures:  
       - For each figure referenced in the foundational claim, extract the following:  
         - **Figure Number and Title:** Provide the figure number and its title.  
         - **Technical Details:** Extract all technical details related to the figure as mentioned in the text. Ensure no technical detail is missed.  
         - **Importance:** Explain the importance of the figure in relation to the foundational claim. Describe how it supports, illustrates, or contradicts the claim.  
  
    2. Extract Text from Paragraphs:  
       - From the paragraphs cited in the foundational claim, extract the relevant text as in the document uploaded and store it in a separate variable.  
  
    3. Workflow for Cases with Images:  
       - If figures are present in the referenced document:  
         - Follow the steps outlined above to extract figure details and technical information.  
         - Ensure that any interpretations of the figures include specific references to the data or concepts depicted.  
  
    4. Workflow for Cases without Images:  
       - If no figures are present:  
         - Focus on extracting and analyzing the text from the referenced document.  
         - Identify and highlight key technical details and concepts that are essential to understanding the foundational claim.  
  
    Input Details:  
  
    Figures:  
    {fig_details}  
  
    Text:  
    {text_details}  
  
    Referenced Document Text:  
    {ref_document_text}  
  
    Example Output Format:  
  
    - Figures Analysis:  
      - **Figure Number and Title:** [Insert here]  
      - **Technical Details:** [Insert here]  
      - **Importance:** [Insert here]  
  
    - Extracted Text from Paragraphs:  
      - [Store text in separate variable]  
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
    try:  
        response = client.chat.completions.create(
        model="GPT-4-Omni", messages=messages, temperature=0.6
        )
        analysis_output = response.choices[0].message.content  
        return analysis_output  
    except Exception as e:  
        st.error(f"Error during figure analysis: {e}")  
        return None  
  
# Function to analyze the filed application  
def analyze_filed_application(filed_application_text, foundational_claim, figure_analysis):  
    """  
    Analyze the filed application to determine if the examiner's rejection is justified.  
    """  
    prompt = f"""  
    Using the foundational claim:  

{foundational_claim}  

and the figure analysis results:  

{figure_analysis}  

analyze the filed application text:  

{filed_application_text}  

Determine if the examiner is correct in rejecting the application under either U.S.C 102 or U.S.C 103.  

Instructions:  
- Cite instances from the application as filed to justify your stance.  
- Provide a clear justification with cited text from the filed application.  
- Make relevant comparisons between the application as filed and the cited text.  
- Give a detailed report.  
- Suggest amendments to the application based on the examiner's reasons for rejection based on the foundational claim.  
- Identify any potential limitations in the current claims and propose specific language or structural changes that could address those limitations.  
- Ensure that the amendments maintain the original intent of the claims while enhancing their clarity and breadth.  
"""
  
    messages = [  
        {  
            "role": "system",  
            "content": (  
                "Adopt the persona of a Person Having Ordinary Skill in the Art (PHOSITA). "  
                "Analyze the filed application text and determine if the examiner is correct "  
                "in rejecting the application under either U.S.C 102 or U.S.C 103. "  
                "Cite instances from the application as filed to justify your stance."  
            )  
        },  
        {  
            "role": "user",  
            "content": prompt,  
        },  
    ]  
  
    # Call OpenAI API for application analysis  
    try:  
        response = client.chat.completions.create(
        model="GPT-4-Omni", messages=messages, temperature=0.6
        ) 
        analysis_output = response.choices[0].message.content  
        return analysis_output  
    except Exception as e:  
        st.error(f"Error during filed application analysis: {e}")  
        return None  
  
def save_analysis_to_word(filed_application_analysis):  
    # Check if the analysis is valid and not None  
    if filed_application_analysis is None or filed_application_analysis.strip() == "":  
        st.error("Analysis data is missing or empty.")  
        return None  
  
    # Create a new Document  
    doc = docx.Document()  
  
    # Add a title to the document  
    doc.add_heading('Filed Application Analysis Results', level=1)  
  
    # Add analysis content  
    doc.add_paragraph(filed_application_analysis)  
  
    # Save the document to a BytesIO object  
    buffer = BytesIO()  
    doc.save(buffer)  
    buffer.seek(0)  
  
    return buffer  
  
# Streamlit app interface  
st.title("Patent Conflict Analyzer")  
  
# Initialize session state for managing uploaded documents and analysis  
if 'conflict_results' not in st.session_state:  
    st.session_state.conflict_results = None  
if 'foundational_claim' not in st.session_state:  
    st.session_state.foundational_claim = None  
if 'figure_analysis' not in st.session_state:  
    st.session_state.figure_analysis = None  
if 'filed_application_analysis' not in st.session_state:  
    st.session_state.filed_application_analysis = None  
  
# Step 1: Upload examiner document  
uploaded_examiner_file = st.file_uploader("Upload Examiner Document", type="pdf", key="examiner")  
if uploaded_examiner_file is not None:  
    # Extract text from the examiner PDF  
    with open("temp_examiner.pdf", "wb") as f:  
        f.write(uploaded_examiner_file.read())  
  
    extracted_examiner_text = extract_text_from_pdf("temp_examiner.pdf")  
  
    # Clean up temporary file  
    os.remove("temp_examiner.pdf")  
  
    # Check for conflicts button  
    if st.button("Check for Conflicts"):  
        conflict_results_raw = check_for_conflicts(extracted_examiner_text)  
        if conflict_results_raw:  
            st.session_state.conflict_results = conflict_results_raw  
            
            # Show foundational claim  
            foundational_claim_match = re.search(  
                r"FOUNDATIONAL CLAIM:(.*?)(?=DOCUMENTS REFERENCED:)",  
                conflict_results_raw,  
                re.DOTALL  
            )  
            documents_referenced_match = re.search(  
                r"DOCUMENTS REFERENCED:(.*?)(?=FIG:)",  
                conflict_results_raw,  
                re.DOTALL  
            )  

            if documents_referenced_match:  
                documents_referenced = documents_referenced_match.group(1).strip()  
                st.write("### Documents Referenced:")  
                st.write(documents_referenced)  # Display documents referenced
            else:  
                st.warning("No referenced documents found in the examiner document.")  
        else:  
            st.error("Failed to check for conflicts.")  


  
# Step 2: After examiner document is uploaded and analyzed  
if st.session_state.get("conflict_results") is not None:  
    uploaded_ref_file = st.file_uploader("Upload Referenced Document", type="pdf", key="referenced")  
    if uploaded_ref_file is not None:  
        # Extract text from the referenced PDF  
        with open("temp_referenced.pdf", "wb") as f:  
            f.write(uploaded_ref_file.read())  
  
        extracted_ref_text = extract_text_from_pdf("temp_referenced.pdf")  
  
        # Clean up temporary file  
        os.remove("temp_referenced.pdf")  
  
        if st.button("Analyze Figures and Cited Text"):  
            figure_analysis_results = extract_figures_and_text(  
                st.session_state.conflict_results,  
                extracted_ref_text  
            )  
            if figure_analysis_results:  
                st.session_state.figure_analysis = figure_analysis_results  
                st.success("Figure analysis completed successfully!")  
            else:  
                st.error("Failed to analyze figures and cited text.")  
  
# Step 3: After referenced document is uploaded and analyzed  
if st.session_state.get("figure_analysis") is not None:  
    uploaded_filed_app = st.file_uploader("Upload Application as Filed", type="pdf", key="filed")  
    if uploaded_filed_app is not None:  
        # Extract text from the filed application  
        with open("temp_filed.pdf", "wb") as f:  
            f.write(uploaded_filed_app.read())  
  
        extracted_filed_app_text = extract_text_from_pdf("temp_filed.pdf")  
  
        # Clean up temporary file  
        os.remove("temp_filed.pdf")  
  
        if st.button("Analyze Filed Application"):  
            filed_app_analysis_results = analyze_filed_application(  
                extracted_filed_app_text,  
                st.session_state.foundational_claim,  
                st.session_state.figure_analysis  
            )  
            if filed_app_analysis_results:  
                st.session_state.filed_application_analysis = filed_app_analysis_results  
                st.success("Filed application analysis completed successfully!")  
                st.text_area(  
                    "Filed Application Analysis Results",  
                    value=filed_app_analysis_results,  
                    height=300  
                )  
            else:  
                st.error("Failed to analyze the filed application.")  
  
# Option to download results  
if st.session_state.get("filed_application_analysis"):  
    # Create a downloadable .docx file only if the analysis is available  
    docx_buffer = save_analysis_to_word(st.session_state.filed_application_analysis)  
  
    if docx_buffer:  
        # Add download button for the Word file  
        st.download_button(  
            label="Download Analysis Results",  
            data=docx_buffer,  
            file_name="filed_application_analysis.docx",  
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"  
        )  

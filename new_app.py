import streamlit as st
import PyPDF2
from rag_pipeline import generate_answer

##### Streamlit APP
def double_sidebar_space():
    st.sidebar.write("")
    st.sidebar.write("")

def double_space():
    st.write("")
    st.write("")


# Set the app's title and layout
st.set_page_config(page_title="SmartRecruitAI", page_icon="🧠", layout="wide")


# Sidebar for API key and resume uploads
st.sidebar.header("Configuration")

# API Key Input
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password", help="To use the app use your OpenAI API Key")
double_sidebar_space()
# File Uploader
uploaded_files=st.sidebar.file_uploader("📄 Upload Resumes (PDF format)", type="pdf", accept_multiple_files=True, help="Drag and drop files here or click to select multiple PDF resumes.")



# Header Section with title and subtitle
st.title("🧠 SmartRecruitAI",)
#st.subheader("Empowering Recruiters with AI-Powered Resume Analysis")

# Usage Description
double_space()
st.caption("How to use SmartRecruitAI")
st.markdown("""
Welcome to **SmartRecruitAI**! This free app helps HR professionals and recruiters to analyze candidate resumes with ease.


Simply provide a **job description**, 

upload **candidate resumes** (on the left), 

enter your **questions**, and let AI assist you in finding the best candidate for the job.""")


st.divider()  # Using st.divider()
# Create two columns
col1, col2 = st.columns(2)

# Job Description in the first column
with col1:
    st.header("Job Description")
    job_description = st.text_area("📝 Enter the job description", height=200, help="Provide a detailed job description to help the AI analyze resumes effectively.")

# Questions in the second column
with col2:
    st.header("Candidate Questions")
    query = st.text_area("❓ Enter your questions about the candidates", height=200, help="Ask specific questions you have about the candidates that the AI should answer based on their resumes.")

# Analyze Resumes button
st.write("")
if st.button("🚀 Run SmartRecruitAI"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    elif not query:
        st.error("Please enter your question(s) about the candidates.")
    else:
        st.info("Analyzing resumes... This may take a few moments.")
        with st.spinner("Processing..."):
            documents = ""
            for idx, uploaded_file in enumerate(uploaded_files, start=1):
                reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents += f"Candidate #{idx}:\n{text}\n\n"
            print("*"*20)
            print(documents)
            print("*"*20)
            
            try:
                answer = generate_answer(query, openai_api_key, documents, job_description)
                st.success("Analysis complete! See the results below.")
                st.write("### AI Response")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer with useful information
st.markdown("---")
st.write("")
st.markdown("""    
<sup>**AI Resume Analyst** is a cutting-edge tool designed to make the recruitment process easier and more efficient. 
Whether you're screening dozens of resumes or trying to match candidates to specific roles, our AI-driven solution provides you with the insights you need.
            
<sup>If you have any questions or need support, please [contact us](mailto:ivanamati@gmail.com).</sup>""", unsafe_allow_html=True)
st.markdown("---")
st.caption("Thank you for using SmartRecruitAI")
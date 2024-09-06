from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


def retriever_RAG(open_api_key, document):
    """this function loads the document, creates chunks, embed each chunk 
    and load it into the vector store and make a retriever"""
    #raw_documents = TextLoader(document).load()

    # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # docs = text_splitter.split_text(text)
    # return docs
    # Split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 200, length_function = len)
    documents = text_splitter.split_text(document)
    # Pass the documents and embeddings to create FAISS vector index
    vectorindex_openai = FAISS.from_texts(documents, OpenAIEmbeddings(api_key=open_api_key))
    # Save the vectorstore object locally
    vectorindex_openai.save_local("vectorindex_openai")
    # Load the vectorstore object
    vectorstore = FAISS.load_local("vectorindex_openai", OpenAIEmbeddings(api_key=open_api_key),allow_dangerous_deserialization=True)
    # Retrieve the information from the vectorestore
    retriever = vectorstore.as_retriever(k=2)
    return retriever


def generate_answer(query, open_api_key, document, job_description):

    llm = ChatOpenAI(temperature=0,
    model="gpt-4o",
    openai_api_key=open_api_key)

    retriever = retriever_RAG(open_api_key, document)
    data = retriever.invoke(query)

    job = retriever_RAG(open_api_key, job_description)
    job_description = job.invoke(query)
    print("this is job description retriever and invoke:", job_description)
    #job_description = retriever_job.invoke(query)

    print(data)
        
    PROMPT_Message = """
    Your task is to help the recruiter to get to find the best candidate for the position described in job and the learn candidates better. 

    **Query Validation:**
    Before answering any query, you must determine if the question is relevant to the task of evaluating candidates for the job described. 
    - The query must be directly related to the candidate's skills, qualifications, experience, education, or how well they match the job description.
    - If the query is about something outside the scope of evaluating a candidate (e.g., asking for irrelevant information or malicious requests), 
    you must respond with: "This question is outside the scope of candidate evaluation or job description analysis. Please ask a relevant question."
    - Do not respond to queries asking for confidential, administrative, or irrelevant information unrelated to the candidate/job.
    - If the query is valid, proceed with answering the question based on the context provided.

    Be sure to know to distinct the candidates.

    Answer the recruiter questions based on the provided Context. Don't put yourself in front, answer based on the Context.

    **Your task when answering valid queries:**
    You will be provided with the following:
    1. **Context**, which includes the candidate's resume.
    2. **Job Description**, which provides details about the job.

    Use this information to answer the recruiter's questions. Follow this structure:
    - **Context**: Summarize the job position briefly.
    - **Candidate Analysis**: Analyze each candidate based on the context and their relevance to the job position.
    - **Conclusion**: Write which candidate is a better fit for the position and why.
    - **Recommendation**: Suggest why the chosen candidate is better and if additional confirmation is needed. 

    When enough informations provided give longer and informative answers. Write it in a structured form that is easier to read. 
    If there are not information in the Context, recommend to contact the job candidate through his/hers email.

    Be helpful and stick to the input information which are Context (candidates Resumes) and job (job description)

    If question asked about specific candidate, use your own reasoning and structure and answer just about this specific candidate. 

    <Context>
    {context}
    </Context> 
    
    <job>
    {job_description}
    </job> 

    """

    # SYSTEM_TEMPLATE = """
    # You are IvyBot, an AI assistant dedicated to assisting Ivana in her job search 
    # by providing recruiters with relevant and concise information and making her a good and valuable candidate for the company. 

    # Your tasks are following:
    # 1. Answer provide informatona about Ivana only.
    # 2. When asked to provide information about projects count at least 4 of them.
    # 3. When asked about education count both - linguistical and developing. 
    # 4. When asked about skills count developing, scholar and personal.
    # 5. If you do not know the answer, politely admit it and let recruiters know how to contact Ivana to get more information directly from her. 

    # Don't put "IvyBot" or a breakline in the front of your answer. Don't make informations up!
    # When you are asked about IvyBot, provide explanation that you are made using RAG approach and GPT-4o to aks questions about Ivana and demonstrating her coding skills. 

    # To answer the recruiters questions about Ivana use ONLY the following informations: 
    # <informations>
    # {context}
    # </informations> 

    # """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                PROMPT_Message,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    answer = document_chain.invoke(
        {
            "context": data,
            "messages": [
                HumanMessage(content=query)
            ],
            "job_description":job_description
        }
    )

    print(answer)
    return answer


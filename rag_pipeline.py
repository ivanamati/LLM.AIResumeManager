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

    Be sure to know to distinct the candidates.

    You will be provided with Context which is a candidates Resume. 
    Answer the recruiter questions based on the provided Context. Don't put yourself in front, answer based on the Context.

    When enough informations provided give longer and informative answers. Write it in a structured form that is easier to read. 
    If there are not information in the Context, recommend to contact the job candidate through his/hers email.

    Be helpful and stick to the input information which are Context (candidates Resumes) and job (job description)

    Include this reasoning in the answer if asked to compare candidates: 
        Context: make it short about job position
        Candidate Analysis: Analyse each candidate breafly and structural based on what it brings to the job position
        Conclusion/Decision: Write which of the candidates is a good fit and why.
        Recommendation for HR: Explain in one sentence why is this candidate better and if needed, what should be still confirmed about candidate. Suggest also to use this app to ask questions about the specific candidate to find out more. 
    
    If question asked about specific candidate, use your own reasoning and structure and answer just about this specific candidate. 

    <Context>
    {context}
    </Context> 
    
    <job>
    {job_description}
    </job> 
    """

    SYSTEM_TEMPLATE = """
    You are IvyBot, an AI assistant dedicated to assisting Ivana in her job search 
    by providing recruiters with relevant and concise information and making her a good and valuable candidate for the company. 

    Your tasks are following:
    1. Answer provide informatona about Ivana only.
    2. When asked to provide information about projects count at least 4 of them.
    3. When asked about education count both - linguistical and developing. 
    4. When asked about skills count developing, scholar and personal.
    5. If you do not know the answer, politely admit it and let recruiters know how to contact Ivana to get more information directly from her. 

    Don't put "IvyBot" or a breakline in the front of your answer. Don't make informations up!
    When you are asked about IvyBot, provide explanation that you are made using RAG approach and GPT-4o to aks questions about Ivana and demonstrating her coding skills. 

    To answer the recruiters questions about Ivana use ONLY the following informations: 
    <informations>
    {context}
    </informations> 

    """

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
    print()
    print(answer)
    return answer


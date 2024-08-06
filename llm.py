
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# langchain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# langchain-ibm
from langchain_ibm import WatsonxLLM

# langchain-community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_llm(credentials, project_id):
  params = {
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.MIN_NEW_TOKENS: 0,
    GenParams.DECODING_METHOD: "greedy",
    GenParams.REPETITION_PENALTY: 1
  }

  # LangChainで使うllm
  llm = WatsonxLLM(
    model_id = 'meta-llama/llama-3-70b-instruct',
    apikey = credentials['apikey'],
    url = credentials['url'],
    params = params,
    project_id = project_id
  )
  
  return llm

def create_embedding(llm, file_path, creds, project_id):
  loader = PyPDFLoader(file_path)

  print("===========================")
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0);
  
  index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(),
    text_splitter = text_splitter 
    ).from_loaders([loader])

  # Initialize watsonx google/flan-ul2 model
  params = {
      GenParams.DECODING_METHOD: "sample",
      GenParams.TEMPERATURE: 0.2,
      GenParams.TOP_P: 1,
      GenParams.TOP_K: 100,
      GenParams.MIN_NEW_TOKENS: 50,
      GenParams.MAX_NEW_TOKENS: 300
  }
  
  retriever = index.vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={'score_threshold': 0.5}
  )
  
  # Init RAG chain
  chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    return_source_documents=True,
    input_key="question"
  )

  return chain 
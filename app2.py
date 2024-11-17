from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi import HTTPException
from typing import List
from pydantic import BaseModel
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import logging

app = FastAPI()
app.state.vectors = None
MAX_FILE_SIZE_MB = 15
pdf_directory = "./data"

load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# groq_api_key = os.getenv("GROQ_API_KEY")
api_keys = {}
class APIKeyRequest(BaseModel):
    google_api_key: str
    groq_api_key: str

@app.post("/set_api_keys/")
async def set_api_keys(request: APIKeyRequest):
    global api_keys
    
    try:
        api_keys = {
            "google_api_key": request.google_api_key,
            "groq_api_key": request.groq_api_key
        }

        try:
            genai.configure(api_key=api_keys["google_api_key"])
        except Exception as e:
            logging.error(f"Error configuring Google API: {str(e)}")
            raise HTTPException(status_code=500, detail="Error configuring Google API.")
        try:
            global llm  # global variable to maintain the LLM instance
            llm = ChatGroq(groq_api_key=api_keys["groq_api_key"], model_name="Llama3-8b-8192")

            return {"message": "API keys have been set successfully."}
        
        except Exception as e:
            logging.error(f"Error configuring Groq API: {str(e)}")
            raise HTTPException(status_code=500, detail="Error configuring Groq API.")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error occurred in configuring API's.")

# llm = ChatGroq(groq_api_key= groq_api_key, model_name="Llama3-8b-8192")

prompt_template=ChatPromptTemplate.from_template(
"""
    Answer the question from the provided context, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    context:\n {context}?\n
    questions: \n{input}\n
"""
)

def vector_embedding(pdf_directory=pdf_directory):
    try:
        if not os.path.exists(pdf_directory) or len(os.listdir(pdf_directory)) == 0:
            raise ValueError("No files available. Please upload PDFs first.")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception as e:
            raise RuntimeError(f"Error initializing embeddings model: {str(e)}")
        try:
            loader = PyPDFDirectoryLoader(pdf_directory)
            docs = loader.load()
        except Exception as e:
            raise RuntimeError(f"Error loading PDF documents: {str(e)}")
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs[:50])
            vectors = FAISS.from_documents(final_documents, embeddings)
        except Exception as e:
            raise RuntimeError(f"Error creating vector store: {str(e)}")
        
        return vectors

    except ValueError as ve:
        raise ve
    except RuntimeError as re:
        raise re
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during vector embedding: {str(e)}\n\nIf error persists try deleting and reuploading all files.")

class QueryRequest(BaseModel):
    question: str

# Endpoint 1 to add PDF files
@app.post("/files/")
async def upload_files(files: List[UploadFile] = File(...)):
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)

    uploaded_file_names = []
    error_files = []

    for file in files:
        try:
            contents = await file.read()
            file_size = len(contents)

            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"File too large. Try with files less than {MAX_FILE_SIZE_MB} MB")
        
            file_path = os.path.join(pdf_directory, file.filename)
            
            # if os.path.exists(file_path):
            #     raise HTTPException(status_code=409, detail=f"File '{file.filename}' already exists.")

            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            
            uploaded_file_names.append(file.filename)

        except Exception as e:
            error_files.append({"file_name": file.filename, "error": str(e)})

    return {"uploaded_files": uploaded_file_names, "errors": error_files}

# Endpoint 2 to ask questions
@app.post("/query/")
async def ask_question(query: QueryRequest):
    api_res = {}
    # Check if vectors are available, if not generate them
    if app.state.vectors is None:
        try:
            app.state.vectors = vector_embedding()
            api_res["message"] = "Response generated from previously uploaded files."
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error while generating vectors: {str(e)}")
    
    try:
        # Create a retriever from the vector store
        retriever = app.state.vectors.as_retriever()
        
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": query.question})

        api_res["answer"] = response['answer']
        api_res["context"] = response["context"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    return api_res

# Endpoint 3 to refresh vector embeddings
@app.post("/refresh_vectors/")
async def refresh_vectors():
    try:
        app.state.vectors = vector_embedding()
        return {"message": "Vector embeddings have been refreshed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while refreshing vectors: {str(e)}")

#Endpoint 4 to list uploaded files
@app.get("/list_files/")
async def list_files():
    # Check if directory exists
    if not os.path.exists(pdf_directory):
        return {"files": []}

    try:
        # List all files in the directory
        files = os.listdir(pdf_directory)
        return {"files": files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


class DeleteFilesRequest(BaseModel):
    files: List[str]
# Endpoint 5 to delete files
@app.delete("/files/")
async def delete_files(request: DeleteFilesRequest):
    deleted_files = []
    error_files = []

    for file_name in request.files:
        try:
            file_path = os.path.join(pdf_directory, file_name)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(file_name)
            else:
                raise FileNotFoundError(f"File '{file_name}' not found")

        except Exception as e:
            # Capture the error and add it to the error list
            error_files.append({"file_name": file_name, "error": str(e)})

    return {"files_deleted": deleted_files, "errors": error_files}

class GetFilesRequest(BaseModel):
    files: List[str]
# Endpoint 6 Retrieve files
@app.get("/files/")
async def get_files(request: GetFilesRequest):
    files_to_return = []
    error_files = []

    for file_name in request.files:
        file_path = os.path.join(pdf_directory, file_name)
        
        try:
            # Check if the file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File '{file_name}' not found")

            files_to_return.append(FileResponse(path=file_path, filename=file_name))
        
        except Exception as e:
            # Capture the error and add it to the error list
            error_files.append({"file_name": file_name, "error": str(e)})

    return {"files": files_to_return, "errors": error_files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

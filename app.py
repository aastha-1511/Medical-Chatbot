from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def ask(query):
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt_text = f"""
                You are a Medical assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If you don't know the answer, say that you don't know.
                Use three sentences maximum and keep the answer concise.

                Context:
                {context}

                Question:
                {query}
                """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt_text}
        ]
    )

    return response.choices[0].message.content


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    response = ask(msg)
    print("Response : ", response)
    return str(response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
from langchain_core.messages import BaseMessage
from typing import Annotated, Dict, TypedDict
from langgraph.graph import END, StateGraph
import pprint
import os
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import Dict

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI

from utils import store_to_db
import json


if "GOOGLE_API_KEY" not in os.environ and "TAVILY_API_KEY" not in os.environ:
    print("No GOOGLE/TAVILY API keys stored in the environment. Fetching those from '.env' file...")
    load_dotenv()

url = "https://docs.pydantic.dev/latest/api/networks/"
db_retriever = store_to_db(url)

with open("prompt_template.json") as file:
    prompt_templates = json.load(file)


def retrieve(state: dict) -> dict:
    documents = db_retriever.get_relevant_documents(state["keys"]["question"])

    return {
        "keys": {
            "question": state["keys"]["question"],
            "documents": documents
        }
    }


def grade_documents(state: dict) -> dict:
    grader_llm = ChatGoogleGenerativeAI(model="gemini-pro")

    prompt = PromptTemplate(
        template=prompt_templates["prompt"]["grader"],
        input_variables=["question", "content"]
    )

    chain = prompt | grader_llm | JsonOutputParser()

    filtered_docs = []
    search = False

    for doc in state["keys"]["documents"]:
        score = chain.invoke(
            {
                "question": state["keys"]["question"],
                "content": doc.page_content
            }
        )

        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(doc)
        else:
            search = True
            continue

    return {
        "keys": {
            "question": state["keys"]["question"],
            "documents": filtered_docs,
            "run_web_search": search
        }
    }


def decide_to_generate(state: dict) -> str:
    return "transform_query" if state["keys"]["run_web_search"] else "generate"


def transform_query(state: dict) -> dict:
    prompt = PromptTemplate(
        template=prompt_templates["prompt"]["transform_query"],
        input_variables=["question"],
    )

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    chain = prompt | llm | StrOutputParser()

    better_question = chain.invoke(
        {
            "question": state["keys"]["question"]
        }
    )

    return {
        "keys": {
            "question": better_question,
            "documents": state["keys"]["documents"]
        }
    }


def web_search(state: dict) -> dict:
    tool = TavilySearchResults(k=3)
    print(f'''Search string: {state["keys"]["question"]}''')
    docs = tool.invoke(
        {
            "query": state["keys"]["question"]
        }
    )
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    state["keys"]["documents"].append(web_results)

    return {
        "keys": {
            "question": state["keys"]["question"],
            "documents": state["keys"]["documents"]
        }
    }


def generate(state: dict) -> dict:
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()

    generation = rag_chain.invoke(
        {
            "context": format_docs(state["keys"]["documents"]),
            "question": state["keys"]["question"]
        }
    )

    return {
        "keys": {
            "documents": state["keys"]["documents"],
            "question": state["keys"]["question"],
            "generation": generation
        }
    }

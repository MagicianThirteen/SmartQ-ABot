from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
from typing import List,Optional
from langsmith import traceable,client
load_dotenv()

#配置好langSmith
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"]="true"
    os.environ.setdefault("LANGSMITH_PROJECT","SmartQABot")
    print(f"LangSmith配置完成-Project:{os.getenv('LANGSMITH_PROJECT')}")
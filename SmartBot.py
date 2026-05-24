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

#定义要输出的结构 Schema
#全用英文，更省token
class QAResponse(BaseModel):
    answer:str=Field(description="The answer to the user's question")
    confidence:str=Field(description="Confidence level:high,medium,or low")
    reasoning:str=Field(description="The reasoning behind the answer provided.")
    follow_up_questions:List[str]=Field(
        description="A list of follow-up questions related to the topic.",
        default_factory=list,
    )
    sources_needed:bool =Field(
        description="Indicates whether sources are needed for the answer.",
        default=False,
    )

#建个类来调用方法
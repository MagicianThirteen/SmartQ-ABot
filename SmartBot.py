from langchain_core.prompts import ChatPromptTemplate
from langchain_core import runnables
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import os
from typing import List,Optional
from langsmith import traceable,Client
load_dotenv()
client = Client()

#配置好langSmith
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"]="true"
    #os.environ["LANGCHAIN_TRACING_V2"]="true"
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
class SmartQABot:
    def __init__(self,
                model_name:str="gpt-4o-mini",
                temperature:float=0,):
        #定义模型
        self.model=ChatOpenAI(
            model=model_name,
            temperature=temperature,
        ).with_structured_output(QAResponse)
        #定义prompt
        self.prompt=ChatPromptTemplate.from_messages(
            [
                ("system",
                 """你是个剑来的热心读者.
                    # Your guidelines:
                    #         - Answer questions accurately and concisely
                    #         - Be honest about uncertainty - set confidence to 'low' if unsure
                    #         - Provide clear reasoning for your answers
                    #         - Suggest relevant follow-up questions
                    #         - Indicate if external sources would help

                    #         Always respond with accurate, helpful information."""
                 
                 ),
                ("human","{question}"),
            ]
        )
        self.chain = self.prompt | self.model
        #print(self.chain)

    #问单个问题
    @traceable(name="ask_question",run_type="chain")
    def ask(self,question:str)->QAResponse:
        try:
            #为什么这里invoke不起作用,因为前面的prompt写成了format_message
            response = self.chain.invoke({"question":question})
            return response
        except Exception as e:
            # 返回错误结构
            return QAResponse(
                answer="I'm sorry,I couldn't process your question at this time.",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=["Could you please try again later?"],
                sources_needed=True,
            )
    
    #问一堆的问题
    @traceable(name="ask_batch",run_type="chain")
    def ask_batch(self,questions:List[str])->List[QAResponse]:
        inputs=[{"question":q}for q in questions]
        #输入一堆list[dict]
        return self.chain.batch(inputs,return_exceptions=True)
    

    #测试
def demo_qa_bot():
        bot=SmartQABot()
        questions=[
            "烽火戏诸侯创作的剑来的女主角是谁",
            #"烽火戏诸侯创作的剑来的男主角是谁",
        ]
        print("="*60)
        print("测试smart Q&A BOT")
        print("="*60)

        for question in questions:
            print(f"\n Question:{question}")
            print("-"*40)
            response=bot.ask(question)

            print(f"Question: {question}")
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence}")
            print(f"Reasoning: {response.reasoning}")
            print(f"Follow-up Questions: {response.follow_up_questions}")
            print(f"Sources Needed: {response.sources_needed}")
            print("-" * 60)    

if __name__ == "__main__":
    try:
        demo_qa_bot()
    finally:
        #把缓存里的trace发出去
        client.flush()
    



from itertools import product

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

#简单的展示下langchain的核心概念，runnables和lecl语言
#测试输出这个：结果：LangChain 是一个用于构建基于大语言模型（LLM）应用程序的框架，它通过将语言模型与外部数据源和工具集成，简化了开发复杂自然语言处理应用的过程。
#环境，api都可用
def langchain_core_concepts():


    #这里实现一个简单问答，问openai，langchain的一个解释
    prompt=ChatPromptTemplate.from_template("你是个ai技术专家，一句话解释下{question}")
    model=ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
    parser=StrOutputParser()
    chain=prompt|model|parser #这里不能随便改顺序，model会不知道怎么输出
    result=chain.invoke({"question":"什么是langchain？"})
    print(f"结果：{result}")
    return chain

if __name__ == "__main__":
    langchain_core_concepts()

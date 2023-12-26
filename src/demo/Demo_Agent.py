from langchain.agents import Tool, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.tools import tool
from langchain import hub
from langchain.tools.render import render_text_description
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import ShellTool
from langchain.agents.agent_toolkits import FileManagementToolkit
from gradio_tools.tools import (
    ImageCaptioningTool,
    StableDiffusionPromptGeneratorTool,
    StableDiffusionTool,
    TextToVideoTool,
)
from langchain_experimental.utilities import PythonREPL

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.postprocessor import SentenceTransformerRerank

import os
import openai
from datetime import datetime
import gradio as gr
import time

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACEHUB_API_TOKEN"
os.environ["SERPAPI_API_KEY"] = "YOUR_SERPAPI_API_KEY"
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"


def agent_response(message, history):
    response = agent_executor.run(message)
    for i in range(len(response)):
        time.sleep(0.03)
        yield response[: i+1]


if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    rerank = SentenceTransformerRerank(top_n=3, model="BAAI/bge-reranker-base")

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=512,
        chunk_overlap=64
    )

    documents_1 = SimpleDirectoryReader("../data/sample_data/AI Career/data").load_data()
    index_1 = VectorStoreIndex.from_documents(documents_1, service_context=service_context, show_progress=True)
    query_engine_1 = index_1.as_query_engine(
        similarity_top_k=10, node_postprocessors=[rerank]
    )

    documents_2 = SimpleDirectoryReader("../data/sample_data/lectures").load_data()
    index_2 = VectorStoreIndex.from_documents(documents_2, service_context=service_context, show_progress=True)
    query_engine_2 = index_2.as_query_engine(
        similarity_top_k=10, node_postprocessors=[rerank]
    )

    rag_tools = [
    Tool(
        name="LlamaIndex_How_to_build_a_career_in_AI",
        func=lambda q: str(query_engine_1.query(q)),
        description="Useful for when you want to answer questions about how to build a career in AI. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
    Tool(
        name="LlamaIndex_ML_CS229_Andrew_NG",
        func=lambda q: str(query_engine_2.query(q)),
        description="Useful for when you want to answer questions about the lectures in Machine Learning CS229 class that was tought by Andrew NG. The input to this tool should be a complete english sentence.",
        return_direct=True,
    ),
]

    # Shell (bash)
    shell_tool = ShellTool()
    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        "{", "{{"
    ).replace("}", "}}")

    # gradio tools
    gradio_tools = [
        StableDiffusionTool().langchain,
        ImageCaptioningTool().langchain,
        StableDiffusionPromptGeneratorTool().langchain,
        TextToVideoTool().langchain,
    ]

    # Python REPL
    python_repl = PythonREPL()
    python_repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )

    @tool
    def time(text: str) -> str:
        """Returns the current date and time, use this for any \
        questions related to knowing the current date and time. \
        The input should always be an empty string, \
        and this function will always return the current date time, \
        any datetime mathmatics should occur outside this function."""
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    tools = load_tools(["serpapi", "llm-math"], llm=llm) + [shell_tool, python_repl_tool, time]
    tools.extend(gradio_tools)
    tools.extend(rag_tools)

    prompt = hub.pull("hwchase17/react-chat")
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm_with_stop = llm.bind(stop=["\nObservation"])

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent_executor = initialize_agent(
        tools,
        llm,
        # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    gr.ChatInterface(
        agent_response,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask me anything!", container=False, scale=7),
        title="LLMs Agent with RAG and Tools",
        description="Ask me any question",
        theme="soft",
        examples=[
            "Calculate 15% of 1234",
            "Create a .txt file, fill its content with 'Hello World' and save it to current directory",
            "Please create a photo of a dog riding a skateboard",
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    ).launch(share=True, debug=True)
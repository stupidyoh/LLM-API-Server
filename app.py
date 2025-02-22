from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# 환경설정을 .env에 저장해두고 그 설정을 읽어들임(dotenv)
load_dotenv()

# load model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]
# print(model.invoke(messages))

# streaming
# for token in model.stream(messages):
#     print(token.content, end="|")

# prompt templates
system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
# print(prompt)

prompt.to_messages()

response = model.invoke(prompt)
print(response.content)
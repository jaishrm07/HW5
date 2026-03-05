from openai import OpenAI

#from main import * # This is the main code for the robot

with open("main.py", "r") as f:
    code = f.read()
#print(code)


# Modify OpenAI's API key and API base to use the server.
openai_api_key = "sk-YOUR-API-KEY"
openai_api_base = "https://llm-api.arc.vt.edu/api/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)



messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"do you understand the code:{code} . your job is to generate a code in while loop to make the robot open the microwave. you need to generate a python function. i will call this function in the while loop of main.py file do not output any thing else."},
]

#print(messages)
model = "gpt-oss-120b"
response = client.chat.completions.create(model=model, messages=messages)
code_str = response.choices[0].message.content
#exec(code_str)
print(response.choices[0].message.content)

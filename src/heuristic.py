import openai
import tiktoken
import xml.etree.ElementTree as ET
from Levenshtein import ratio

model = "gpt-4"
openai.api_key = "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" 
threshold = 0.4

def load_benchmark(
    path:str
) -> []:
    with open(path, 'r') as file:
        xml_content = file.read()

    topics = []
    root = ET.fromstring(xml_content)
    for topic in root.findall('topic'):
        t = ""
        for element in topic:
            t+= f"<{element.tag}>{element.text}</{element.tag}>\n"
        topics.append(t)
        
    return topics

def splitter(
    point:str
) -> tuple:
    enc = tiktoken.encoding_for_model(model)
    q = enc.encode(point)
    q1 = enc.decode(q[:len(q)//2])
    q2 = enc.decode(q[len(q)//2:])
    return q1,q2

def sim(
   completion:str,
   q2:str     
)-> float:
    return ratio(completion,q2)

def complete(
    q1:str,
    q2:str
) -> str:
    enc = tiktoken.encoding_for_model(model)

    if model=="text-davinci-003":
        text = openai.Completion.create(
            model=model,
            prompt=q1,
            max_tokens=len(enc.encode(q2)),
            temperature=0
        )
        return text['choices'][0]['text']

    else:
        response = openai.ChatCompletion.create(
              model=model,
              messages=[
                  {"role": "system", "content": "You are a chatbot"},
                  {"role": "user", "content": q1},
              ],
              max_tokens=len(enc.encode(q2)),
              temperature=0 # to ensure reproducibility
          )

        result = ''
        for choice in response.choices:
            result += choice.message.content
        return result

if __name__ == "__main__":
    topics = load_benchmark('../benchmark/misinfo-2020-topics.xml')

    matches = 0
    for topic in topics:
        q1, q2 = splitter(topic)
        print(q1)
        print(q2)
        compl = complete(q1,q2)
        print(compl)
        rat = sim(compl,q2)
        print(rat)
        print()
        if rat>threshold:
            matches+=1
    
    print(matches)
    print(matches/len(topics))
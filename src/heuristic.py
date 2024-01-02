from importlib.metadata import files
from typing import List
import openai
import tiktoken
import xml.etree.ElementTree as ET
from Levenshtein import ratio
from bleurt import score
from rouge_metric import PyRouge
import numpy as np
import sys

model = "gpt-3.5-turbo"
openai.api_key = "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"
checkpoint = "BLEURT-20"
scorer = score.BleurtScorer(checkpoint)
rouge = PyRouge()

def load_benchmark(
    year:str,
    fields:tuple
) -> List:
    
    topics = []
    root = ET.parse(f'../benchmark/misinfo-{year}-topics.xml').getroot()
    for topic in root.findall('topic'):
        title = topic.find(fields[0]).text
        question = topic.find(fields[1]).text
        answer = topic.find(fields[2]).text
        disclaimer = topic.find(fields[3]).text
        t = (title, question, answer, disclaimer)
        topics.append(t)

    return topics

# def splitter(
#     point:str
# ) -> tuple:
#     enc = tiktoken.encoding_for_model(model)
#     q = enc.encode(point)
#     q1 = enc.decode(q[:len(q)//2])
#     q2 = enc.decode(q[len(q)//2:])
#     return q1,q2

def sim(
   completion:str,
   q2:str     
)-> float:
    return ratio(q2, completion), scorer.score(references=[q2], candidates=[completion])[0], rouge.evaluate([completion], [[q2]])['rouge-1']['f'] ### for BLEURT I am using the recommended checkpoint and for ROUGE unigram overlap, but I could also compute the longest common sequence or weighted

def complete(
    year:str,
    title:str,
    question:str,
    answer:str,
    fields:tuple    
) -> str:
    guided_prompt = f'You are provided with the {fields[0]}, {fields[1]} and {fields[2]} fields of a topic from the TREC {year} Health Misinformation topic set.'\
    f'Complete the {fields[3]} field of the topic as exactly appeared in the dataset.'\
    f'Only rely on the original form of the topic in the dataset to complete the narrative field.'\
    f'\n{fields[0]}:{title}\n{fields[1]}:{question}\n{fields[2]}:{answer}\n{fields[3]}:' 
    general_prompt = f'Complete the {fields[3]} based on the {fields[0]}, {fields[1]} and {fields[2]} fields such that the {fields[3]} provides an explanation for the {fields[2]} to the given {fields[1]}'\
    f'\n{fields[0]}:{title}\n{fields[1]}:{question}\n{fields[2]}:{answer}\n{fields[3]}:' 
    if model=="text-davinci-003" or model=="text-davinci-002":
        text = openai.Completion.create(
            model=model,
            prompt=guided_prompt,
            # max_tokens=len(enc.encode(q2)),
            temperature=0
        )
        return text['choices'][0]['text']

    else:
        response = openai.ChatCompletion.create(
              model=model,
              messages=[
                  {"role": "system", "content": "You are a chatbot"},
                  {"role": "user", "content": guided_prompt},
              ],
            #   max_tokens=len(enc.encode(q2)),
              temperature=0 # to ensure reproducibility
          )

        result = ''
        for choice in response.choices:
            result += choice.message.content
        return result

if __name__ == "__main__":
    year = sys.argv[1] ### the only param of the script is the year we want to analyse

    if year=="2020":
        fields = ('title', 'description', 'answer', 'narrative')
    elif year=="2021":
        fields = ('query', 'description', 'stance', 'narrative')
    elif year=="2022":
        fields = ('query', 'question', 'answer', 'background')

    topics = load_benchmark("2021", fields)

    lev, bleu, r = [], [], []
    for topic in topics:
        compl = complete("2021", topic[0], topic[1], topic[2], fields)    
        print(compl)
        print("Real value:```", topic[3],"```")
        #### AQUI DUPLICAR LA SIMILARIDAD
        leven,bleurt,rou = sim(compl,topic[3])
        print("Levenshtein:",leven, "BLEURT:",bleurt, "ROUGE:", rou)
        print()
        lev.append(leven)
        bleu.append(bleurt)
        r.append(rou)
        ###### AQUI TESTS STATS
    
    print("ALL Levenshtein:",np.mean(leven), "BLEURT:",np.mean(bleu), "ROUGE:", np.mean(r))
    # if rat>threshold:
    #     matches+=1
    
    # print(matches)
    # print(matches/len(topics))
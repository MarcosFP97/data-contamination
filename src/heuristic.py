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
from scipy.stats import wilcoxon
from transformers import LlamaTokenizer

model = "../models/llama-2-13b-chat.Q8_0.gguf"
openai.api_base = "http://localhost:8000/v1"
openai.api_key = ""
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
    fields:tuple,
    max_tokens:int   
) -> str:
    guided_prompt = f'<s>[INST] <<SYS>>\n<</SYS>>\nYou are provided with the {fields[0]}, {fields[1]} and {fields[2]} fields of a topic from the TREC {year} Health Misinformation topic set. '\
    f'Complete the {fields[3]} field of the topic as exactly appeared in the dataset. '\
    f'Only rely on the original form of the topic in the dataset to complete the narrative field.'\
    f'\n{fields[0]}: {title}\n{fields[1]}: {question}\n{fields[2]}: {answer} \n{fields[3]}: [\INST]' 
    general_prompt = f'<s>[INST] <<SYS>>\n<</SYS>>\nComplete the {fields[3]} based on the {fields[0]}, {fields[1]} and {fields[2]} fields such that the {fields[3]} provides an explanation for the {fields[2]} to the given {fields[1]}.'\
    f'\n{fields[0]}: {title}\n{fields[1]}: {question}\n{fields[2]}: {answer} \n{fields[3]}: [\INST]' 
    
    if model=="text-davinci-003" or model=="text-davinci-002":
        print(general_prompt)
        general_text = openai.Completion.create(
            model=model,
            prompt=general_prompt,
            max_tokens=max_tokens,
            temperature=0
        )
        print(general_text['choices'][0]['text'])
        print("=========================================================")
        print(guided_prompt)
        guided_text = openai.Completion.create(
            model=model,
            prompt=guided_prompt,
            max_tokens=max_tokens,
            temperature=0
        )
        print(guided_text['choices'][0]['text'])

        return general_text['choices'][0]['text'], guided_text['choices'][0]['text']

    else:
        print(general_prompt)
        general_response = openai.ChatCompletion.create(
              model=model,
              messages=[
                  {"role": "system", "content": "You are a chatbot"},
                  {"role": "user", "content": general_prompt},
              ],
              max_tokens=max_tokens,
              temperature=0 # to ensure reproducibility
          )

        general_result = ''
        for choice in general_response.choices:
            general_result += choice.message.content
        print(general_result)
        print("=========================================================")
        print(guided_prompt)
        guided_response = openai.ChatCompletion.create(
              model=model,
              messages=[
                  {"role": "system", "content": "You are a chatbot"},
                  {"role": "user", "content": guided_prompt},
              ],
              max_tokens=max_tokens,
              temperature=0 # to ensure reproducibility
          )

        guided_result = ''
        for choice in guided_response.choices:
            guided_result += choice.message.content
        print(guided_result)

        return general_result, guided_result

if __name__ == "__main__":
    year = sys.argv[1] ### the only param of the script is the year we want to analyse

    if year=="2020":
        fields = ('title', 'description', 'answer', 'narrative')
    elif year=="2021":
        fields = ('query', 'description', 'stance', 'narrative')
    elif year=="2022":
        fields = ('query', 'question', 'answer', 'background')

    topics = load_benchmark(year, fields)

    lev_gen, bleu_gen, r_gen, lev_guid, bleu_guid, r_guid = [], [], [], [], [], []
    count_leven, count_bleu, count_rouge = 0, 0, 0
    for topic in topics:
        real_narrative = topic[3] 

        if "llama" in model:
            tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
            tokens = len(tokenizer.encode(real_narrative))
        else:
            enc = tiktoken.encoding_for_model(model)
            tokens = len(enc.encode(real_narrative))

        general, guided = complete(year, topic[0], topic[1], topic[2], fields, tokens) 
        print(f"Real value:\"{real_narrative}\"")
        #### AQUI DUPLICAR LA SIMILARIDAD
        leven_gen,bleurt_gen,rou_gen = sim(general,topic[3])
        print("GENERAL--- Levenshtein:",leven_gen, "BLEURT:",bleurt_gen, "ROUGE:", rou_gen)
        leven_guid,bleurt_guid,rou_guid = sim(guided,topic[3])
        print("GUIDED---- Levenshtein:",leven_guid, "BLEURT:",bleurt_guid, "ROUGE:", rou_guid)
        print()

        if leven_guid > leven_gen:
            count_leven+=1

        if bleurt_guid > bleurt_gen:
            count_bleu+=1

        if rou_guid > rou_gen:
            count_rouge+=1

        lev_gen.append(leven_gen)
        bleu_gen.append(bleurt_gen)
        r_gen.append(rou_gen)
        lev_guid.append(leven_guid)
        bleu_guid.append(bleurt_guid)
        r_guid.append(rou_guid)
        print()
        print()

    ###### AQUI TESTS STATS!!!! 
    print("ALL GENERAL--- Levenshtein:",np.mean(lev_gen), "BLEURT:",np.mean(bleu_gen), "ROUGE:", np.mean(r_gen))
    print("ALL GUIDED--- Levenshtein:",np.mean(lev_guid), "BLEURT:",np.mean(bleu_guid), "ROUGE:", np.mean(r_guid))
    print(f"{count_leven}, {count_bleu}, {count_rouge}")
    
    if np.mean(lev_guid) > np.mean(lev_gen):
        print("Wilcoxon Levensthein", wilcoxon(lev_guid, lev_gen, alternative='greater'))

    if np.mean(bleu_guid) > np.mean(bleu_gen):
        print("Wilcoxon BLEURT", wilcoxon(bleu_guid, bleu_gen, alternative='greater'))

    if np.mean(r_guid) > np.mean(r_gen):
        print("Wilcoxon ROUGE", wilcoxon(r_guid, r_gen, alternative='greater'))
import re

import pandas as pd


def remove_contractions(sentences):
    for i in range(len(sentences)):
        sentences[i] = (sentences[i].replace('i\'m', 'i am')
                        .replace('you\'re', 'you are')
                        .replace('he\'s', 'he is')
                        .replace('she\'s', 'she is')
                        .replace('it\'s', 'it is')
                        .replace('we\'re', 'we are')
                        .replace('they\'re', 'they are')
                        .replace('don\'t', 'do not')
                        .replace('can\'t', 'can not')
                        .replace('won\'t', 'will not')
                        .replace('wouldn\'t', 'would not')
                        .replace('shouldn\'t', 'should not')
                        .replace('isn\'t', 'is not')
                        .replace('aren\'t', 'are not')
                        .replace('haven\'t', 'have not')
                        .replace('hadn\'t', 'had not')
                        .replace('wasn\'t', 'was not')
                        .replace('weren\'t', 'were not')
                        .replace('i\'ve', 'i have')
                        .replace('you\'ve', 'you have'))

    return sentences


def remove_punctuations(sentences):
    for i in range(len(sentences)):
        sentences[i] = re.sub('[^a-zA-Z0-9\\s]', '', sentences[i])
        sentences[i] = re.sub('\\s+', ' ', sentences[i])

    return sentences


dataset_delim = '+++$+++'

raw_conversation_meta = []
raw_conversation_lines = []

conversation_file = open('./data/raw/movie_conversations.txt', encoding='ISO-8859-1')
lines_file = open('./data/raw/movie_lines.txt', encoding='ISO-8859-1')

for meta in conversation_file.read().split('\n'):
    raw_conversation_meta.append(meta)

for line in lines_file.read().split('\n'):
    raw_conversation_lines.append(line)

conversation_lines = {}
for line in raw_conversation_lines:
    conv_components = line.split(dataset_delim)
    conversation_lines[conv_components[0].strip()] = conv_components[-1].strip()

conversation_meta = []
for meta in raw_conversation_meta:
    meta_components = meta.split(dataset_delim)[-1][2:-1].split(', ')
    meta_tokens = [meta.replace('\'', '') for meta in meta_components]
    conversation_meta.append(meta_tokens)

print('First set of conversation line ids:', conversation_meta[0])
print('First set of conversation lines:', [conversation_lines[conv_id] for conv_id in conversation_meta[0]])

questions = []
answers = []

for meta in conversation_meta:
    for index in range(len(meta) - 1):
        questions.append(conversation_lines[meta[index]].lower())
        answers.append(conversation_lines[meta[index + 1]].lower())

print('No. of questions & answers:', (len(questions), len(answers)))

questions = remove_contractions(questions)
answers = remove_contractions(answers)

questions = remove_punctuations(questions)
answers = remove_punctuations(answers)

print('Writing processed dataset to file...')
dataset = pd.DataFrame({'question': questions, 'answer': answers})
dataset.to_csv('./data/processed_conv_data.csv', encoding='utf-8', index=False)
print('Dataset saved in ./data/processed_conv_data.csv!')

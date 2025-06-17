import pandas as pd
import numpy as np
import time
import heapq
import requests
from tqdm import tqdm
import os


ds = pd.read_csv('English_word_freq.csv')
ds.set_index('word', inplace=True, drop=False)

# print(ds.head())

def similar_set(word):
    similar_words=[]
    len_w = len(word)
    for text in ds['word']:
        len_t=len(text)
        if (abs(len_t-len_w)>3):
            continue
        elif ((text[0]==word[0]) and ((text[-1]==word[-1]) or(text[1]==word[1]))):
            score=0
            word_s = set(word); text_s = set(text)
            score = (len(word_s.intersection(text_s))-(len(word_s - text_s)+len(text_s-word_s)))/len_w
            # score = np.log(np.log(ds.at[text, 'count']))*(score/len_w)
            similar_words.append((text, score))
            
    return heapq.nlargest(10, similar_words, key=lambda x: x[1])

def Corrector( word):
    similar_words = similar_set(word)
    len_w = len(word)
    final_words = []
    for text, value in similar_words:
        score=0
        len_t=len(text)
        mx = min(len_t, len_w)
        for i in range(1, mx):
            if ((text[i]==word[i]) and (text[i-1]==word[i-1])):
                score+=1
            else:
                rev_text = text[::-1] 
                rev_word=word[::-1]
                mx = min(len_t, len_w)
                for j in range(1, i):
                    if ((rev_text[j]==rev_word[j]) and (rev_text[j-1]==rev_word[j-1])):
                        score+=1
            temp = score*value*(np.log(np.log(ds.at[text,'count']))) 
        final_words.append((text, temp, score, value, np.log(np.log(ds.at[text,'count']))))
    
    return max(final_words, key= lambda x:x[1])[0] if final_words else None

def testing_corrector_novig(file_url, verbose=False): 
    start=time.time()
    response = requests.get(file_url)
    response.raise_for_status()
    texts = response.text.strip().split('\n')
    cnt=0; right=0; unknown=0
    for text in tqdm(texts):
        correct, words = text.strip().split(':')
        words = words.strip().split(' ')
        for w in words:
            cnt+=1
            word = Corrector(w)
            right += (correct==word)
            if correct!=word:
                if correct not in ds.index:
                    unknown += 1
                if verbose:
                    print(f"{w}-->{word} out of {correct}")
    end = time.time()
    print("time", cnt/(end-start))
    print('correct', (right)*100/cnt, 'total', cnt, unknown)

def test_sample(text, verbose=False):
    start = time.time()
    lines = text
    total=0; right=0; unknown=0
    for line in tqdm(lines):
        correct, misspell = line.strip().split(':')
        correct = correct.strip()
        misspelled = misspell.strip().split(' ')
        # print(correct)
        for word in misspelled:
            # print(word)
            model_word = Corrector(word.strip())
            right+= (model_word == correct)
            total+=1
            if model_word != correct:
                if correct not in ds.index:
                    unknown += 1
                if verbose: print(f"Word: {word.strip()}, Model_word: {model_word}, Expected: {correct.strip()}")
    print(f"Total correct: {(right*100/total):.2f}, Unknown words: {unknown}")
    print(f"Time taken: {(total/(time.time() - start)):.2f} seconds")

if __name__ == "__main__":
    
    # test1 = 'https://norvig.com/spell-testset1.txt'
    # test2 = 'https://norvig.com/spell-testset2.txt'
    # testing model
    # testing_corrector_novig(test1, verbose=0)

    with open('test.txt', 'r') as f:
        lines = f.readlines()
        test_sample(lines, verbose=0)

    # print(Corrector('applle'))



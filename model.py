import pandas as pd
import numpy as np
import time
import heapq
import requests
from tqdm import tqdm


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
    lines = text.strip().split('\n')
    total=0; right=0; unknown=0
    for line in tqdm(lines):
        correct, misspell = line.split(':')
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

    test = '''search: searc serch saerch seerch
        please: plese pleas pleese pleaze
        online: onlne onine olnine onliene
        people: peaple peeple peple peapol
        system: systm sytem sistern sysstem
        health: helth heallth healthe healt
        email: emial eamil emale emaill
        school: shool scool schoool scholl
        review: reveiw revie reivew revu
        number: numbr numbeer nummber numbar
        about: abot abaut abouht abbout
        these: thees thes theze thesse
        their: thier ther theihr theyr
        video: vedio vidoe viedo vidio
        first: frist ferst forst fisrt
        music: musc musik musec musiq
        click: clik click clcik cliick
        price: prce priece priec prcie
        value: valie valew valoue valye
        report: repot reoprt reoprt repurt
        store: stroe storr sture stoar
        issue: isue isseu issu isshu
        other: oter othr othe othar
        order: oder ordr odrer orddr
        login: logn loggin ligin logen
        mobile: moblie moble mobiel mobille
        items: itams itmes itesm iitems
        design: desgin desgine deesgn desin
        access: accces acsess acces acesss
        forum: formu forrum furum forom
        office: ofice ofise offce offise
        books: boks boooks bokks boocs
        small: smal smaall smoll smull
        apply: aplay aply apli aplly
        money: mony monay muny monee
        offer: ofer oferr ofor offar
        today: todya tday tooday todaay
        below: bellow beloe belw belwo
        files: fiels fiels filez filse
        image: imgae imaje emage imige
        login: loggin logn logiin logein
        media: meedia medeia medea mdeia
        index: indek indix indax indez
        group: gruop groop grupe groub
        email: eamil emile emial emael
        event: evnt evenet evant eevent
        field: feild fild fielld feeld
        phone: phonne fone phne phoen
        level: leval leevl leveel lebel
        paper: papre papr paaper ppaper
        offer: oferr ooffer ofer ofar
        again: agian agin agaen agien
        among: amoung amog amon amogn
        check: chek checck cheak ceck
        white: wite whitte whight whyte
        login: logen loign logan lgin
        based: bsaed baesd basd baced
        email: eimail emael emal emaile
        women: wommen womn wemen womman
        child: chiled cild chil chield
        while: whille whil whiel whail
        photo: fotho poto phote phto
        entry: entrey entri enetry intri
        value: vlaue vuale valuw valoo
        quote: quate qoute quoto quotte
        login: loign logn logen loggin
        items: itams itmes itms iitems
        label: lable labal labeel labell
        below: belwo bellow beloe belaw
        click: click cllick cliick clikc
        build: bield bulid builld bild
        event: evant eevnt eveent evnet
        email: eemail emaill emial emael
        court: cort courtt corth courd
        clear: cleer cear clearr cleor
        march: marh marhc marchh marck
        hotel: hotle hoetl hotal hotil
        watch: wathc wach wacth watsh
        house: hause huse hous houze
        tools: tols toolz toolss toools
        think: thnik tnk thinkk thikn
        files: fils fiels filez filse
        north: norrth nirth nort norht
        wrong: wrng wron wrogn wroong
        point: ponit pint poit poynt
        press: pres prss prress prass
        place: plase plaec plcace plaece
        entry: enetry entri entra entiry
        drive: drve driev driev drrive
        image: imgae imaje imige emage
        email: emial eimail emaile emil
        study: studdy stuyd sstudy stude
        short: shrt shourt shor shurt
        watch: wathc wach waatch watsh
        store: stor storre stoar sture
        music: musick musc musiq muisc
        video: vedio vidoe vidio viedo
        guest: guets gest gues gueest
        print: prinnt prnt prient preent
        apple: aple appple aplle applo
        world: wrold whorld worled worlld
        light: ligt lighht lighte ligt
        forum: forrum foorum furum foarum
        again: agian agin agaen agaain
        night: nite nigth nigght niight
        march: marh maarch marhc marchh
        wrong: wrng wrogn wroong wrang
        '''
    
    test1 = 'https://norvig.com/spell-testset1.txt'
    test2 = 'https://norvig.com/spell-testset2.txt'
    # testing model
    # testing_corrector_novig(test1, verbose=0)
    # test_sample(test, verbose=0)

    print(Corrector('applle'))



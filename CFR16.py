import xml.etree.ElementTree as ET
import re
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
import time

# import the xml file as a single string
full_text = open('/Users/rodrigotejeida/Desktop/CFR16.xml').read()
# full_text = full_text.lower()

# Split the text on '§'
sections = full_text.split("<SECTNO>§", )
# Removing the first entry
sections = sections[1:]


# between method to obtain a substring between two other substring
# txt - string to be cut
# f - first string after which returned substring starts
# l - last substring before which returned substring ends
def between(txt, f, l):
    try:
        start = txt.index(f) + len(f)
        end = txt.index(l, start)
        return txt[start:end]
    except ValueError:
        return ""


# returns the Section Number, Subject and text after subject from text t1
def sec_sub(t1):
    # Obtains the Section Number
    sectno = between(t1, ' ', '</SECTNO>')
    # Obtains the subject
    subject = between(t1, '<SUBJECT>', '</SUBJECT>')
    # Obtains all text before subsections
    f = t1.split('</SUBJECT>')
    if (type(f) == str):
        return sectno, subject, f
    elif (subject == ''):
        return sectno, 'reserved', 1
    else:
        return sectno, subject, f[1]


# Cleans the file from xml operators from text t1
def clean(t1):
    # Remove <SECTION> and </SECTION>
    t1 = t1.replace('<SECTION>', '')
    t1 = t1.replace('</SECTION>', '')
    # Remove <SUBJECT> and </SUBJECT>
    t1 = t1.replace('<SUBJECT>', '')
    t1 = t1.replace('</SUBJECT>', '')
    # Remove <SECTNO> and </SECTNO>
    t1 = t1.replace('<SECTNO>', '')
    t1 = t1.replace('</SECTNO>', '')
    # Remove <E T="03"> and </E>
    t1 = t1.replace('<E T="03">', '')
    t1 = t1.replace('</E>', '')
    # Remove <NOTE> and </NOTE>
    t1 = t1.replace('<NOTE>', '')
    t1 = t1.replace('</NOTE>', '')
    # Remove <HD SOURCE="HED">Note:</HD>
    t1 = t1.replace('<HD SOURCE="HED">Note:</HD>', '')
    # # Remove <P> and </P>
    # t1 = t1.replace('<P>','')
    # t1 = t1.replace('</P>','')
    # Remove \n
    t1 = t1.replace('\n', '')
    return t1

# Divide the subsections if they exist
def divide(t1, sectno):
    r = re.compile('<P>\([a-z0-9]\)')
    l = r.split(t1)
    l = [i.replace('<P>', '').strip() for i in l]
    l = [i.replace('</P>', '').lower() for i in l]
    m = re.findall(r, t1)
    nm = list()
    ne = ''
    for i in range(len(m)):
        if (ord((m[i])[4]) >= 97 and ord((m[i])[4]) <= 122):
            ne = sectno + '.' + (m[i])[4]
            nm.append(ne)
        else:
            nn = ne + '.' + (m[i])[4]
            nm.append(nn)
    return nm, l

# Convert the full string into a pandas Data Frame
id = list()
sno = list()
sbj = list()
txt = list()
for i in range(len(sections)):
    t0 = sections[i]
    se, su, f = sec_sub(t0)
    if (f == 1):
        id.append(se)
        sno.append(se)
        sbj.append(su)
        txt.append('reserved')
    else:
        t0 = clean(f)
        nm, l = divide(t0, se)
        if (len(l) != len(nm)):
            id.append(se)
        id.extend(nm)
        sno.extend(len(l) * [se])
        sbj.extend(len(l) * [su])
        txt.extend(l)

df = pd.DataFrame()
df['id'] = id
df['section_no'] = sno
df['subject'] = sbj
df['text'] = txt

# Creating the corpus from the full cleaned string
cln_txt = clean(full_text)
cln_txt = cln_txt.replace('<P>', '')
cln_txt = cln_txt.replace('</P>', '')
cln_txt = cln_txt.lower()
corpus = nltk.word_tokenize(cln_txt)
# length corpus 651,542

# Remove duplicate entries from corpus
# corpus length 12811
corpus = list(set(corpus))
corpus = sorted(corpus)
# length corpus 18033

# remove all symbols and numbers
start = 0
end   = 0
for i in range(len(corpus)):
    if(ord((corpus[i])[0])==97):
        start = i
        break
for i in range(len(corpus)-1,0,-1):
    if(ord((corpus[i])[0])==122):
        end = i
        break
corpus = corpus[start:end]
# length corpus 12811

# Remove words with symbols from corpus
crp_temp = list()
for i in range(len(corpus)):
    if corpus[i].isalnum():
        crp_temp.append(corpus[i])
corpus = crp_temp
# corpus length 10835

# Remove commonly used english words from corpus
sw = stopwords.words('english')
for i in range(ord('a'),ord('z')+1):
    sw.append(chr(i))
crp_temp = [w for w in corpus if w not in sw]
corpus = crp_temp
# corpus length 10692

# Create the frequency of each word in the corpus
cps_freq = [0]*len(corpus)
t0 = time.clock()
for i in range(len(df)):
    print("%.2f" % ((i*100)/len(df)),'%')
    for j in range(len(corpus)):
        if (bool(re.findall(corpus[j],df.text[i]))):
            cps_freq[j] += 1
print('time',(time.clock()-t0)/60,'min')
# Time to run 91 min

# Entropy function
# returns p*log(p) if p>0
# returns 0 if p=0
# p - is the probability of occurence of some variable
def entropy(p):
    if p==0: return 0
    else: return p*np.log(p)

# Estimate probability and entropy of words in corpus
cps_prob = [0]*len(corpus)
cps_enpy = [0]*len(corpus)
N = sum(cps_freq)
for i in range(len(cps_freq)):
    cps_prob[i] = cps_freq[i]/N
    cps_enpy[i] = entropy(cps_prob[i])

df2 = pd.DataFrame()
df2['word'] = corpus
df2['frequency'] = cps_freq
df2['probability'] = cps_prob
df2['entropy'] = cps_enpy

# Sort the resulting Data Frame of word frequency, probability and entropy
t = df2.sort_values('frequency',ascending=False)
t = t.reset_index(drop=True)

# To print the 5 ngrams with the highest frequency
print('word:',t.word[0],'\nfrequency:',t.frequency[0],
      '\nprobability:',t.probability[0],'\nentropy:',t.entropy[0])
print('word:',t.word[1],'\nfrequency:',t.frequency[1],
      '\nprobability:',t.probability[1],'\nentropy:',t.entropy[1])
print('word:',t.word[2],'\nfrequency:',t.frequency[2],
      '\nprobability:',t.probability[2],'\nentropy:',t.entropy[2])
print('word:',t.word[3],'\nfrequency:',df2.frequency[3],
      '\nprobability:',t.probability[3],'\nentropy:',t.entropy[3])
print('word:',t.word[4],'\nfrequency:',t.frequency[4],
      '\nprobability:',t.probability[4],'\nentropy:',t.entropy[4])















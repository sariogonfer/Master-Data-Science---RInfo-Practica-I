import spacy
import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

nlp = spacy.load('en')

corpus_dir="./corpus_text"

texts = []

global_TF = pd.DataFrame()

counter = 0

for path in sorted([f for f in os.listdir(corpus_dir)
                    if f.endswith(".txt")]):
                 
                    
    with open(os.path.join(corpus_dir, path), "r") as f_:
    
        text = f_.read()
        tokens = nlp(text)
        words = []
        
        words = [t.lemma_ for t in tokens if not t.is_stop and not t.is_punct and str(t) not in ['PROPN', 'CONJ', 'ADP', 'DET']]
                
        word_TF = Counter(words)
        word_TF = pd.DataFrame.from_dict(word_TF, orient='index').reset_index()
        word_TF = word_TF.rename(index=str, columns={"index": "token", 0: "TF"})
        #word_TF = word_TF.sort_values('TF', ascending=False)
        
        #Almacenamos en la columna 'file' el identificador del fichero, comenzando por 0
        word_TF['files'] = counter

        #Concatenamos el DataFrame temporal en un DF global para tener todos los datos en el
        global_TF = global_TF.append(word_TF)
        f_.close()

        #Incrementamos el identificador del fichero en 1
        counter += 1
        

print(global_TF.groupby(['token']).agg({'files': np.size, 'TF' : np.sum}).sort_values(by=['files', 'TF'], ascending = False))

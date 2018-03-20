import spacy
import os
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

def _translate_text(text):
    from textblob import TextBlob

    tb = TextBlob(text)
    if tb.detect_language() == 'en':
        return text
    else:
        return str(tb.translate(to='en'))


if __name__ == "__main__":
    #Carga de paquete de inglés de spacy
    nlp = spacy.load('en')
    
    corpus_dir = "./corpus_text"

    global_TF = pd.DataFrame()

    counter = 0

    #Procesamos cada fichero de texto de noticias del directorio corpus_text
    for path in sorted([f for f in os.listdir(corpus_dir)
                        if f.endswith(".txt")]):
                     
        
        with open(os.path.join(corpus_dir, path), "r") as f_:
            #Lectura del texto, tranducción a inglés y tokenización
            tokens = nlp(_translate_text(f_.read()))
            
            #Recuperación de tokens que no sean stopwords, signos de puntuación ni una serie de tipologías de palabras (nombres propios, etc)
            words = [t.lemma_ for t in tokens if not t.is_stop and not t.is_punct and str(t) not in ['PROPN', 'CONJ', 'ADP', 'DET']]
                    
            #Contamos el número de palabras
            word_TF = Counter(words)
            word_TF = pd.DataFrame.from_dict(word_TF, orient='index').reset_index()
            word_TF = word_TF.rename(index=str, columns={"index": "token", 0: "TF"})
            
            #Almacenamos en la columna 'file' el identificador del fichero, comenzando por 0
            word_TF['files'] = counter

            #Concatenamos el DataFrame temporal en un DF global para tener todos los datos en el
            global_TF = global_TF.append(word_TF)
            f_.close()

            #Incrementamos el identificador del fichero en 1
            counter += 1
            

    print(global_TF.groupby(['token']).agg({'files': np.size, 'TF' : np.sum}).sort_values(by=['files', 'TF'], ascending = False))
    
    

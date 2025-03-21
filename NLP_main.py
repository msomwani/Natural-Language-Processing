import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus  import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag,ne_chunk

#Downloading NLTK resuorces

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

def nlp_pipeline(inp):
    sentences=sent_tokenize(inp)
    cleaned_sent=[]
    for sent in sentences:
        #removing puctuation and converting to lower case
        sent_clean=re.sub(r"[^a-zA-Z0-9]"," ",sent).lower()
        #Tokenizing(Making each word as a diffrent token)
        words=word_tokenize(sent_clean)
        #Removing stopwords
        #If you want to see what are the stop words uncomment the  next line
        #print(stopwords.words('english'))
        words=[w for w in words if w not in stopwords.words('english')]
        #stemming
        stemmed=[PorterStemmer().stem(w) for w in words]
        #lemmatization
        lemmatized=[WordNetLemmatizer().lemmatize(w) for w in stemmed]

        cleaned_sent.append(' '.join(lemmatized))
    all_tokens=word_tokenize(' '.join(cleaned_sent))
    pos_tags=pos_tag(all_tokens)
    ner_tree=ne_chunk(pos_tags)
    print("POS Tags:", pos_tags)
    print("Named Entities:", ner_tree)
    return ' '.join(cleaned_sent)

if __name__=="__main__":
    inp=input("Enter your text: ")
    cleaned=nlp_pipeline(inp)
    print("Cleaned text:",cleaned)

import nltk
# nltk.download()

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import state_union

from  nltk.tokenize import PunktSentenceTokenizer


Example_text  = "hello Mr. peeyush what is up? what are you doing?"

# print(word_tokenize(Example_text))

# print(sent_tokenize(Example_text))

stop_words= set(stopwords.words('english'))

# print(stop_words)

words = word_tokenize(Example_text)

filtered_sentence  = [w for w in words if not w in stop_words]

# print(filtered_sentence)

ps = PorterStemmer()

stemmed_words = [ps.stem(w) for w in words]

# print(stemmed_words)

train_text = state_union.raw("2005-GWBush.txt") 
test_text =  state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer  = PunktSentenceTokenizer(train_text)

tokenize = custom_sent_tokenizer.tokenize(test_text)

def process_content():
    try:
        for i in tokenize[:5]:
            word = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(word)
            chunkgram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            # chunkParser = nltk.RegexpParser(chunkgram)
            # chunkgram = r"""Chunk: {<.*>+}
                                    # }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkgram)
            chunked = chunkParser.parse(tagged)
            namedEnt = nltk.ne_chunk(tagged,binary=True)
            namedEnt.draw()
            # print(chunked)
            # for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
            #     print(subtree)
            # chunked.draw()
    except Exception as e:
        print(str(e))

process_content()

print(nltk.__file__)
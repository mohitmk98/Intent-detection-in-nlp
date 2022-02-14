from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

class Detect:

    def __init__(self):
        self.intent = ['cab' ,	'flight' ,	'hotel', 	'weather']
        self.model = load_model("weights.hdf5")
        with open("tokenizer.pkl", 'rb') as o:
            self.tokenizer = pickle.loads(o.read())


    def __preprocess(self, query):
        one_hot = [0]*4

        maxlen = 30
        query = self.tokenizer.texts_to_sequences([query])
        query = pad_sequences(query, maxlen)
        return np.array(query), np.array([one_hot])



    def process(self,query):
        query = self.__preprocess(query)

        pred_glove_val_y = self.model.predict(query)
        label = np.argmax(pred_glove_val_y[0])
        result = {"service" : "flight."+self.intent[label],  "confidence": pred_glove_val_y[0][label]}
        return result


# d = Detect()
# d.process("book me flight ", "cancel")
# {'confidence': 0.99938893, 'service': 'flight.book'}

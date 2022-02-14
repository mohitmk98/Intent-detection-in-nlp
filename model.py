from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

class Detect:
    
    def __init__(self):
        self.intent = ['affirmation' ,	'book' ,	'cancel', 	'check-in', 	'greet' ,	'negation' ,	'status']
        self.PrevIntents = ["book", "cancel", "check-in", "status"]
        self.model = load_model("modelwithCONV.h5")
        with open("tokenizer.pkl", 'rb') as o:
            self.tokenizer = pickle.loads(o.read())
            
            
    def __preprocess(self, query,prev_intent):
        one_hot = [0]*4
        if prev_intent!=-1:
            one_hot[prev_intent]=1
            
        maxlen = 30
        query = self.tokenizer.texts_to_sequences([query])
        query = pad_sequences(query, maxlen)
        return np.array(query), np.array([one_hot])
    
    
    
    def process(self,query, prev_intent=None):
        prev_intent = self.PrevIntents.index(prev_intent) if prev_intent is not None else -1

        query, prev_intent = self.__preprocess(query, prev_intent)
        
        pred_glove_val_y = self.model.predict([query, prev_intent])
        label = np.argmax(pred_glove_val_y[0])
        result = {"service" : "flight."+self.intent[label],  "confidence": pred_glove_val_y[0][label]}
        return result
    
    
# d = Detect()
# d.process("book me flight ", "cancel")
# {'confidence': 0.99938893, 'service': 'flight.book'}
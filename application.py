import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('/path/to/your/chatbot_model.h5')
import json
import random

#Add relevant filepaths below
intents = json.loads(open('/path/to/your/intents.json').read())
words = pickle.load(open('/path/to/your/words.pkl','rb'))
classes = pickle.load(open('/path/to/your/classes.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the input pattern 
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold of 0.25 
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

import tkinter as tk
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk 

def send():
    
    msg = EntryBox.get("1.0", "end-1c").strip()
    EntryBox.delete("1.0", "end")

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n', "user")
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        # chatbot response logic
        res = chatbot_response(msg)
        ChatLog.insert(tk.END, "Bot: " + res + '\n\n', "bot")

        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

#Creating Chat User Interface
base = tk.Tk()
base.title("UH Query Bot")
base.geometry("450x500")

# Create Chat window
ChatLog = tk.Text(base, bd=0, height="11", width="50", font="Arial", bg="light yellow") #set the attributes as desired
ChatLog.config(state=tk.DISABLED)
ChatLog.tag_configure("user", foreground="blue")
ChatLog.tag_configure("bot", foreground="green")

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="sb_v_double_arrow")
ChatLog['yscrollcommand'] = scrollbar.set

# Create a text input area with the same width as ChatLog
EntryBox = tk.Text(base, bd=0, bg="light cyan",fg="black", height="5", width="50", font=("Arial", 12)) #set the attributes as desired

# Load the send button image
send_image = Image.open("send.png")  # Replace "send.png" with your image file
send_image = send_image.resize((30, 30), Image.LANCZOS)
send_icon = ImageTk.PhotoImage(send_image)

# Place the send button inside the text input area
SendButton = tk.Button(EntryBox, image=send_icon, bd=0, command=send)
SendButton.image = send_icon  # Store a reference to the image

SendButton.image = send_image 
SendButton.place(relx=1, rely=1, anchor="se")

# Place all components on the screen
ChatLog.grid(row=0, column=0, padx=6, pady=6, columnspan=2)
scrollbar.grid(row=0, column=2, padx=0, pady=6, sticky="ns")
EntryBox.grid(row=1, column=0, padx=6, pady=6, columnspan=2)

base.mainloop()

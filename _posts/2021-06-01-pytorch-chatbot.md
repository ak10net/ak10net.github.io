---
layout: post
title: "Deep learning Chatbot"
author: "Ankit"
tags: chatbot pytorch 
excerpt_separator: <!--more-->
---

## In this post we will try to build FAQ chatbot<!--more-->

Let's get some theory out of the way

### What is a chatbot?
- A Chatbot is a service, powered by rules and sometimes artificial intelligence.
- The Chatbot service could be FAQ based, flow based or open ended conversations.
- Chatbot have some form of text interface such as Website extension, Whatsapp, Telegram.
- Chatbot works by understanding the text responses coming from users and producing reasonable replies.
- Chatbot work by recognizing intents and entities and build response around them

### Why do we need chatbot?
- Consumers have easy access to providers through technology 
- Consumers are demanding round-the-clock service
- Business chatbots are goal oriented. Chatbot is a tag-team partner to improve services.
- Chatbot bridges the gap between service demand and providers service ability
- Reduce the load for routine FAQs and self-service

### What could be potential features of chatbot?
- Automatically finds answers from knowledge base
- Domain specific with corpus-based training that ensures contextual conversation leveraging previous interaction history
- Multi-turn conversation – Can assess the user intent, context and drive personalized conversations
- Multilingual – converse in the language of your customers
- Ensures consistent experience for user across channels through retention of context

### Types of chatbot
There are three types of chatbot
+ FAQ bots that give fixed response or can query for user
+ Flow based chatbot that can maintain state of conversation
+ Open ended chatbot that exhibit certain levels of intelligence and can maintain conversations

![Types of chatbots](/assets/types_of_chatbot.png)

### Basic architecture of a FAQ chatbot

According to buisness domain business rules can be framed that will guide conversation flow.
Once conversation flow is build, intents and responses can be implemented. Now, model can be trained on 
intents and can be saved for later invocation at the time of conversation. When app is run pre-trained
model is queried for response that are served on web page.
This is very simple architecture, But in reality it can get complex.

![Chatbot architecture](/assets/chatbot_pipeline.png)

Let's get into code now:-

### What all do we need to build the chatbot?

+ Intents file to train the pytorch classifier 'intents.json'

This file host all the intents that we want classifier to train on
```json
{
  "intents": [
  	{
      "tag": "payments",
      "patterns": [
        "Do you take credit cards?",
        "Do you accept Mastercard?",
        "Can I pay with Paypal?",
        "Are you cash only?"
      ],
      "responses": [
        "We accept VISA, Mastercard and Paypal",
        "We accept most major credit cards, and Paypal"
      ]
    },
   ]
}
```

+ Text processing utility 'utils.py'

Basic text processing pipeline
```python
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
```

+ Basic model 'model.py'

Vanilla torch model
```python
import torch
import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
```

+ Training script 'train.py'

Training script to train model on intents. To train the model run 'python train.py'
that will save the model for later use in app
```python
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

num_epochs = 100
batch_size = 8
learning_rate = 0.01
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')
data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
```

+ Interface for interaction 'chat.py'

To play with the bot run 'python chat.py' and bot will open in browser.
```python
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from flask import Flask, render_template, request

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jeeves"
app = Flask(__name__)
@app.route("/")
def home():    
    return render_template("home.html") 

@app.route("/get")
def get_bot_response():    
    sentence = request.args.get('msg')
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
            	return random.choice(intent['responses'])

 
if __name__ == "__main__":    
    app.run()
```

+ HTML file under templates folder 'html.py'

Basic HTML template to interact with the bot. It takes the user input and send to bot api endpoint for 
intent prediction. Once bot predicts intent successfully, it responds with a prefixed response which gets
displayed in conversation window.

```html
	<div id='chatborder'>
    	<div id="chatbox">
          <p class="botText">
            <span>Hi!</span>
          </p>
        </div>     
  	</div>
  	<div id="userInput">
          <input id="textInput" type="text" name="msg" placeholder="How can i help you ?" onfocus="placeHolder()">
        </div>  
  	</div>
    <script>
        function getBotResponse() {
          var rawText = $("#textInput").val();
          var userHtml = '<p class="userText"><span>' + rawText + "</span></p>";
          $("#textInput").val("");
          $("#chatborder").append(userHtml);
          document
            .getElementById("userInput")
            .scrollIntoView({ block: "start", behavior: "smooth" });
          $.get("/get", { msg: rawText }).done(function(data) {
            var botHtml = '<p class="botText"><span>' + data + "</span></p>";
            $("#chatborder").append(botHtml);
            document
              .getElementById("userInput")
              .scrollIntoView({ block: "start", behavior: "smooth" });
          });
        }
        $("#textInput").keypress(function(e) {
          if (e.which == 13) {
            getBotResponse();
          }
        });
      </script>
    </div>
```

**Voila! Here is your chatbot**

![Chatbot](/assets/chatbot.png)


Thanks to [python engineer](https://www.youtube.com/channel/UCbXgNpp0jedKWcQiULLbDTA)

Please checkout his youtube series.





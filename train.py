
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r', encoding= "utf-8") as f:
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


ignore_words = ['?', '!', '.', ',', '¿', '¡']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(all_words)
""" print(tags) """

x_train = []
y_train = []

for (pattern_sentence, tag) in  xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label) #CrossEntropyLoss

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = torch.LongTensor(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def  __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# Hyperparameters  
batch_size = 10
hidden_size = 10
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 2000

""" print(input_size, len(all_words))
print(output_size, tags) """

datset = ChatDataset()
train_loader = DataLoader(dataset=datset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if  (epoch + 1) % 100 == 0 :
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

print(f'final loss, loss = {loss.item():.4f}')   

data = {
   "model_state": model.state_dict(),
   "input_size": input_size,
   "hidden_size": hidden_size,
   "output_size": output_size,
   "all_words": all_words,
   "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. file saved to {FILE}')

# Test the network on the test data
""" total, correct = 0, 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print('Test Accuracy: {}%'.format(100 * correct / total)) """
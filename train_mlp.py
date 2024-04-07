import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from utils import seed_everything
seed_everything(384)

direction_map = {'Left': 0, 'Right': 1}
labels_map = {'Left': 0, 'Right': 1}

with open(r'dataset\prior_dict',encoding='utf-8') as f:
    priors = eval(f.read())
    #prior_data = np.array([priors[str(i)] if str(i) in priors else [0, 0, 0] for i in range(26)])

with open(r'dataset\combine_dataset.txt',encoding='utf-8') as f:
    data = f.readlines()
    for idx in range(len(data)):
        data[idx] = eval(data[idx])

categories = []
directions = []
values = []
prior_data = []
for item, _ in data:
    for sub_item in item:
        categories.append([sub_item[1]])
        directions.append(direction_map[sub_item[2]])
        values.append(sub_item[3])
        prior_data.append(priors.get(str(sub_item[1]), [0, 0, 0]))

# onehot
encoder = OneHotEncoder(sparse=False)
one_hot_categories = encoder.fit_transform(categories)

max_length = max(len(item[0]) for item in data)
feature_dim = one_hot_categories.shape[1] + 2 + 3  # 加2是因为还包括方向和数值,加3是因为先验长度为3
processed_data = np.zeros((len(data), max_length, feature_dim))

index = 0
for i, (item, _) in enumerate(data):
    for j in range(len(item)):
        processed_data[i, j, :-5] = one_hot_categories[index]
        processed_data[i, j, -5:-2] = prior_data[index]
        processed_data[i, j, -2] = directions[index]
        processed_data[i, j, -1] = values[index]
        index += 1

labels = np.array([labels_map[label] for _, label in data])
#print(labels.shape,processed_data.shape)
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# prepare dataset
X = processed_data.reshape(processed_data.shape[0], -1)
print(X.shape)
Y = labels

# convert to torch tensors
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# dataloader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

input_size = X.shape[1]
hidden_size = 250  # 可以调整
num_classes = 2
model = SimpleNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
#print(len(test_loader))
# train
num_epochs = 300
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # test
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            #print(predicted,labels)
            print(f'Accuracy of the model on the test set: {100 * correct / total} %')

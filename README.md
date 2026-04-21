# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model
<img width="2048" height="1162" alt="nn" src="https://github.com/user-attachments/assets/c2910205-dfbf-4355-8143-1a3d74eead06" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:yaswanth kumar

### Register Number:212224230310

```class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')



```

### Dataset Information
<img width="236" height="257" alt="image" src="https://github.com/user-attachments/assets/e0f9d331-58e6-4f19-8bbf-ac60441cf624" />

### OUTPUT
Epoch [0/2000], Loss: 873.640076
Epoch [200/2000], Loss: 796.774597
Epoch [400/2000], Loss: 505.085938
Epoch [600/2000], Loss: 390.118164
Epoch [800/2000], Loss: 372.173584
Epoch [1000/2000], Loss: 353.497528
Epoch [1200/2000], Loss: 334.146484
Epoch [1400/2000], Loss: 316.845001
Epoch [1600/2000], Loss: 305.790710
Epoch [1800/2000], Loss: 300.603210


### Training Loss Vs Iteration Plot
  <img width="748" height="572" alt="image" src="https://github.com/user-attachments/assets/bff4a962-ed4b-46de-91fd-6c5e5a997551" />


### New Sample Data Prediction
<img width="375" height="39" alt="image" src="https://github.com/user-attachments/assets/84fa1637-6f22-4c9d-8d2e-221cbe259af6" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.

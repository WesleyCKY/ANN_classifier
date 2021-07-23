# ANN_classifier
ANN workflow design:
1.	Config the network size and activation function.
a.	Preset network structure and activation function
2.	Initialize the model with random weights.
a.	Each connection in the model has a weight that can be randomly given a value at initialization time.
3.	Predict the result according to the input of data and weight.
a.	introduce errors to measure the deviation between predict and true data. The initial parameters are set randomly, so the obtained results must be quite different from the real results, so we should calculate an error here, which reflects the difference between the predicted results and the real results.
4.	Adjust the weight from the learning weight by using activation function. 
a.	Set the learning rate, which is aimed at the error. After each error is obtained, the weight on the connection is adjusted according to the ratio of the error, and a smaller error is expected in the next calculation. After several cycles, we can choose to stop and output the model when we reach a lower loss value, or we can choose a definite cycle to end.

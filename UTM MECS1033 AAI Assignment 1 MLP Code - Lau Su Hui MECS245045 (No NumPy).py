import random
import math


def sigmoid(x):
    if x < -700: return 0
    if x > 700: return 1
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(output):
    return output * (1 - output)


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_hidden = [0.0] * hidden_size
        self.bias_output = [0.0] * output_size

    def forward(self, inputs):
        self.hidden_inputs = [0.0] * len(self.bias_hidden)
        self.hidden_outputs = [0.0] * len(self.bias_hidden)

        for j in range(len(self.bias_hidden)):
            activation = self.bias_hidden[j]
            for i in range(len(inputs)):
                activation += inputs[i] * self.weights_input_hidden[i][j]
            self.hidden_inputs[j] = activation
            self.hidden_outputs[j] = sigmoid(activation)

        self.final_outputs = [0.0] * len(self.bias_output)

        for k in range(len(self.bias_output)):
            activation = self.bias_output[k]
            for j in range(len(self.hidden_outputs)):
                activation += self.hidden_outputs[j] * self.weights_hidden_output[j][k]
            self.final_outputs[k] = sigmoid(activation)

        return self.final_outputs

    def train(self, inputs, expected_output):
        output = self.forward(inputs)

        output_deltas = [0.0] * len(output)
        for k in range(len(output)):
            error = expected_output[k] - output[k]
            output_deltas[k] = error * sigmoid_derivative(output[k])

        hidden_deltas = [0.0] * len(self.hidden_outputs)
        for j in range(len(self.hidden_outputs)):
            error = 0.0
            for k in range(len(output)):
                error += output_deltas[k] * self.weights_hidden_output[j][k]
            hidden_deltas[j] = error * sigmoid_derivative(self.hidden_outputs[j])

        for j in range(len(self.weights_hidden_output)):
            for k in range(len(self.weights_hidden_output[0])):
                change = output_deltas[k] * self.hidden_outputs[j]
                self.weights_hidden_output[j][k] += self.learning_rate * change

        for i in range(len(self.weights_input_hidden)):
            for j in range(len(self.weights_input_hidden[0])):
                change = hidden_deltas[j] * inputs[i]
                self.weights_input_hidden[i][j] += self.learning_rate * change

        for k in range(len(self.bias_output)):
            self.bias_output[k] += self.learning_rate * output_deltas[k]
        for j in range(len(self.bias_hidden)):
            self.bias_hidden[j] += self.learning_rate * hidden_deltas[j]


if __name__ == "__main__":
    dataset = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]

    print("--- 1. INITIALIZATION ---")
    nn = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)

    print("\n--- 2. PRE-TRAINING CHECK ---")
    print(f"{'Input':<15} | {'Prediction':<10}")
    for inputs, expected in dataset:
        print(f"{str(inputs):<15} | {nn.forward(inputs)[0]:.5f}")

    print("\n--- 3. TRAINING PHASE ---")
    epochs = 10000
    for epoch in range(epochs):
        total_error = 0
        for inputs, expected in dataset:
            nn.train(inputs, expected)
            output = nn.forward(inputs)[0]
            total_error += (expected[0] - output) ** 2

        if epoch == 0 or (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}: Loss {total_error:.6f}")

    print("\n--- 4. FINAL RESULTS ---")
    print("-" * 35)
    print(f"{'Input':<15} | {'Expected':<8} | {'Predicted':<10}")
    print("-" * 35)

    for inputs, expected in dataset:
        prediction = nn.forward(inputs)[0]
        rounded = round(prediction)
        print(f"{str(inputs):<15} | {expected[0]:<8} | {prediction:.5f} ({rounded})")
    print("-" * 35)

import numpy as np

# --- CONFIGURATION ---
# This ensures the random numbers are the same every time you run it.
# Great for beginners: guarantees your "calibration" works on the first try.
np.random.seed(1)


# --- HELPER FUNCTIONS (The Math Parts) ---
def sigmoid(x):
    """
    Activation Function: Acts like a gate.
    Squashes any number into a range between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Used during 'calibration' (backpropagation).
    Tells us how much to adjust the weights.
    """
    return x * (1 - x)


# --- THE NEURAL NETWORK CLASS ---
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate

        # Initialize the "knobs" (Weights) of the machine with random values.
        # Layer 1: Weights between Input and Hidden Layer
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))

        # Layer 2: Weights between Hidden and Output Layer
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        """
        Step 1: The Forward Pass (Prediction)
        The data flows through the network layers.
        """
        # 1. Process Input -> Hidden Layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # 2. Process Hidden -> Output Layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, output):
        """
        Step 2: The Backward Pass (Correction)
        We compare the prediction to the real answer and adjust weights.
        """
        # Calculate the Error (Difference between Truth and Prediction)
        error = y - output

        # --- CALCULATE ADJUSTMENTS (The Calculus Part) ---
        # How much should we change the Output weights?
        d_output = error * sigmoid_derivative(output)

        # How much should we change the Hidden layer weights?
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # --- UPDATE THE WEIGHTS (The Learning Part) ---
        # Adjust Hidden -> Output weights
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        # Adjust Input -> Hidden weights
        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        print(f"Starting training process for {epochs} cycles...")

        for i in range(epochs):
            # 1. Predict
            output = self.forward(X)

            # 2. Correct (Learn)
            self.backward(X, y, output)

            # 3. Report Progress (Every 1000 times)
            if (i % 1000) == 0:
                loss = np.mean(np.square(y - output))  # How wrong is the model?
                print(f"Epoch {i}: Loss {loss:.6f}")  # We want Loss to go to 0


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Prepare Data (XOR Logic Gate)
    # Inputs: [0,0], [0,1], [1,0], [1,1]
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Targets: 0, 1, 1, 0
    y = np.array([[0], [1], [1], [0]])

    print("--- 1. INITIALIZATION ---")
    # Network Setup: 2 Inputs -> 4 Hidden Neurons -> 1 Output
    nn = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)

    print("\n--- 2. PRE-TRAINING CHECK ---")
    print("Predictions before learning (Should be random):")
    print(nn.forward(X))

    print("\n--- 3. TRAINING PHASE ---")
    nn.train(X, y, epochs=10000)

    print("\n--- 4. FINAL RESULTS ---")
    final_output = nn.forward(X)

    # Print a clean report table
    print(f"{'Input':<15} {'Expected':<10} {'Predicted':<15}")
    print("-" * 45)
    for i in range(len(X)):
        # We round the random numbers to make them readable strings
        pred_val = f"{final_output[i][0]:.4f}"
        print(f"{str(X[i]):<15} {str(y[i]):<10} {pred_val:<15}")
import numpy as np

class SimpleBugPredictor:
    """
    Dead simple neural network: learns to predict bugs from code complexity
    Think: "Smart calculator that learns from experience"
    """
    
    def __init__(self):
        # Just two numbers that start random - these are our "experience"
        self.weight = np.random.random()  # How much complexity matters (0-1)
        self.bias = np.random.random()    # Base assumption about bugs (0-1)
        
        print(f"🧠 New AI starting with random guesses:")
        print(f"   Weight: {self.weight:.2f}, Bias: {self.bias:.2f}")
    
    def predict(self, complexity):
        """
        Predict if code has bugs based on complexity (0-10 scale)
        Returns probability between 0-1
        """
        # Simple formula: prediction = complexity * weight + bias
        raw_score = complexity * self.weight + self.bias
        
        # Convert to probability (0-1) using sigmoid
        probability = 1 / (1 + np.exp(-raw_score))
        return probability
    
    def learn(self, complexity, actual_had_bugs):
        """
        Learn from one example: adjust weight and bias if we were wrong
        """
        # Make prediction
        prediction = self.predict(complexity)
        
        # How wrong were we?
        error = actual_had_bugs - prediction
        
        # Adjust our "experience" slightly
        learning_rate = 0.1
        self.weight += learning_rate * error * complexity
        self.bias += learning_rate * error
        
        return abs(error)

def demo():
    """Simple demo showing learning in action"""
    
    ai = SimpleBugPredictor()
    
    # Training examples: [complexity, had_bugs]
    examples = [
        (1, 0),   # Simple code, no bugs
        (2, 0),   # Simple code, no bugs  
        (8, 1),   # Complex code, had bugs
        (9, 1),   # Very complex, had bugs
        (3, 0),   # Medium simple, no bugs
        (7, 1),   # Pretty complex, had bugs
    ]
    
    print("\n📚 LEARNING PHASE:")
    print("-" * 40)
    
    # Train multiple times
    for round_num in range(3):
        print(f"\nRound {round_num + 1}:")
        total_error = 0
        
        for complexity, had_bugs in examples:
            error = ai.learn(complexity, had_bugs)
            prediction = ai.predict(complexity)
            total_error += error
            
            print(f"  Complexity {complexity} → Predicted: {prediction:.2f}, Actual: {had_bugs}, Error: {error:.2f}")
        
        print(f"  📊 Average error this round: {total_error/len(examples):.3f}")
        print(f"  🧠 Current experience: weight={ai.weight:.2f}, bias={ai.bias:.2f}")
    
    print("\n🚀 TESTING ON NEW CODE:")
    print("-" * 40)
    
    # Test on new examples
    test_cases = [1, 3, 5, 7, 10]
    
    for complexity in test_cases:
        prob = ai.predict(complexity)
        risk_level = "HIGH" if prob > 0.7 else "MED" if prob > 0.3 else "LOW"
        print(f"  Complexity {complexity} → Bug probability: {prob:.1%} ({risk_level} RISK)")

if __name__ == "__main__":
    demo()
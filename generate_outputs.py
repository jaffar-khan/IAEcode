# generate_outputs.py
import os
from app import MultiAgentSystem

def generate_outputs():
    """Generate output files for all test scenarios"""
    system = MultiAgentSystem()
    
    scenarios = [
        ("simple_query", "What are the main types of neural networks?"),
        ("complex_query", "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs."),
        ("memory_test", "What did we discuss about neural networks earlier?"),
        ("multi_step", "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges."),
        ("collaborative", "Compare two machine-learning approaches and recommend which is better for our use case.")
    ]
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    for filename, query in scenarios:
        print(f"Processing {filename}...")
        result = system.process_query(query)
        
        # Save to file
        with open(f"outputs/{filename}.txt", "w") as f:
            f.write(f"Query: {query}\n\n")
            if result["status"] == "success":
                f.write(f"Response: {result['response']}\n")
            else:
                f.write(f"Error: {result.get('message', 'Unknown error')}\n")
    
    print("All outputs generated!")

if __name__ == "__main__":
    generate_outputs()
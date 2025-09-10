import json
import os
from datetime import datetime
from agents import ResearchAgent, AnalysisAgent, MemoryAgent, Coordinator


class MultiAgentSystem:
    """Main multi-agent system class"""
    
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.memory_agent = MemoryAgent()
        self.coordinator = Coordinator(self.research_agent, self.analysis_agent, self.memory_agent)
        
        print("Multi-Agent System Initialized")
        print(f"Agents: {self.research_agent}, {self.analysis_agent}, {self.memory_agent}, {self.coordinator}")
    
    def process_query(self, query: str):
        """Process a user query and return the response"""
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        result = self.coordinator.process_query(query)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\nRESPONSE (in {processing_time:.2f}s):")
        if result["status"] == "success":
            print(result["response"])
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")
        
        return result


def run_test_scenarios():
    """Run the test scenarios"""
    system = MultiAgentSystem()
    
    scenarios = [
        ("Simple Query", "What are the main types of neural networks?"),
        ("Complex Query", "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs."),
        ("Memory Test", "What did we discuss about neural networks earlier?"),
        ("Multi-step", "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges."),
        ("Collaborative", "Compare two machine-learning approaches and recommend which is better for our use case.")
    ]
    
    results = {}
    
    for name, query in scenarios:
        print(f"\n\n{'#'*80}")
        print(f"RUNNING SCENARIO: {name}")
        print(f"{'#'*80}")
        results[name] = system.process_query(query)
    
    return results


def generate_output_files():
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
        print(f"\n{'='*50}")
        print(f"GENERATING OUTPUT FOR: {filename}")
        print(f"{'='*50}")
        
        result = system.process_query(query)
        
        # Save to file
        with open(f"outputs/{filename}.txt", "w", encoding="utf-8") as f:
            f.write(f"Query: {query}\n\n")
            f.write("Agent Interaction Trace:\n")
            f.write("=" * 50 + "\n")
            
            if result["status"] == "success":
                f.write(f"Response: {result['response']}\n")
            else:
                f.write(f"Error: {result.get('message', 'Unknown error')}\n")
        
        print(f"Saved output to outputs/{filename}.txt")
    
    print("\nAll outputs generated!")


if __name__ == "__main__":
    # Run the test scenarios and generate output files
    generate_output_files()
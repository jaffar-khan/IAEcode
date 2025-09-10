# agents.py
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.id = str(uuid.uuid4())[:8]
    
    @abstractmethod
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def __str__(self):
        return f"{self.name}_{self.id}"


class ResearchAgent(Agent):
    """Simulates information retrieval from a knowledge base"""
    
    def __init__(self):
        super().__init__("ResearchAgent")
        # Mock knowledge base
        self.knowledge_base = {
            "neural networks": {
                "types": ["Feedforward Neural Networks", "Convolutional Neural Networks (CNN)", 
                         "Recurrent Neural Networks (RNN)", "Long Short-Term Memory Networks (LSTM)",
                         "Gated Recurrent Units (GRU)", "Radial Basis Function Networks (RBFN)",
                         "Multilayer Perceptrons (MLP)", "Self-Organizing Maps (SOM)",
                         "Deep Belief Networks (DBN)", "Restricted Boltzmann Machines (RBM)",
                         "Autoencoders"],
                "description": "Neural networks are computing systems inspired by the human brain that consist of interconnected nodes (neurons) that process and transmit information."
            },
            "transformer architectures": {
                "types": ["Original Transformer", "BERT", "GPT", "T5", "RoBERTa", "DistilBERT", "XLNet"],
                "computational_efficiency": "Transformers are computationally intensive due to self-attention mechanism which has O(n²) complexity. Recent variants like Linformer and Performer aim to improve efficiency.",
                "trade_offs": "Better performance on sequence tasks vs higher computational requirements. Parallelization advantages during training vs memory constraints."
            },
            "reinforcement learning": {
                "papers": [
                    {"title": "Playing Atari with Deep Reinforcement Learning", "year": 2013, "methodology": "Q-learning with CNN"},
                    {"title": "Human-level control through deep reinforcement learning", "year": 2015, "methodology": "Deep Q-Network (DQN)"},
                    {"title": "Proximal Policy Optimization Algorithms", "year": 2017, "methodology": "Policy optimization with clipped objective"},
                    {"title": "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", "year": 2018, "methodology": "Maximum entropy reinforcement learning"}
                ],
                "challenges": ["Sample inefficiency", "Credit assignment problem", "Exploration-exploitation tradeoff", "Reward engineering"]
            },
            "machine learning optimization techniques": {
                "techniques": ["Gradient Descent", "Stochastic Gradient Descent (SGD)", "Mini-batch Gradient Descent", 
                              "Momentum", "Nesterov Accelerated Gradient", "Adagrad", "Adadelta", "RMSprop", "Adam"],
                "effectiveness": "Adam generally performs well across various tasks. SGD with momentum often achieves better generalization for deep learning. Choice depends on problem characteristics."
            }
        }
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate research by retrieving information from the knowledge base"""
        query = task.get("query", "")
        context = task.get("context", {})
        
        print(f"[{self}] Researching: {query}")
        
        # Simple keyword-based retrieval
        results = {}
        for keyword in self.knowledge_base.keys():
            if keyword in query.lower():
                results[keyword] = self.knowledge_base[keyword]
        
        # If no direct match, try to find related content
        if not results:
            for keyword, content in self.knowledge_base.items():
                if any(word in query.lower() for word in keyword.split()):
                    results[keyword] = content
        
        return {
            "status": "success",
            "results": results,
            "query": query,
            "context": context,
            "confidence": 0.8 if results else 0.3
        }


class AnalysisAgent(Agent):
    """Performs comparisons, reasoning, and simple calculations"""
    
    def __init__(self):
        super().__init__("AnalysisAgent")
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the provided data"""
        data = task.get("data", {})
        analysis_type = task.get("analysis_type", "general")
        context = task.get("context", {})
        
        print(f"[{self}] Analyzing data with type: {analysis_type}")
        
        if analysis_type == "comparison" and "techniques" in data:
            # Simulate comparison analysis
            techniques = data.get("techniques", [])
            analysis = f"Based on the research data, I've analyzed {len(techniques)} techniques. "
            analysis += "Adam and RMSprop generally perform well for most deep learning tasks. "
            analysis += "SGD with momentum can achieve better generalization but requires careful tuning. "
            analysis += "The choice depends on your specific problem characteristics and constraints."
            
            return {
                "status": "success",
                "analysis": analysis,
                "compared_items": techniques,
                "context": context,
                "confidence": 0.9
            }
        
        elif analysis_type == "efficiency" and "computational_efficiency" in data:
            # Simulate efficiency analysis
            efficiency_data = data.get("computational_efficiency", "")
            analysis = f"The computational efficiency analysis shows: {efficiency_data}. "
            analysis += "Transformer architectures are powerful but computationally intensive. "
            analysis += "Recent improvements like sparse attention and model distillation help address efficiency concerns."
            
            return {
                "status": "success",
                "analysis": analysis,
                "context": context,
                "confidence": 0.85
            }
        
        elif analysis_type == "methodologies":
            # Analyze research methodologies
            papers = data.get("papers", [])
            methodologies = {}
            for paper in papers:
                methodology = paper.get("methodology", "unknown")
                methodologies[methodology] = methodologies.get(methodology, 0) + 1
            
            analysis = f"Analyzed {len(papers)} papers. Found methodologies: "
            for method, count in methodologies.items():
                analysis += f"{method} ({count} papers), "
            analysis += ". Common trends show increased use of deep learning approaches in recent years."
            
            return {
                "status": "success",
                "analysis": analysis,
                "methodologies": methodologies,
                "context": context,
                "confidence": 0.9
            }
        
        # Default analysis
        analysis = "I've analyzed the provided data. It appears to be relevant to the query. "
        analysis += "Further specific analysis could be performed with more precise instructions."
        
        return {
            "status": "success",
            "analysis": analysis,
            "context": context,
            "confidence": 0.7
        }


class MemoryAgent(Agent):
    """Manages long-term storage, retrieval, and context updates"""
    
    def __init__(self):
        super().__init__("MemoryAgent")
        self.memory_store = {
            "conversations": [],
            "knowledge": [],
            "agent_states": {}
        }
        
        # Simple vector storage simulation (using word frequency vectors)
        self.vector_index = {}  # word -> [document_ids]
        self.documents = {}     # id -> document
    
    def _create_vector(self, text: str) -> Dict[str, int]:
        """Create a simple word frequency vector"""
        words = text.lower().split()
        vector = {}
        for word in words:
            # Simple stemming - just remove common suffixes
            if word.endswith('s'):
                word = word[:-1]
            if word.endswith('ing'):
                word = word[:-3]
            if word.endswith('ed'):
                word = word[:-2]
                
            vector[word] = vector.get(word, 0) + 1
            
            # Update inverted index
            if word not in self.vector_index:
                self.vector_index[word] = set()
        return vector
    
    def _cosine_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
            
        # Get all unique words
        words = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in words)
        mag1 = sum(val ** 2 for val in vec1.values()) ** 0.5
        mag2 = sum(val ** 2 for val in vec2.values()) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory operations: store, retrieve, search"""
        operation = task.get("operation", "retrieve")
        data = task.get("data", {})
        context = task.get("context", {})
        
        print(f"[{self}] Performing memory operation: {operation}")
        
        if operation == "store":
            # Store information in memory
            memory_type = data.get("type", "knowledge")
            content = data.get("content", {})
            topics = data.get("topics", [])
            source = data.get("source", "unknown")
            
            memory_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().isoformat()
            
            memory_item = {
                "id": memory_id,
                "type": memory_type,
                "content": content,
                "topics": topics,
                "source": source,
                "timestamp": timestamp,
                "confidence": data.get("confidence", 0.5)
            }
            
            # Add to appropriate store
            if memory_type == "conversation":
                self.memory_store["conversations"].append(memory_item)
            elif memory_type == "knowledge":
                self.memory_store["knowledge"].append(memory_item)
            elif memory_type == "agent_state":
                agent_id = source
                if agent_id not in self.memory_store["agent_states"]:
                    self.memory_store["agent_states"][agent_id] = []
                self.memory_store["agent_states"][agent_id].append(memory_item)
            
            # Index for search
            text_content = str(content)
            self.documents[memory_id] = {
                "text": text_content,
                "vector": self._create_vector(text_content),
                "item": memory_item
            }
            
            for word in self._create_vector(text_content).keys():
                self.vector_index[word].add(memory_id)
            
            return {
                "status": "success",
                "message": f"Stored in memory with ID: {memory_id}",
                "memory_id": memory_id,
                "context": context
            }
        
        elif operation == "retrieve":
            # Retrieve information from memory
            query = data.get("query", "")
            memory_type = data.get("memory_type", None)
            max_results = data.get("max_results", 5)
            
            results = []
            
            # Keyword search
            query_words = set(query.lower().split())
            matched_ids = set()
            
            for word in query_words:
                if word in self.vector_index:
                    matched_ids.update(self.vector_index[word])
            
            # Get documents and calculate similarity
            scored_docs = []
            query_vector = self._create_vector(query)
            
            for doc_id in matched_ids:
                doc = self.documents[doc_id]
                similarity = self._cosine_similarity(query_vector, doc["vector"])
                scored_docs.append((similarity, doc["item"]))
            
            # Sort by similarity and filter by type if specified
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            for similarity, item in scored_docs:
                if memory_type is None or item["type"] == memory_type:
                    results.append({
                        "item": item,
                        "similarity": similarity
                    })
                    if len(results) >= max_results:
                        break
            
            return {
                "status": "success",
                "results": results,
                "query": query,
                "context": context
            }
        
        elif operation == "search_by_topic":
            # Search by topic keywords
            topics = data.get("topics", [])
            memory_type = data.get("memory_type", None)
            max_results = data.get("max_results", 5)
            
            results = []
            for topic in topics:
                for store_key in ["conversations", "knowledge"]:
                    for item in self.memory_store[store_key]:
                        if memory_type is None or item["type"] == memory_type:
                            if any(topic.lower() in str(t).lower() for t in item["topics"]):
                                results.append(item)
                                if len(results) >= max_results:
                                    break
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break
            
            return {
                "status": "success",
                "results": results,
                "topics": topics,
                "context": context
            }
        
        return {
            "status": "error",
            "message": "Unknown operation",
            "context": context
        }


class Coordinator(Agent):
    """Orchestrates worker agents to answer user questions"""
    
    def __init__(self, research_agent: ResearchAgent, analysis_agent: AnalysisAgent, memory_agent: MemoryAgent):
        super().__init__("Coordinator")
        self.research_agent = research_agent
        self.analysis_agent = analysis_agent
        self.memory_agent = memory_agent
        self.conversation_history = []
    
    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process method required by the abstract base class"""
        query = task.get("query", "")
        return self.process_query(query)
    
    def _check_memory_first(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if we have relevant information in memory before processing"""
        print(f"[{self}] Checking memory for relevant information...")
        
        # Search memory for similar queries
        memory_result = self.memory_agent.process({
            "operation": "retrieve",
            "data": {
                "query": query,
                "memory_type": "knowledge",
                "max_results": 3
            }
        })
        
        if memory_result["status"] == "success" and memory_result["results"]:
            # Check if any result has high similarity
            for result in memory_result["results"]:
                if result["similarity"] > 0.7:  # Threshold for considering it relevant
                    print(f"[{self}] Found relevant information in memory")
                    return {
                        "status": "success",
                        "result": f"I found some relevant information from previous conversations: {result['item']['content']}",
                        "source": "memory",
                        "confidence": result['item'].get('confidence', 0.7)
                    }
        
        return None
    
    def _decompose_task(self, query: str) -> List[Dict[str, Any]]:
        """Simple task decomposition based on keywords"""
        query_lower = query.lower()
        tasks = []
        
        # Check if this is a memory query
        if any(word in query_lower for word in ["what did we", "previous", "before", "earlier", "discussed"]):
            tasks.append({
                "agent": "memory",
                "operation": "retrieve",
                "data": {
                    "query": query,
                    "memory_type": "conversation"
                }
            })
            return tasks
        
        # Check if research is needed
        research_keywords = ["find", "research", "search", "look up", "what are", "what is"]
        if any(keyword in query_lower for keyword in research_keywords):
            tasks.append({
                "agent": "research",
                "query": query
            })
        
        # Check if analysis is needed
        analysis_keywords = ["analyze", "compare", "efficiency", "effective", "trade-off", "recommend"]
        if any(keyword in query_lower for keyword in analysis_keywords):
            # Analysis tasks depend on research results
            if tasks and tasks[0]["agent"] == "research":
                tasks.append({
                    "agent": "analysis",
                    "data": {"from_previous": True},
                    "analysis_type": self._determine_analysis_type(query)
                })
            else:
                # Direct analysis without research
                tasks.append({
                    "agent": "analysis",
                    "data": {"direct_query": query},
                    "analysis_type": self._determine_analysis_type(query)
                })
        
        # If no specific tasks identified, default to research
        if not tasks:
            tasks.append({
                "agent": "research",
                "query": query
            })
        
        return tasks
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of analysis needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["compare", "versus", "vs", "difference"]):
            return "comparison"
        elif any(word in query_lower for word in ["efficiency", "effective", "performance"]):
            return "efficiency"
        elif any(word in query_lower for word in ["method", "approach", "methodology"]):
            return "methodologies"
        
        return "general"
    
    def _execute_task_sequence(self, tasks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Execute the sequence of tasks and combine results"""
        context = {"original_query": query, "intermediate_results": {}}
        final_result = None
        
        for i, task in enumerate(tasks):
            agent_type = task["agent"]
            task_data = task.copy()
            del task_data["agent"]
            
            # Add context to task
            task_data["context"] = context
            
            # Execute task
            if agent_type == "research":
                result = self.research_agent.process(task_data)
                context["intermediate_results"]["research"] = result
                
                # Store research results in memory
                if result["status"] == "success" and result["results"]:
                    self.memory_agent.process({
                        "operation": "store",
                        "data": {
                            "type": "knowledge",
                            "content": result["results"],
                            "topics": list(result["results"].keys()),
                            "source": str(self.research_agent),
                            "confidence": result.get("confidence", 0.7)
                        }
                    })
                
                if not final_result:
                    final_result = result
            
            elif agent_type == "analysis":
                # Check if analysis needs data from previous step
                if "from_previous" in task_data.get("data", {}) and "research" in context["intermediate_results"]:
                    task_data["data"] = context["intermediate_results"]["research"]["results"]
                
                result = self.analysis_agent.process(task_data)
                context["intermediate_results"]["analysis"] = result
                final_result = result
                
                # Store analysis results in memory
                if result["status"] == "success" and "analysis" in result:
                    self.memory_agent.process({
                        "operation": "store",
                        "data": {
                            "type": "knowledge",
                            "content": result["analysis"],
                            "topics": ["analysis"],
                            "source": str(self.analysis_agent),
                            "confidence": result.get("confidence", 0.7)
                        }
                    })
            
            elif agent_type == "memory":
                result = self.memory_agent.process(task_data)
                context["intermediate_results"]["memory"] = result
                final_result = result
        
        return final_result or {"status": "error", "message": "No tasks produced results"}
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main method to process a user query"""
        print(f"\n[{self}] Processing query: '{query}'")
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "speaker": "user"
        })
        
        # Store conversation in memory
        self.memory_agent.process({
            "operation": "store",
            "data": {
                "type": "conversation",
                "content": query,
                "topics": self._extract_topics(query),
                "source": "user"
            }
        })
        
        # First check memory to avoid redundant work
        memory_check = self._check_memory_first(query)
        if memory_check:
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "response": memory_check["result"],
                "speaker": "system"
            })
            return memory_check
        
        # Decompose the task
        tasks = self._decompose_task(query)
        print(f"[{self}] Decomposed into {len(tasks)} tasks: {[t['agent'] for t in tasks]}")
        
        # Execute the tasks
        result = self._execute_task_sequence(tasks, query)
        
        # Prepare response
        if result["status"] == "success":
            if "analysis" in result:
                response = result["analysis"]
            elif "results" in result:
                # Handle different result formats
                if isinstance(result["results"], dict):
                    # Format research results (dictionary)
                    response = "Here's what I found:\n"
                    for topic, content in result["results"].items():
                        response += f"\n{topic.title()}:\n"
                        if isinstance(content, dict):
                            for key, value in content.items():
                                response += f"  {key}: {value}\n"
                        else:
                            response += f"  {content}\n"
                elif isinstance(result["results"], list):
                    # Format memory results (list)
                    response = "I found these relevant items from memory:\n"
                    for i, item in enumerate(result["results"], 1):
                        if isinstance(item, dict) and "item" in item:
                            # This is a similarity-based result
                            memory_item = item["item"]
                            response += f"\n{i}. {memory_item.get('content', 'No content')}\n"
                            response += f"   (Source: {memory_item.get('source', 'unknown')}, "
                            response += f"Similarity: {item.get('similarity', 0):.2f})\n"
                        elif isinstance(item, dict):
                            # This is a direct memory item
                            response += f"\n{i}. {item.get('content', 'No content')}\n"
                            response += f"   (Source: {item.get('source', 'unknown')})\n"
                else:
                    response = str(result["results"])
            else:
                response = str(result)
        else:
            response = "I encountered an error while processing your request. Please try again."
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "response": response,
            "speaker": "system"
        })
        
        # Store final response in memory
        self.memory_agent.process({
            "operation": "store",
            "data": {
                "type": "conversation",
                "content": response,
                "topics": self._extract_topics(query),
                "source": str(self)
            }
        })
        
        return {
            "status": "success",
            "response": response,
            "confidence": result.get("confidence", 0.7)
        }
        
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract simple topics from text"""
        words = text.lower().split()
        topics = []
        important_words = ["neural", "network", "transformer", "reinforcement", "learning", 
                          "machine", "optimization", "efficiency", "analysis", "compare"]
        
        for word in words:
            if word in important_words and word not in topics:
                topics.append(word)
        
        return topics
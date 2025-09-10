# Multi-Agent Chat System

A Python implementation of a multi-agent system with a coordinator agent and three specialized worker agents (Research, Analysis, and Memory).

## Architecture

The system consists of four main components:

1. **Coordinator Agent**: Orchestrates the workflow, decomposes tasks, and manages communication between agents
2. **Research Agent**: Simulates information retrieval from a pre-loaded knowledge base
3. **Analysis Agent**: Performs comparisons, reasoning, and simple calculations
4. **Memory Agent**: Manages long-term storage, retrieval, and context updates with vector search capabilities

## How to Run

# Build the Docker image
docker build -t multi-agent-system .

# Run the container
docker run multi-agent-system

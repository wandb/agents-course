# Agents Course
Click here to get the course: [Agents Course](https://wandb.me/agents)
A comprehensive course on building AI agents using Python, OpenAI, and Weave. This course covers various aspects of agent development, from basic workflows to complex multi-agent systems with memory and evaluation capabilities.

## Course Structure & Order

The course is designed to be followed in order, with each module building on the previous one:

1. **Basic Workflow** (`_1_workflow.py`)  
   *Learn how to use OpenAI and Weave for simple prompt-response workflows.*  
   Run with:  
   ```bash
   python _1_workflow.py
   ```

2. **Simple Agent** (`_2_agent.py`)  
   *Implement a minimal agent that can use tools and process user input.*  
   Run with:  
   ```bash
   python _2_agent.py
   ```

3. **Memory & Retrieval** (`_3_memory_retrieval.py`)  
   *Add memory storage and retrieval capabilities to your agent, enabling it to remember and recall information.*  
   Run with:  
   ```bash
   python _3_memory_retrieval.py
   ```

4. **Multi-Agent Systems** (`_4_multi_agents.py`)  
   *Build systems with multiple specialized agents that can hand off tasks to each other.*  
   Run with:  
   ```bash
   python _4_multi_agents.py
   ```

5. **Evaluation** (`_5_evals.py`)  
   *Test and evaluate agent performance using automated evaluation tools and scenarios.*  
   Run with:  
   ```bash
   python _5_evals.py
   ```

6. **Simple Evaluations** (`_5_simple_evals.py`)  
   *Quickly test agent responses with lightweight, script-based evaluation.*  
   Run with:  
   ```bash
   python _5_simple_evals.py
   ```

7. **MCP Integration** (`_6_mcp.py`)  
   *Integrate with the Model Context Protocol (MCP) to allow agents to interact with the filesystem and external tools.*  
   Run with:  
   ```bash
   python _6_mcp.py
   ```

> **Tip:** Follow the modules in order for the best learning experience. Each script can be run independently as shown above.

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended for dependency management)
- OpenAI API key
- Weights & Biases (wandb) API key
- [npx](https://www.npmjs.com/package/npx) (for MCP integration, see `_6_mcp.py`)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/agents-course.git
    cd agents-course
    ```

2. **Create and activate a virtual environment with all dependencies:**
    ```bash
    uv venv .venv
    uv sync
    ```

3. **Set up your environment variables:**
    Create a `.env` file in the project root with:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    WANDB_API_KEY=your_wandb_api_key
    ```

---

For more information or to get the course, visit: [Agents Course](https://wandb.me/agents)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path

import dotenv
from viseval import Dataset, Evaluator
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from llmx import TextGenerator # Assuming this is available and needed for other agents.

# Import the new LangGraph agent
from langgraph_agent_repurposed import DataVisualizationAgent as LangGraphAgent

dotenv.load_dotenv()

def configure_llm(model: str, agent: str):
    """Configures and returns the appropriate LLM based on model and agent type."""
    # This function is from your original file, adapted to use the new LangGraphAgent
    if agent == "lida":
        from llmx import llm
        return llm(provider="openai", api_type="azure", model=model, models={"max_tokens": 4096, "temperature": 0.0})
    else:
        if model == "gemini-pro":
            return ChatGoogleGenerativeAI(
                model=model, temperature=0.0, convert_system_message_to_human=True
            )
        elif model in ["gpt-35-turbo", "gpt-4"]:
            return AzureChatOpenAI(
                model_name=model,
                max_retries=999,
                temperature=0.0,
                request_timeout=20,
            )
        elif model == "codellama-7b":
            from model.langchain_llama import ChatLlama
            return ChatLlama("../llama_models/CodeLlama-7b-Instruct")
        else:
            raise ValueError(f"Unknown model {model}")

def config_agent(agent: str, model: str, config: dict):
    """Configures and returns the agent instance."""
    # This function is also from your original file.
    # We will instantiate your DataVisualizationAgent directly here for evaluation.
    if agent == "langgraph_agent":
        return LangGraphAgent(dataset_summaries_file="dataset_summaries.json")
    else:
        raise ValueError(f"Unknown agent {agent}")

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument(
        "--type", type=str, choices=["all", "single", "multiple"], default="all"
    )
    parser.add_argument("--irrelevant_tables", type=bool, default=False)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-35-turbo",
        choices=["gpt-4", "gpt-35-turbo", "gemini-pro", "codellama-7b"],
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="langgraph_agent", # Changed default to your new agent
        choices=["langgraph_agent"], # Restrict choices to the new agent
    )
    parser.add_argument(
        "--library", type=str, default="matplotlib", choices=["matplotlib", "seaborn"]
    )
    parser.add_argument("--logs", type=Path, default="./logs")
    parser.add_argument("--webdriver", type=Path, default="/usr/bin/chromedriver")

    args = parser.parse_args()

    # config dataset
    dataset = Dataset(args.benchmark, args.type, args.irrelevant_tables)

    # config agent
    agent = config_agent(
        args.agent,
        args.model,
        {"library": args.library},
    )

    vision_model = AzureChatOpenAI(
        model_name="gpt-4-turbo-v",
        max_retries=999,
        temperature=0.0,
        request_timeout=20,
        max_tokens=4096,
    )

    # config evaluator
    evaluator = Evaluator(webdriver_path=args.webdriver, vision_model=vision_model)

    # evaluate agent
    config = {"library": args.library, "logs": args.logs}
    result = evaluator.evaluate(agent, dataset, config)
    score = result.score()
    print(f"Score: {score}")

if __name__ == "__main__":
    _main()
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path
import dotenv
from viseval import Dataset, Evaluator
from langchain_openai import ChatOpenAI
from profilers_agent import DataProfilerAgent
from visualization_agent import DataVisualizationAgent

dotenv.load_dotenv()

def configure_llm(model: str, agent: str):
    if agent == "lida":
        from llmx import llm
        return llm(provider="openai", api_type="azure", model=model, models={"max_tokens": 4096, "temperature": 0.0})
    else:
        if model in ["gpt-35-turbo", "gpt-4"]:
            return ChatOpenAI(
                model=model,
                max_retries=999,
                temperature=0.0,
                request_timeout=20,
            )
        elif model == "codellama-7b":
            from model.langchain_llama import ChatLlama
            return ChatLlama("../llama_models/CodeLlama-7b-Instruct")
        else:
            raise ValueError(f"Unknown model {model}")

def config_agent(agent_type: str, model: str, config: dict):
    if agent_type == "langgraph_agent":
        llm = configure_llm(model, agent_type)
        return DataVisualizationAgent(llm=llm, dataset_summaries_file=config["summary_file_path"], datasets_base_path=config["datasets_base_path"])
    else:
        raise ValueError(f"Unknown agent {agent_type}")

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
        choices=["gpt-4", "gpt-35-turbo", "codellama-7b"],
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="langgraph_agent",
        choices=["langgraph_agent"],
    )
    parser.add_argument(
        "--library", type=str, default="matplotlib", choices=["matplotlib", "seaborn"]
    )
    parser.add_argument("--logs", type=Path, default="./logs")
    parser.add_argument("--webdriver", type=Path, default="/usr/bin/chromedriver")

    args = parser.parse_args()

    # Step 1: Profile the datasets first
    datasets_base_path = Path(args.benchmark) / "databases"
    if not datasets_base_path.exists():
        raise FileNotFoundError(f"Databases directory not found at {datasets_base_path}")
    
    print("---RUNNING DATA PROFILER---")
    profiler = DataProfilerAgent()
    summary_file_path = profiler.run_profiler(datasets_base_path)
    print("---DATA PROFILING COMPLETE---")

    # Step 2: Configure the Data Visualization Agent with the new paths
    agent = config_agent(
        args.agent,
        args.model,
        {"library": args.library, "summary_file_path": summary_file_path, "datasets_base_path": str(datasets_base_path)},
    )
    
    # Step 3: Configure the Evaluator
    vision_model = ChatOpenAI(
        model="gpt-4-vision-preview",
        max_retries=999,
        temperature=0.0,
        request_timeout=20,
        max_tokens=4096,
    )

    evaluator = Evaluator(webdriver_path=args.webdriver, vision_model=vision_model)

    # Step 4: Evaluate the agent on a limited number of queries
    dataset = Dataset(args.benchmark, args.type, args.irrelevant_tables)
    
    # Limit to first 3 queries
    limited_queries = list(dataset.all_queries.keys())[:3]
    dataset.all_queries = {k: dataset.all_queries[k] for k in limited_queries}
    
    config = {"library": args.library, "logs": args.logs}
    result = evaluator.evaluate(agent, dataset, config)
    score = result.score()
    print(f"Score: {score}")

if __name__ == "__main__":
    _main()
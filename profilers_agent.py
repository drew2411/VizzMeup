# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pandas as pd
from dotenv import load_dotenv
import json
from typing import TypedDict, Annotated
import chromadb
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from pathlib import Path

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    count: int
    target: int
    dataset_paths: list[str]
    current_summary: str

class DataProfilerAgent:
    def __init__(self, target_count: int = 2):
        load_dotenv(dotenv_path='.env')
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.dataset_summaries = {}
        self.model = ChatOpenAI(model="gpt-4.1-2025-04-14")
        self.model_with_tools = self.model.bind_tools(self.get_tools())
        self.graph_workflow = StateGraph(State)
        self.graph_workflow.add_node("agent", self.call_model)
        self.graph_workflow.add_node("tools", ToolNode(self.get_tools()))
        self.graph_workflow.add_edge(START, "agent")
        self.graph_workflow.add_conditional_edges("agent", tools_condition)
        self.graph_workflow.add_edge("tools", "agent")
        self.agent = self.graph_workflow.compile()
        self.initial_state = {
            "messages": [],
            "count": 0,
            "target": target_count,
            "dataset_paths": []
        }
        self.datasets_dir = None

    def _get_dataset_path(self, dataset_count: int) -> str:
        if self.datasets_dir:
            dataset_files = os.listdir(self.datasets_dir)
            if 0 <= dataset_count < len(dataset_files):
                return os.path.join(self.datasets_dir, dataset_files[dataset_count])
        return None

    def peek(self, dataset_count: int) -> str:
        dataset_path = self._get_dataset_path(dataset_count)
        if not dataset_path:
            return f"Error: Dataset not found for index {dataset_count}"
        
        try:
            df = pd.read_csv(dataset_path, nrows=50)
            file_stats = os.stat(dataset_path)
            dataset_name = os.path.basename(dataset_path)
            
            total_rows, total_cols = df.shape
            column_summary = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                sample_values = df[col].dropna().head(3).tolist()
                column_summary.append(f"Column: {col}")
                column_summary.append(f"  - Data Type: {col_type}")
                column_summary.append(f"  - Null values in sample: {null_count}")
                column_summary.append(f"  - Unique values in sample: {unique_count}")
                column_summary.append(f"  - Sample values: {sample_values}")
                column_summary.append("")

            summary = f"""DATASET PEEK SUMMARY: {dataset_name}
        BASIC INFO (First 50 rows):
        - Rows in sample: {total_rows}
        - Total columns: {total_cols}
        - File size: {file_stats.st_size:,} bytes

        COLUMN SUMMARY:
        {chr(10).join(column_summary)}

        FIRST 50 ROWS DATA:
        {df.to_string(max_cols=10, max_colwidth=20)}

        This peek shows the structure and sample data from the first 50 rows of the dataset."""
            return summary
        except Exception as e:
            return f"Error reading dataset {dataset_path}: {str(e)}"

    def update(self, summary: str, current_count: int) -> dict:
        """Updates the state with the summary for the current dataset and stores it in ChromaDB."""
        # The rest of your function code remains the same
        self.dataset_summaries[current_count] = summary
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("dataset_summaries")
        dataset_path = self._get_dataset_path(current_count)
        dataset_name = os.path.basename(dataset_path) if dataset_path else f"dataset_{current_count}"
        doc_id = dataset_name
        
        try:
            existing = collection.get(ids=[doc_id])
            if existing['ids']:
                collection.update(
                    ids=[doc_id],
                    documents=[summary],
                    metadatas=[{"dataset_name": dataset_name, "dataset_id": current_count}]
                )
                print(f"Updated existing document: {doc_id}")
            else:
                collection.add(
                    documents=[summary],
                    metadatas=[{"dataset_name": dataset_name, "dataset_id": current_count}],
                    ids=[doc_id]
                )
                print(f"Added new document: {doc_id}")
        except Exception:
            collection.add(
                documents=[summary],
                metadatas=[{"dataset_name": dataset_name, "dataset_id": current_count}],
                ids=[doc_id]
            )
            print(f"Added new document: {doc_id}")
        
        saved_data = collection.get(ids=[doc_id])
        print(f"Verification - Saved document count: {len(saved_data['ids'])}")
        print(f"ChromaDB updated successfully. Database path: {os.path.abspath('./chroma_db')}")

        return {
            "messages": [],
            "count": current_count + 1
        }

    def semantic_summary(self, state):
        """Generates a high-level semantic summary of all profiled datasets.
        
        Args:
            state (dict): The current state of the LangGraph.
        """

        print("---GENERATING SEMANTIC SUMMARY---")
        all_summaries = [v for k, v in self.dataset_summaries.items() if k != 'database_intent']
        prompt = (
            "You are a data documentation expert. Given the following dataset summaries, "
            "write a comprehensive, detailed, and semantic description of what the entire collection of datasets is about. "
            "Describe the overall purpose, domains, relationships, and analytical potential. "
            "Highlight any patterns, strengths, or limitations. Do NOT repeat the summaries verbatim, but synthesize them into a single, coherent, high-level description.\n\n"
            "DATASET SUMMARIES:\n" + "\n---\n".join(all_summaries) + "\n\nDATABASE INTENT SUMMARY:"
        )
        model = ChatOpenAI(model="gpt-3.5-turbo")
        result = model.invoke(prompt)
        summary = result.content
        self.dataset_summaries['database_intent'] = summary
        print("Semantic summary generated and stored.")
        return "done"

    def get_tools(self):
        return [self.peek, self.update, self.semantic_summary]

    def call_model(self, state):
        messages = [
            SystemMessage(content=f"""You are an expert data profiling agent. Your task is to iteratively profile datasets until the target number is reached.
    •   Current dataset number: {state["count"]}
    •   Target number of datasets: {state["target"]}
For each dataset, generate a comprehensive profiling summary containing the following three sections:
1. Dataset Summary and Use Case
    Start with a section called 'METADATA' that includes:
    - Dataset name
    - File size
    - Creation date
    - Last modified date
    •   Provide a clear, high-level verbal summary of what the dataset is about.
    •   Identify the possible analytical tasks or domains where this dataset can be applied (e.g., classification, regression, forecasting, customer analysis, NLP, etc.).
    •   Express the summary in a general, non-technical tone for easy understanding.
2. Column-wise Description and Statistics
For each column in the dataset:
    •   Start with the exact column name.
    •   Provide a semantic description explaining what the column represents. and for which purpose this coloumn can be used that is the most importatn informatino
    •   Provide a statistical summary, curated to the column type:
    •   Numerical columns: count, mean, std, min, max, percentiles, skewness, kurtosis, missing values.
    •   Categorical columns: number of unique values, mode, top value frequencies, missing values.
    •   Datetime columns: min/max dates, frequency distribution, temporal gaps, granularity.
    •   Text columns: average length, top terms, language (if applicable), missing values.
3. Overall Dataset Metadata and Statistical Summary
    •   Include overall metadata: number of rows, columns, column type distribution (numeric, categorical, datetime, text), total missing values, and file size (if available).
    •   Provide a high-level statistical overview: any imbalance, skewed distributions, sparsity, duplicates, or anomalies.
    •   Comment on the dataset’s overall structure, quality, and readiness for machine learning or analytical tasks.
4. data quality

    after creating the summary call the updating step
    1. First, use peek({state["count"]}) to get dataset preview
2. Create a summary based the above data and follow the guidelines which i gave u
3. After creating the summary, call update(summary, {state["count"]}) to store it
4.finally call the semantic summarizer
            """)
        ] + state["messages"]
        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}

    def run_profiler(self, dataset_dir: Path):
        self.datasets_dir = dataset_dir
        self.initial_state["dataset_paths"] = os.listdir(dataset_dir)
        self.initial_state["target"] = len(self.initial_state["dataset_paths"])
        
        # Stream the agent's execution to generate summaries
        for step in self.agent.stream(self.initial_state):
            print("Step:", step)

        # After the streaming is complete, save the final summaries
        summaries_file_path = os.path.join(dataset_dir, "dataset_summaries.json")
        with open(summaries_file_path, "w") as f:
            json.dump(self.dataset_summaries, f, indent=4)
        print(f"Dataset summaries saved to {summaries_file_path}")
        return summaries_file_path
    
if __name__ == "__main__":
    # Example usage:
    # profiler = DataProfilerAgent()
    # profiler.run_profiler(dataset_dir="D:\\Projects\\VizzMeup\\visEval_dataset\\databases")
    pass # No longer run directly to avoid unexpected behavior
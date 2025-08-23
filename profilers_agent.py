import os
import pandas as pd
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import TypedDict, Annotated

# State def of profiler
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import chromadb
from chromadb.config import Settings


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    count: int
    target: int


class DataProfilerAgent:
    def __init__(self, target_count: int = 2):
        # Load environment variables
        load_dotenv(dotenv_path='.env')
        # Set OpenAI API key from environment variable
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        # Initialize global variables for the class instance
        self.dataset_summaries = {}

        # Configure LLMs
        self.model = ChatOpenAI(model="gpt-4.1-2025-04-14")
        self.model_with_tools = self.model.bind_tools(self.get_tools())

        # Initialize Langgraph workflow
        self.graph_workflow = StateGraph(State)
        self.graph_workflow.add_node("agent", self.call_model)
        self.graph_workflow.add_node("tools", ToolNode(self.get_tools()))
        self.graph_workflow.add_edge(START, "agent")
        self.graph_workflow.add_conditional_edges("agent", tools_condition)
        self.graph_workflow.add_edge("tools", "agent")
        self.agent = self.graph_workflow.compile()

        # Set initial state
        self.initial_state = {
            "messages": [],
            "count": 0,
            "target": target_count
        }

    def peek(self, dataset_count: int) -> str:
        """This helps in getting a peek of the dataset helps you get an idea about the dataset initially about the column name and the values in the first 50 rows to get a brief idea about the dataset"""
        try:
            # Read only the first 50 rows of the dataset
            dataset_path = "datasets/" + os.listdir("datasets")[dataset_count - 1]
            file_stats = os.stat(dataset_path)
            dataset_name = os.path.basename(dataset_path)
            df = pd.read_csv("datasets/" + os.listdir("datasets")[dataset_count - 1], nrows=50)

            # Get basic info
            total_rows, total_cols = df.shape

            # Create column summary
            column_summary = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()

                # Get sample values (first 3 non-null values)
                sample_values = df[col].dropna().head(3).tolist()

                column_summary.append(f"Column: {col}")
                column_summary.append(f"  - Data Type: {col_type}")
                column_summary.append(f"  - Null values in sample: {null_count}")
                column_summary.append(f"  - Unique values in sample: {unique_count}")
                column_summary.append(f"  - Sample values: {sample_values}")
                column_summary.append("")

            # Create the summary output
            summary = f"""DATASET PEEK SUMMARY: {"datasets/" + os.listdir("datasets")[dataset_count - 1]}
        BASIC INFO (First 50 rows):
        - Rows in sample: {total_rows}
        - Total columns: {total_cols}
        - File size: {os.path.getsize("datasets/" + os.listdir("datasets")[dataset_count - 1]):,} bytes

        COLUMN SUMMARY:
        {chr(10).join(column_summary)}

        FIRST 50 ROWS DATA:
        {df.to_string(max_cols=10, max_colwidth=20)}

        This peek shows the structure and sample data from the first 50 rows of the dataset."""
            return summary

        except Exception as e:
            return f"Error reading dataset {dataset_path}: {str(e)}"

    def update(self, summary: str, current_count: int) -> dict:
        """Updates the state with the summary for the current dataset and increments count"""
        try:
            # Store in class instance dict
            self.dataset_summaries[current_count] = summary

            # Create ChromaDB client with proper configuration
            client = chromadb.PersistentClient(path="./chroma_db")

            # Get or create collection
            collection = client.get_or_create_collection("dataset_summaries")

            # Check if datasets directory exists and has files
            datasets_dir = "datasets"
            if not os.path.exists(datasets_dir):
                print(f"Warning: {datasets_dir} directory not found")
                dataset_name = f"dataset_{current_count}"
            else:
                dataset_files = os.listdir(datasets_dir)
                if current_count < len(dataset_files):
                    dataset_name = dataset_files[current_count]
                else:
                    print(f"Warning: current_count {current_count} exceeds available datasets")
                    dataset_name = f"dataset_{current_count}"

            # Check if document already exists
            doc_id = dataset_name
            try:
                existing = collection.get(ids=[doc_id])
                if existing['ids']:
                    # Update existing document
                    collection.update(
                        ids=[doc_id],
                        documents=[summary],
                        metadatas=[{"dataset_name": dataset_name, "dataset_id": current_count}]
                    )
                    print(f"Updated existing document: {doc_id}")
                else:
                    # Add new document
                    collection.add(
                        documents=[summary],
                        metadatas=[{"dataset_name": dataset_name, "dataset_id": current_count}],
                        ids=[doc_id]
                    )
                    print(f"Added new document: {doc_id}")
            except Exception as e:
                # If get fails, assume document doesn't exist and add it
                collection.add(
                    documents=[summary],
                    metadatas=[{"dataset_name": dataset_name, "dataset_id": current_count}],
                    ids=[doc_id]
                )
                print(f"Added new document: {doc_id}")

            # Verify the data was saved
            saved_data = collection.get(ids=[doc_id])
            print(f"Verification - Saved document count: {len(saved_data['ids'])}")
            print(f"ChromaDB updated successfully. Database path: {os.path.abspath('./chroma_db')}")

        except Exception as e:
            print(f"Error updating ChromaDB: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()

        return {
            "message": [],
            "count": current_count + 1
        }

    def semantic_summary(self, state):
        "i am semantic summary"
        all_summaries = [v for k, v in self.dataset_summaries.items() if k != 'database_intent']
        prompt = (
            "You are a data documentation expert. Given the following dataset summaries, "
            "write a comprehensive, detailed, and semantic description of what the entire collection of datasets is about. "
            "Describe the overall purpose, domains, relationships, and analytical potential. "
            "Highlight any patterns, strengths, or limitations. Do NOT repeat the summaries verbatim, but synthesize them into a single, coherent, high-level description.\n\n"
            "DATASET SUMMARIES:\n" + "\n---\n".join(all_summaries) + "\n\nDATABASE INTENT SUMMARY:"
        )
        Model = ChatOpenAI(model="gpt-3.5-turbo")
        result = Model.invoke(prompt)
        summary = result.content
        self.dataset_summaries['database_intent'] = summary
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

    def run_profiler(self):
        # Synchronous streaming
        for step in self.agent.stream(self.initial_state):
            print("Step:", step)

        # Save dataset_summaries to a file
        with open("dataset_summaries.json", "w") as f:
            json.dump(self.dataset_summaries, f)


if __name__ == "__main__":
    profiler = DataProfilerAgent(target_count=2)
    profiler.run_profiler()
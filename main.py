from llama_index.core.tools import FunctionTool
# from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent, FunctionAgent
from llama_index.core import SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex
# import asyncio
from tools import *
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# qdrant_client = QdrantClient(url="http://localhost:6333")

# Initialize the tool
load_study_tool = FunctionTool.from_defaults(load_study)

get_data_quality_tool = FunctionTool.from_defaults(get_data_quality)

predict_sleep_stage_tool = FunctionTool.from_defaults(predict_sleep_stage)

compute_sleep_metrics_tool = FunctionTool.from_defaults(compute_sleep_metrics)

load_metrics_tool = FunctionTool.from_defaults(
    load_metrics
)

db = chromadb.PersistentClient(path="./knowledge_chroma_db")
chroma_collection = db.get_collection("knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = Ollama(
    # model="llama3.2:3b",
    model="qwen3:8b",
    request_timeout=600
)

query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

retrieval_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="retrieval_tool",
    description="Provides sleep science references."
)

# llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

data_quality_agent = ReActAgent(
    name="data_quality_agent",
    description="Is able to load ppg and acc signal from study and / or evaluate signal quality of data",
    system_prompt="""
    You are a Data Quality Analysis Agent.

    Your responsibilities:

    1. Load study data when a study ID is provided.
    2. Evaluate data quality metrics for the study.
    3. Report data completeness and wear-time quality.

    Available metrics may include:

    - accelerometer_data_availability_percent:
      Percentage of recording duration with valid accelerometer data.

    - ppg_data_availability_percent:
      Percentage of recording duration with valid accelerometer data.

    - device_worn_ppg_percent:
      Percentage of recording duration where the device was worn and valid physiological PPG signals were detected.

    Instructions:

    - Use tools whenever signal quality information is requested.
    - If a study has not been loaded yet, load it first.
    - Summarize signal quality in plain language.
    - Do not perform sleep interpretation.
    - Do not diagnose medical conditions.
    - Focus only on data quality and recording quality.

    You cannot:

    - Interpret sleep quality because you only have data quality
    - Compute sleep metrics
    - Diagnose diseases

    If sleep metrics are requested,
    handoff to the appropriate agent.
    """,
    tools=[load_study_tool, get_data_quality_tool],
    llm=llm,
    streaming=False
)

sleep_stage_agent = ReActAgent(
    name="sleep_stage_agent",
    description="Is able to load ppg and acc signal from study and predict sleep stage",
    system_prompt="A helpful assistant that can use a tool to load ppg and acc signal from study and predict sleep stage.",
    tools=[load_study_tool, predict_sleep_stage_tool],
    llm=llm,
    streaming=False
)

sleep_metrics_agent = ReActAgent(
    name="sleep_metrics_agent",
    description="Is able to compute sleep metrics from hypnogram file of study",
    system_prompt="You are a helpful assistant that can use a tool to compute sleep metrics from hypnogram file of study. Your task is to return all the computed sleep metrics. After computing metrics: "
                  "DO NOT answer the user. DO NOT generate interpretations. Immediately handoff to interpretation_agent. "
                  "Your job ends after metrics generation.",
    tools=[compute_sleep_metrics_tool],
    llm=llm,
    streaming=False,
    can_handoff_to=[
        "sleep_stage_agent",
        "interpretation_agent"
    ],
)

interpretation_agent = ReActAgent(
    name="interpretation_agent",

    description="""
    Interprets sleep metrics and generates sleep reports.
    """,

    llm=llm,

    tools=[
        load_metrics_tool
    ],

    can_handoff_to=[
        "sleep_metrics_agent",
        "knowledge_agent"
    ],

    # max_iterations=8,

    system_prompt="""
You are a Sleep Interpretation Expert.

Your job is to interpret sleep studies.

Workflow:

Step 1:
Load metrics using load_metrics.
if there is no metrics file yet, handoff to sleep_metrics_agent.

Step 2:
Determine whether scientific evidence is needed.

If evidence is needed:
handoff to knowledge_agent.

Step 3:
Use returned evidence.

Step 4:
Generate the final report.

Important:

You MUST NOT answer before loading metrics.

You MUST use knowledge_agent whenever:

- sleep quality is requested
- recommendations are requested
- explanations are requested
- sleep science references are requested

You are responsible for:

- sleep quality summary
- sleep insights
- recommendations

You are NOT responsible for:

- retrieving documents
- querying vector database

knowledge_agent handles retrieval.

Output format:

Final Answer:

## Summary

...

## Key Findings

...

## Recommendations

...
"""
)

# interpretation_agent = ReActAgent(
#     name="interpretation_agent",
#     description="Interpret sleep metrics or sleep quality of study and generate sleep insights",
#     llm=llm,
#     tools=[load_metrics_tool, retrieval_tool],
#     # tools=[interpret_sleep_report_tool],
#     max_iterations=3,
#     system_prompt="""
# You are a sleep interpretation expert.
#
# You receive sleep metrics generated from wearable sensors. Your job is to interpret these metrics.
#
# Your tasks:
# 1. Load metrics file using load_metrics_tool
# 2. Summarize sleep quality.
# 3. Retrieve from knowledge base about sleep quality using retrieval_tool
# 4. Explain important metrics.
# 5. Highlight unusual findings.
#
# Rules:
# - You must call retrieval_tool before answering final interpretation.
# - Do not diagnose diseases.
# - Do not claim medical conditions.
# - Only provide sleep insights.
# - Explain metrics in simple language.
# - Mention both strengths and weaknesses.
# - Use cautious language.
# - Mention uncertainty when appropriate.
#
# Example:
#
# Sleep efficiency below 80% may indicate fragmented sleep.
#
# Elevated sleep latency In sleep medicine, it is a crucial metric for evaluating overall sleep quality and diagnosing sleep disorders, helping to indicate whether you are getting sufficient, restorative rest.
#
# Low REM percentage may indicate reduced restorative sleep.
#
# IMPORTANT:
# Before interpreting any sleep metric, you MUST query by retrieval_tool to retrieve supporting sleep science evidence.
#
# Do not answer directly from your own knowledge.
#
# Always retrieve relevant references first.
#
# After interpreting the metrics,
# and when you have enough information:
#
# Respond ONLY in this format:
#
# Final Answer:
#
# ## Summary
#
# ...
#
# ## Key Findings
#
# ...
#
# ## Recommendations
#
# ...
#
# Do not output Thought.
# Do not output Action.
# Do not output Action Input.
#
# """
# )

knowledge_agent = ReActAgent(
    name="knowledge_agent",

    description="""
    Retrieves sleep science knowledge from the knowledge base.
    """,

    tools=[
        retrieval_tool
    ],

    llm=llm,

    system_prompt="""
You are a retrieval-only agent.

RULES:

You MUST call retrieval_tool before answering.

You are NOT allowed to answer from your own knowledge.

You are NOT allowed to infer normal ranges.

You are NOT allowed to interpret metrics.

Workflow:

1. Call retrieval_tool.
2. Read retrieved content.
3. Return retrieved content.

If you have not called  yet:
DO NOT answer.

Output format:

EVIDENCE:
...
"""
)

coordinator_agent = ReActAgent(
    name="coordinator_agent",

    description="""
    Routes user requests to the most appropriate sleep analysis agent.
    """,

    llm=llm,

    can_handoff_to=[
        "data_quality_agent",
        "sleep_stage_agent",
        "sleep_metrics_agent",
        "interpretation_agent",
        "knowledge_agent",
    ],

    system_prompt="""
You are the coordinator of a sleep analysis system.

Your job is to determine which specialized agent should handle the user's request.

IMPORTANT DEFINITIONS

DATA QUALITY:
- sensor quality
- PPG quality
- ACC quality
- wear time
- signal coverage
- missing data
- recording quality
- whether the device was worn correctly

SLEEP QUALITY:
- sleep efficiency
- REM sleep
- NREM sleep
- WASO
- sleep latency
- awakenings
- hypnogram interpretation
- overall sleep assessment

Available agents:

1. data_quality_agent
   - Handles DATA QUALITY only.
   - loads study to get acc and ppg data if the study wasn't processed yet
   - evaluates signal quality not sleep quality
   - evaluates wear time
   - evaluates data coverage

2. sleep_stage_agent
   - loads study to get acc and ppg data if the study wasn't processed yet
   - predicts sleep stages

3. sleep_metrics_agent
   - computes sleep metrics from hypnogram

4. interpretation_agent
   - load sleep quality or sleep metrics using load_metrics_tool.
   - Use retrieved evidence in your report.

5. knowledge_agent
   - retrieves scientific evidence from sleep_knowledge_base

Always hand off to the most appropriate agent.
Do not perform analysis yourself.

Example:
User: Interpret sleep quality
→ interpretation_agent

User: Explain sleep quality
→ interpretation_agent

User: Summarize sleep quality
→ interpretation_agent

User: Evaluate signal quality
→ data_quality_agent

User: Evaluate PPG quality
→ data_quality_agent

User: Evaluate sensor quality
→ data_quality_agent
"""
)


# Create the workflow
# workflow = AgentWorkflow(
#     agents=[coordinator_agent, data_quality_agent, sleep_stage_agent, sleep_metrics_agent, interpretation_agent, knowledge_agent],
#     root_agent="coordinator_agent",
#     verbose=True,
# )

def create_workflow():
    return AgentWorkflow(
        agents=[
            coordinator_agent,
            data_quality_agent,
            sleep_stage_agent,
            sleep_metrics_agent,
            interpretation_agent,
            knowledge_agent
        ],
        root_agent="coordinator_agent",
        verbose=True,
    )

# Run the system
# async def run_multi_agent(query):
#     response = await workflow.run(query)
#     print("Healthcare Agent Response:")
#     print(response)
#
# asyncio.run(run_multi_agent("Interpret the sleep quality of study 2026-05-20-BaoLuu-4mm-10mA"))
import streamlit as st
import asyncio
from main import workflow
import pandas as pd
import plotly.graph_objects as go
import glob
import os
import json
import numpy as np

from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult
)
from cache import analysis_cache
from sleep_core.define import TEMP_FOLDER


if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

if "current_study" not in st.session_state:
    st.session_state.current_study = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = {}

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

st.set_page_config(
    page_title="Sleep Analysis Multi-Agent",
    layout="wide"
)

st.title("🛌 Sleep Analysis Multi-Agent Copilot")

with st.sidebar:

    st.header("Studies")


    studies = [os.path.basename(path) for path in glob.glob("/mnt/DataSet2/Projects/POC_Sleep_study/hsat_data/sleep_study/sleep_test/*")]

    selected_study = st.selectbox(
        "Select Study",
        studies
    )

    # Detect study change
    if (
            st.session_state.current_study is not None
            and st.session_state.current_study != selected_study
    ):
        st.session_state.messages = []
        st.session_state.agent_logs = []
        st.session_state.analysis_result = {}

    # Update current study
    st.session_state.current_study = selected_study

chat_col, trace_col = st.columns(
    [2, 1]
)

with chat_col:

    st.subheader("Sleep Copilot")

    chat_container = st.container(
        height=500
    )

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(
                    msg["role"]
            ):
                st.markdown(
                    msg["content"]
                )

with trace_col:

    st.subheader(
        "🤖 Agent Trace"
    )

    trace_container = st.container(
        height=500
    )

    with trace_container:

        for log in st.session_state.agent_logs:

            st.markdown(log)

st.divider()

st.subheader(
    "📊 Visualization Panel"
)

tab1, tab2, tab3 = st.tabs(
    [
        "Hypnogram",
        "Metrics",
        "Signal Quality"
    ]
)

with tab1:
    result = st.session_state.analysis_result
    if os.path.exists(os.path.join(TEMP_FOLDER, f"{selected_study}_hypnogram.parquet")):
    # if "hypnogram_file" in result:
        hyp = pd.read_parquet(
            os.path.join(TEMP_FOLDER, f"{selected_study}_hypnogram.parquet")
        )["sleep_stage"]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(hyp)),
                y=hyp,
                mode="lines",
                line_shape="hv",
                name="Sleep Stage"
            )
        )

        fig.update_yaxes(
            autorange="reversed",
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=["Wake", "REM", "NREM"]
        )

        fig.update_layout(
            title="Hypnogram",
            xaxis_title="Time (sec)",
            yaxis_title="Stage",
            height=400
        )

        st.plotly_chart(
            fig,
            # width=True
        )

with tab2:
    result = st.session_state.analysis_result
    if os.path.exists(os.path.join(TEMP_FOLDER, f"{selected_study}_metrics.json")):
    # if "metrics" in result:
        with open(os.path.join(TEMP_FOLDER, f"{selected_study}_metrics.json"), 'r') as file:
            metrics = json.load(file)
        # metrics = result["metrics"]

        metric_config = [
            ("Total Sleep Time", "total_sleep_time", " minutes"),
            ("Sleep Efficiency", "sleep_efficiency", "%"),
            ("REM %", "rem_percent", "%"),
            ("NREM %", "nrem_percent", "%"),
            ("Wake %", "wake_percent", "%"),
            ("REM/NREM Ratio", "rem_nrem_ratio", "%"),
            ("Sleep Latency", "sleep_onset_latency_in_minutes", " minutes"),
        ]

        cols_per_row = 4

        for i in range(0, len(metric_config), cols_per_row):

            row_metrics = metric_config[i:i + cols_per_row]

            cols = st.columns(cols_per_row)

            for col_idx, (label, key, unit) in enumerate(row_metrics):
                value = metrics.get(key, "N/A")

                cols[col_idx].metric(
                    label,
                    f"{value}{unit}"
                )

with tab3:
    result = st.session_state.analysis_result


    if "signal_quality" in result:
        sq = result["signal_quality"]

        sq_config = [
            ("Total ACC Duration", "total_duration_acc", "seconds"),
            ("Total PPG Duration", "total_duration_ppg", "seconds"),
            ("ACC Data Availability Percent", "accelerometer_data_availability_percent", "%"),
            ("PPG Data Availability Percent", "ppg_data_availability_percent", "%"),
            ("Device Worn PPG Percent", "device_worn_ppg_percent", "%"),

        ]

        cols_per_row = 3

        for i in range(0, len(sq_config), cols_per_row):

            row_metrics = sq_config[i:i + cols_per_row]

            cols = st.columns(cols_per_row)

            for col_idx, (label, key, unit) in enumerate(row_metrics):
                value = sq.get(key, "N/A")

                cols[col_idx].metric(
                    label,
                    f"{value}{unit}"
                )



user_prompt = st.chat_input(
    "Ask anything about this sleep study..."
)

query = f"""
        Current Study:
        
        {selected_study}
        
        User Question:
        
        {user_prompt}
        """

# async def run_query(query):
#
#     handler = workflow.run(query)
#
#     logs = []
#
#     async for event in handler.stream_events():
#
#         logs.append(str(event))
#
#         # DEBUG
#         if isinstance(event, AgentOutput):
#
#             print("\n" + "=" * 80)
#             print("AGENT OUTPUT")
#             print("=" * 80)
#
#             for block in event.response.blocks:
#                 print(type(block))
#                 print(block)
#
#         elif isinstance(event, ToolCall):
#
#             print("\nTOOL CALL")
#             print(event.tool_name)
#             print(event.tool_kwargs)
#
#         elif isinstance(event, ToolCallResult):
#
#             print("\nTOOL RESULT")
#             print(event.tool_name)
#             print(event.tool_output)
#
#     response = await handler
#
#     return response, logs

async def run_query(query):

    handler = workflow.run(query)

    trace = []

    async for event in handler.stream_events():

        event_name = event.__class__.__name__

        if event_name == "AgentSetup":
            trace.append(
                f"🤖 Agent: {event.current_agent_name}"
            )

        elif event_name == "ToolCall":
            trace.append(
                f"🔧 Tool: {event.tool_name}"
            )

        elif event_name == "ToolCallResult":
            trace.append(
                f"✅ Finished: {event.tool_name}"
            )

    response = await handler

    return response, trace

if user_prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    response, logs = asyncio.run(
        run_query(query)
    )

    st.session_state.analysis_result = response
    st.session_state.agent_logs = logs
    print("\n" + "=" * 80)
    print("Selected study: ", selected_study)
    if selected_study in analysis_cache:
        print("\n" + "=" * 80)
        print(analysis_cache[selected_study])
        st.session_state.analysis_result = analysis_cache[selected_study]
    else:
        print("\n" + "=" * 80)
        print("There is no selected study.")
        print("analysis_cache: ", analysis_cache)
        st.session_state.analysis_result = {}

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": str(response)
        }
    )

    st.rerun()

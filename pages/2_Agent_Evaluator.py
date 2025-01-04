import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
from llm_in_production.openai_utils import get_openai_client
from io import StringIO
import hashlib
from app import get_overall_conversation_score_and_tips, evaluate_conversation, render_rule_evaluation

CLIENT = get_openai_client()


def load_rules(file_name):
    with open(Path("evaluation_rules") / Path(file_name), "rt") as f:
        evaluation_rules = yaml.safe_load(f)
        st.session_state["evaluation_rules"] = evaluation_rules


if (
    "rule_file_name" not in st.session_state
    or st.session_state["rule_file_name"] is None
):
    st.session_state["rule_file_name"] = "evaluation_rules_en.yaml"

if (
    "evaluation_rules" not in st.session_state
    or st.session_state["evaluation_rules"] is None
):
    load_rules(st.session_state["rule_file_name"])

if "conversation" not in st.session_state:
    st.session_state["conversation"] = ""

if "agent_chat_data" not in st.session_state:
    st.session_state["agent_chats"] = []


with st.sidebar:

    st.page_link("app.py", label="Conversation Evaluator")
    st.page_link("pages/1_Quality_Rule_Settings.py", label="Quality Rule Settings")
    st.page_link("pages/2_Agent_Evaluator.py", label="Agent Evaluator")
    st.title("Settings")
    # Pressing this button will trigger the on_click_generate function defined above
    # This will generate a description and extract the features

    evaluation_rule_files = [
        file.name
        for file in Path("evaluation_rules/").iterdir()
        if file.name.endswith(".yaml")
    ]
    selected_rules_to_load = st.selectbox(
        label="Select the rule set to load",
        options=evaluation_rule_files,
        index=evaluation_rule_files.index(st.session_state["rule_file_name"]),
        key="agent_evaluator_rules_select_box"
    )
    load_rules_button = st.button("Load Rules", icon=":material/refresh:")
    if load_rules_button:
        load_rules(selected_rules_to_load)

    options = st.multiselect(
        "Select the rules that you want to evaluate",
        options=[rule["rule_name"] for rule in st.session_state["evaluation_rules"]],
        default=[rule["rule_name"] for rule in st.session_state["evaluation_rules"]],
    )


st.title("Agent Evaluator")
is_loaded = lambda: (
    False
    if "agent_chat_files" not in st.session_state
    or len(st.session_state["agent_chat_files"]) == 0
    else True
)
with st.expander(
    "Loaded chat file(s)" if is_loaded() else "Load chat file(s)",
    expanded=not is_loaded(),
    icon=":material/upload_file:",
):
    uploaded_files = st.file_uploader(
        "Choose file(s) contains conversations.",
        accept_multiple_files=True,
        type=["csv", "md", "txt"],
        help="Each file should contain one chat session.",
    )
    st.session_state["agent_chat_files"] = uploaded_files

evaluation_complete = False
if is_loaded():
    st.success('Chat files are successfully loaded!', icon="✅")
    progress = 0
    progress_bar = st.progress(progress, text=f"Evaluating the conversations... Please wait. ")
    for uploaded_file in st.session_state["agent_chat_files"]:
        chat_conversation = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

        evaluation_results = evaluate_conversation(conversation=chat_conversation, rules=st.session_state["evaluation_rules"])
        overall_conversation_score, tips, topic = get_overall_conversation_score_and_tips(conversation=chat_conversation, 
                                                                     evaluation_results=evaluation_results)

        st.session_state["agent_chats"].append(
            {
                "id": hashlib.sha1(str(chat_conversation).encode("utf-8")).hexdigest()[:8],
                "topic": topic,
                "overall_conversation_score": overall_conversation_score,
                "conversation": chat_conversation,
                "tips": tips,
                "evaluation_results": evaluation_results,
                "filename": uploaded_file.name,
            }
        )

        progress += 100//len(st.session_state["agent_chat_files"])
        progress_bar.progress(progress, text=f"Evaluating the conversations... Please wait. ")
        evaluation_complete = True

    st.dataframe(pd.DataFrame(st.session_state["agent_chats"]), hide_index=True)
else:
    st.write("No chat files to evaluate yet.")

if evaluation_complete:
    st.subheader("Evaluation Results", divider=True)
    average_conversation_score = pd.DataFrame(st.session_state["agent_chats"])["overall_conversation_score"].mean().round(0).astype(int)
    evaluation_results = pd.DataFrame(st.session_state["agent_chats"])["evaluation_results"].values
    failed_rules = []
    for evaluation_result in evaluation_results:
        failed_rules.extend([item for item in evaluation_result if item["evaluation_result"] != "successful"])

    a, b, c = st.columns(3)
    a.metric("🏆 Overall Score", f"{average_conversation_score}%", border=False)
    b.metric("❌ Failed Rules", f"{len(failed_rules)}", border=False)
    c.markdown("❌ Failed Rules (Details)")
    s = ""
    for rule_name, count in pd.DataFrame(failed_rules)["rule_name"].value_counts().to_dict().items():
        s+=f"- {rule_name}: {count}\n"
    c.markdown(s)

    a, b = st.columns(2)
    a.subheader("Improvement Recommendation", divider=True)
    for tip in pd.DataFrame(st.session_state["agent_chats"])["tips"].values:
        a.write(tip)
    
    with b:
        st.subheader("Failed Rules", divider=True)
        for failed_rule in failed_rules:
            render_rule_evaluation(
                rule_name=failed_rule["rule_name"], 
                rule_description=failed_rule["rule_description"], 
                level=failed_rule["level"], 
                evaluation_result=failed_rule["evaluation_result"], 
                thoughts=failed_rule["thoughts"], 
            )



else:
    st.write("Waiting for evaluation...")
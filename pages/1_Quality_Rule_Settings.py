import streamlit as st
import pandas as pd
import yaml
from pathlib import Path

if "rule_file_name" not in st.session_state or st.session_state["rule_file_name"] is None:
    st.session_state["rule_file_name"] = "evaluation_rules_en.yaml"

st.set_page_config(page_title="Quality Rule Settings")
st.title("Quality Rule Settings")  

with st.sidebar:
    st.page_link('app.py', label='Conversation Evaluator')
    st.page_link('pages/1_Quality_Rule_Settings.py', label='Quality Rule Settings')

def load_rules(path="evaluation_rules/evaluation_rules_en.yaml"):
    with open(path, "rt") as f:
        evaluation_rules = yaml.safe_load(f)
        st.session_state["evaluation_rules"] = evaluation_rules

col1, col2 = st.columns(2, )
with col1:
    st.page_link('app.py', label='Back', icon=":material/arrow_back:")
    load_default = st.button('Import from CSV', icon=":material/upload_file:")
with col2:
    evaluation_rule_files = [file.name for file in Path("evaluation_rules/").iterdir() if file.name.endswith(".yaml")]
    selected_rules_to_load = st.selectbox(label="Select the rule set to load", 
                                options=evaluation_rule_files, 
                                index=evaluation_rule_files.index(st.session_state["rule_file_name"]))
    load_default = st.button('Load', icon=":material/download:")

if load_default:
    load_rules(path=Path("evaluation_rules/") / Path(selected_rules_to_load))

if "evaluation_rules" not in st.session_state or st.session_state["evaluation_rules"] is None:
    load_rules(path=Path("evaluation_rules/") / Path(selected_rules_to_load))


edited_rules = st.data_editor(data=pd.DataFrame(st.session_state["evaluation_rules"]),
                                                        column_order=("rule_name", "rule_description", "level"),
                                                        num_rows="dynamic"
                                                        ).to_dict(orient="records")



file_name = st.text_input(label="File Name", value="evaluation_rules_session.yaml")
col1, col2 = st.columns(2)
with col1:
    save = st.button('Save', icon=":material/check:")
with col2:
    download = st.button('Download as CSV', icon=":material/file_export:")


if save:
    
    st.session_state["evaluation_rules"] = edited_rules
    with open(Path("evaluation_rules/") / Path(file_name), "+wt") as f:
        evaluation_rules = yaml.dump(st.session_state["evaluation_rules"], f)
    st.session_state["rule_file_name"] = file_name

    st.success('Custom evaluation rules saved.', icon="✅")
import json
import os
import pandas as pd
from typing import Any
import datetime
import pydantic
import yaml
import streamlit as st
from llm_in_production.openai_utils import get_openai_client
from llm_in_production.text_extraction import (
    BooleanFeature,
    DigitFeature,
    StringFeature,
)
from pathlib import Path

CLIENT = get_openai_client()

def load_rules(file_name):
    with open(Path("evaluation_rules") / Path(file_name), "rt") as f:
        evaluation_rules = yaml.safe_load(f)
        st.session_state["evaluation_rules"] = evaluation_rules


if "rule_file_name" not in st.session_state or st.session_state["rule_file_name"] is None:
    st.session_state["rule_file_name"] = "evaluation_rules_en.yaml"

if "evaluation_rules" not in st.session_state or st.session_state["evaluation_rules"] is None:
    load_rules(st.session_state["rule_file_name"])

if "conversation" not in st.session_state:
    st.session_state["conversation"] = ""

####################
# Logic for the UI #
####################



def generate_example_chat_conversation(new_conversation=False) -> str:
    """
    Generate a text description of a house based on the data provided by the user.
    :param data: A dictionary where the keys are the feature names and the values are the feature values
    :return: A text description of the house
    """
    messages = [
        {"role": "system", 
        "content": f"""
Please create an example of chat conversation between an mobile subscription company's customer service agents and a customer who is late on his payment. 
Customer wants to agree on a payment arrangement but wants a lower monthly payment because he had lost his job recently. 
Agent need to varify few things regarding the customer income and expenses but eventually they agree on a payment arrangement.
The agent is a new joiner so sometimes he makes mistakes and does not use a professional tone.
You can give example names and amount while creating the conversation. Please also add a timestamps for each message.
Please generate the conversation in {st.session_state["language_option"]}.
Write customer messages after **Customer (timestamp):** and agent messages after **Agent (timestamp):**.
"""},
        # {"role": "user", "content": json.dumps(data)},
    ]

    response = CLIENT.chat.completions.create(
        model=os.environ["GPT_4_MODEL_NAME"],
        messages=messages,
        temperature=1.0,
    )

    message = response.choices[0].message.content
    with open(f"example_conversations/example_conversation_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt", "+wt") as f:
        f.write(message)
        

    return message




def on_click_generate_conversation():
    """
    This function is called when the user clicks the "Generate" button.
    It generates a description and extracts the features from it for validation.
    :param data: This is dict with the user input values. E.g. `{"rent": 1000, "city": "Amsterdam", ...}`.
    :return: None
    """

    conversation = generate_example_chat_conversation()

    st.session_state["conversation"] = conversation

def on_click_evaluate_conversation():
    """
    This function is called when the user clicks the "Generate" button.
    It generates a description and extracts the features from it for validation.
    :param data: This is dict with the user input values. E.g. `{"rent": 1000, "city": "Amsterdam", ...}`.
    :return: None
    """


    evaluation_results = evaluate_conversation(conversation=st.session_state["conversation"], rules=st.session_state["evaluation_rules"])
    st.session_state["evaluation_results"] = evaluation_results



######################
# Start Streamlit UI #
######################

title = "Customer Service Quality Assessment"
st.set_page_config(
    page_title=title,
    # page_icon="👋",
    layout="wide",

)
st.title(title)        


# This is the sidebar where we collect the user input
with st.sidebar:
    
    st.page_link('app.py', label='Conversation Evaluator')
    st.page_link('pages/1_Quality_Rule_Settings.py', label='Quality Rule Settings')
    st.page_link('pages/2_Agent_Evaluator.py', label='Agent Evaluator')
    st.title("Settings")
    st.session_state["language_option"] = st.selectbox(
        "Language",
        ("English", "Dutch"),
    )
    # Pressing this button will trigger the on_click_generate function defined above
    # This will generate a description and extract the features

    evaluation_rule_files = [file.name for file in Path("evaluation_rules/").iterdir() if file.name.endswith(".yaml")]
    selected_rules_to_load = st.selectbox(label="Select the rule set to load", 
                                options=evaluation_rule_files, 
                                index=evaluation_rule_files.index(st.session_state["rule_file_name"]),
                                )
    load_rules_button = st.button('Load Rules', icon=":material/refresh:")
    if load_rules_button:
        load_rules(selected_rules_to_load)
    
    
    options = st.multiselect(
    "Select the rules that you want to evaluate",
    options=[rule["rule_name"] for rule in st.session_state["evaluation_rules"]],
    default=[rule["rule_name"] for rule in st.session_state["evaluation_rules"]],
    )


def render_rule_evaluation(
    rule_name: str, 
    rule_description: str, 
    level: list[str] | None, 
    evaluation_result: str| None, 
    thoughts: str| None, 
    count: int | None = None,
):


    if evaluation_result == "successful":
        st.write(f"#### {rule_name}")
        st.write(rule_description)
        st.write(f"✅ Passed: {thoughts}")
    elif rule_description == "not_successful":
        st.write(f"#### {rule_name}")
        st.write(rule_description)
        if count:
            st.write(f"❌ Failed {count} time(s) (Rule Severity: {level}): {thoughts}")
        else:
            st.write(f"❌ Failed (Rule Severity: {level}): {thoughts}")
    else:
        st.write(f"#### {rule_name}")
        st.write(rule_description)
        if count:
            st.write(f"❌ Failed {count} time(s) (Rule Severity: {level}): {thoughts}")
        else:
            st.write(f"❌ Failed (Rule Severity: {level}): {thoughts}")
    st.write()



class EvaluationDataFormat(pydantic.BaseModel):

    rule_name: str = pydantic.Field(
        description=f"The name of the quality rule."
    )
    rule_description: str  = pydantic.Field(
        description=f"The description of the quality rule."
    )
    level: str = pydantic.Field(
        description=f"The importance level of the quality rule. Must be one of the following: High, Medium, Low"
    )
    evaluation_result: str = pydantic.Field(
        description=f"The result of evaluation of the quality rule. Must be one of the following: successful, not_successful, unknown"
    )
    thoughts: str = pydantic.Field(
        description=f"What was the reason to come up with this evaluation result."
    )

class EvaluationDataFormatList(pydantic.BaseModel):
    evaluation_results: list[EvaluationDataFormat]  = pydantic.Field(
        description=f"List of evaluation results for each rule"
    )

def question_about_conversation(question:str, conversation:str):
    """
    Generate a text description of a house based on the data provided by the user.
    :param data: A dictionary where the keys are the feature names and the values are the feature values
    :return: A text description of the house
    """
    system_prompt = f"""You will be given a conversation between a customer and a customer service agent and a set of quality rules.
    Answer the question based on the conversation.
    """
    # YOUR CODE HERE END
    messages = [
        # YOUR CODE HERE START: Add the system prompt if needed
        {"role": "system", "content": system_prompt},
        # YOUR CODE HERE END
        {"role": "user", 
         "content": f"Question: {question}"
         },
    ]

    response = CLIENT.chat.completions.create(
        model=os.environ["GPT_4_MODEL_NAME"],
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0.0,
    )

    return response.choices[0].message.content

def get_overall_conversation_score_and_tips(conversation:str, evaluation_results: list[dict]):

    class OverallConversationDataFormat(pydantic.BaseModel):
        tips: str = pydantic.Field(
            description=f"Tips for improvements for the customer service agent based on the conversation and quality assesment results provided."
        )
        overall_score: int  = pydantic.Field(
            description=f"Averall score between 0 and 100 for the customer service agent based on the conversation and quality assesment results provided."
        )
        topic: str = pydantic.Field(
            description=f"What was the request of the customer? Plese provide a one or two word topic based on the conversation."
        )

    system_prompt = f"""You will be given a conversation between a customer and a customer service agent, and quality assesment results about the conversation below.
        How can this agent improve? Please write some tips for the agent based on the conversation and quality assesment results provided. Do not use bullets or numbers for the tips.
        Please also provide an overall score for the agent between 0 and 100 based on the quality of the conversation and quality assesment results provided and their level.
        Finally, provide a topic regarding the customers request.
        Please keep it short and concise. You response must be valid list of JSON parsable by Pydantic using the following schema for each rule:
        {OverallConversationDataFormat.model_json_schema()}

        Conversation: {conversation}

        Quality Assessment Results: {evaluation_results}
        """
        
    messages = [
        {'role': 'system', 'content': system_prompt},
    ]
    response = CLIENT.chat.completions.create(
    model=os.environ["GPT_4_MODEL_NAME"],
    messages=messages,
    response_format={"type": "json_object"},
    temperature=0.0,

)
    message = response.choices[0].message.content
    overall_conversation_score_and_tips = OverallConversationDataFormat.model_validate_json(message, strict=False)
    overall_conversation_score = overall_conversation_score_and_tips.overall_score
    tips = overall_conversation_score_and_tips.tips
    topic = overall_conversation_score_and_tips.topic
    return overall_conversation_score, tips, topic


def evaluate_conversation(conversation:str, rules:dict):
    """
    Generate a text description of a house based on the data provided by the user.
    :param data: A dictionary where the keys are the feature names and the values are the feature values
    :return: A text description of the house
    """
    print("DataFormat", EvaluationDataFormatList.model_json_schema())
    system_prompt = f"""You will be given a conversation between a customer and a customer service agent and a set of quality rules. Please evaluate the conversation based on the given rules.
    You response must be valid list of JSON parsable by Pydantic using the following schema for each rule:
    {EvaluationDataFormatList.model_json_schema()}
    """
    # YOUR CODE HERE END
    messages = [
        # YOUR CODE HERE START: Add the system prompt if needed
        {"role": "system", "content": system_prompt},
        # YOUR CODE HERE END
        {"role": "user", 
         "content": f"""Rules:
{json.dumps(rules)}
Conversation:
{conversation}
         """ },
    ]

    response = CLIENT.chat.completions.create(
        model=os.environ["GPT_4_MODEL_NAME"],
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0.0,
    )

    message = response.choices[0].message.content
    print(message)

    evaluation_results = EvaluationDataFormatList.model_validate_json(message, strict=False)
    # try:
    #     evaluation_results = EvaluationDataFormat.model_validate_json(message, strict=False)
    # except pydantic.ValidationError as e:
    #     messages.append({"role": "assistant", "content": message})

    #     messages.append(
    #         {
    #             "role": "user",
    #             "content": f"There are some errors in your response! Respond only with JSON that does not have these errors: {e.errors()}",
    #         }
    #     )

    #     response = CLIENT.chat.completions.create(
    #     model=os.environ["GPT_4_MODEL_NAME"],
    #     response_format={"type": "json_object"},
    #     messages=messages,
    #     temperature=0.0,
    # )
    #     message = response.choices[0].message.content
    #     evaluation_results = EvaluationDataFormat.model_validate_json(message, strict=False)
    
    return evaluation_results.dict()["evaluation_results"]


# Here we create two columns to render the description and the extracted features side by side
col1, col2 = st.columns(2)


# This is the left column where we render the generated description
with col1:
        st.subheader("Conversation", divider=True)
        sub_col1, sub_col2, sub_col3, sub_col4  = st.columns(4)
        with sub_col1:
            submit = st.button("Generate", 
                               on_click=lambda: on_click_generate_conversation(), 
                               icon=":material/autorenew:")
        
        with sub_col2:
            edit_button = st.button("Edit", icon=":material/edit:")
        
        with sub_col3:
            save_button = st.button("Save", icon=":material/check:")


        with sub_col4:
            submit = st.button("Evaluate", 
                               on_click=lambda: on_click_evaluate_conversation(),
                               icon=":material/play_arrow:"
                               )
            
        
        if "conversation" not in st.session_state or st.session_state["conversation"] == "" or edit_button:
            conversation_input = st.text_area(label="Conversation", 
                                                value=st.session_state["conversation"],
                                                height=500, 
                                                placeholder="""You can copy paste a conversation in the following format.
    **Customer (10:00 AM):** ...
    **Agent (10:02 AM):** ..."""
                        )
            st.session_state["conversation"] = conversation_input

        else:
            st.write(st.session_state["conversation"])

        if save_button:
            st.session_state["conversation"] = conversation_input
            st.rerun()
        





# This is the right column where we render the extracted features
with col2:
    st.subheader("Quality Assessment", divider=True)

    sub_col1, sub_col2, sub_col3  = st.columns(3)
    with sub_col1:
        edit_button = st.button("Edit ", icon=":material/edit:")
        
    with sub_col2:
        save_button = st.button("Save ", icon=":material/check:")

    with sub_col3:
        submit = st.button("Generate PDF ", 
                            on_click=lambda: on_click_evaluate_conversation(),
                            icon=":material/picture_as_pdf:"
                            )
    if "evaluation_results" in st.session_state:    
        overall_conversation_score, tips, topic = get_overall_conversation_score_and_tips(conversation=st.session_state["conversation"], 
                                                                                        evaluation_results=st.session_state["evaluation_results"])
        
        number_of_failed_rules = len([item for item in st.session_state["evaluation_results"] if item["evaluation_result"] != "successful"])
        a, b, c = st.columns(3)
        a.markdown(f"💬 Topic")
        a.markdown(f"**{topic}**")
        b.metric("🏆 Overall Score", f"{overall_conversation_score}%", border=False)
        c.metric("❌ Failed Rules", f"{number_of_failed_rules}", border=False)
        st.subheader("Improvement Recommendation")
        st.write(tips)
    else:
        st.write("No feedback yet")
    
    st.subheader("Assessment Results")
    if "evaluation_results" in st.session_state:
        evaluation_results = st.session_state["evaluation_results"]
        if evaluation_results is None:
            st.write("Features could not be extracted.")
        else:
            if type(evaluation_results) is list and type(evaluation_results[0]) is dict:
                evaluation_results = pd.DataFrame(evaluation_results).sort_values("evaluation_result").to_dict('records')
                for rule_result in evaluation_results:
                     render_rule_evaluation(
                        rule_name=rule_result.get("rule_name", ""), 
                        rule_description=rule_result.get("rule_description", ""), 
                        level=rule_result.get("level", ""), 
                        evaluation_result=rule_result.get("evaluation_result", "unknown"), 
                        thoughts=rule_result.get("thoughts", ""), 
                    )
            else:
                st.write(str(evaluation_results))

    else:
        st.write("No evaluation results yet")

st.subheader("Extra Questions", divider=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Your question about the conversation?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        system_prompt = f"""You will be given a conversation between a customer and a customer service agent, and quality assesment results about the conversation below.
        Answer the question based on the conversation and quality assessment results.

        Conversation: {st.session_state["conversation"]}

        Quality Assessment Results: {st.session_state["evaluation_results"]}
        """
        messages = [

            {'role': 'system', 'content': system_prompt},
        ]
        for m in st.session_state.messages:
            messages.append({"role": m["role"], "content": m["content"]})
        response = CLIENT.chat.completions.create(
        model=os.environ["GPT_4_MODEL_NAME"],
        messages=messages,
        temperature=0.0,
    )
        assistant_message = response.choices[0].message.content
        message_placeholder.markdown(assistant_message)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message}
    )


    
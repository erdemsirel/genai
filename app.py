import json
import os
from typing import Any
import datetime
import pydantic

import streamlit as st
from llm_in_production.openai_utils import get_openai_client
from llm_in_production.text_extraction import (
    BooleanFeature,
    DigitFeature,
    StringFeature,
)

HOUSE_TYPES = ["Apartment", "House", "Studio"]
CLIENT = get_openai_client()


evaluation_rules = [

  {
    "rule_name": "Clarity of Communication",
    "rule_description": "The agent provides clear and concise responses, avoiding jargon and ambiguity.",
    "level": "High"
  },
  {
    "rule_name": "Empathy and Politeness",
    "rule_description": "The agent demonstrates understanding of the customer's emotions and responds in a polite and professional manner.",
    "level": "High"
  },
  {
    "rule_name": "Response Time",
    "rule_description": "The agent responds to the customer promptly, minimizing delays during the conversation.",
    "level": "Medium"
  },
  {
    "rule_name": "Issue Resolution",
    "rule_description": "The agent effectively addresses and resolves the customer's issue or provides clear next steps.",
    "level": "High"
  },
  {
    "rule_name": "Proactive Assistance",
    "rule_description": "The agent anticipates the customer's needs and offers additional support or solutions without being asked.",
    "level": "Medium"
  },
  {
    "rule_name": "Knowledge Accuracy",
    "rule_description": "The agent provides accurate and relevant information in response to the customer's queries.",
    "level": "High"
  },
  {
    "rule_name": "Professional Tone",
    "rule_description": "The agent maintains a respectful, positive, and professional tone throughout the interaction.",
    "level": "High"
  },
  {
    "rule_name": "Engagement",
    "rule_description": "The agent actively engages with the customer by acknowledging their concerns and confirming their satisfaction.",
    "level": "Medium"
  },
  {
    "rule_name": "Grammar and Spelling",
    "rule_description": "The agent's messages are free from grammatical errors",
    "level": "Medium"
  }
]


####################
# Logic for the UI #
####################

try:
    with open(f"example_conversation_20241214175326.txt", "+rt") as f:
        messages = "\n".join(f.readlines())
    st.session_state["conversation"] = messages
except:
    pass

def generate_example_chat_conversation(new_conversation=False) -> str:
    """
    Generate a text description of a house based on the data provided by the user.
    :param data: A dictionary where the keys are the feature names and the values are the feature values
    :return: A text description of the house
    """
    messages = [
        {"role": "system", 
        "content": """
Please create an example of chat conversation between an mobile subscription company's customer service agents and a customer who is late on his payment. 
Customer wants to agree on a payment arrangement but wants a lower monthly payment because he had lost his job recently. 
Agent need to varify few things regarding the customer income and expenses but eventually they agree on a payment arrangement.
The agent is a new joiner so sometimes he makes mistakes and does not use a professional tone.
You can give example names and amount while creating the conversation. Please also add a timestamps for each message.
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
    with open(f"example_conversation_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt", "+wt") as f:
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


    evaluation_results = evaluate_conversation(conversation=st.session_state["conversation"], rules=evaluation_rules)
    st.session_state["evaluation_results"] = evaluation_results



######################
# Start Streamlit UI #
######################

title = "Customer Service Quality Evaluator"
st.set_page_config(
    page_title=title,
    page_icon="👋",
    layout="wide",
)
st.title(title)


# This is the sidebar where we collect the user input
with st.sidebar:
    st.title("Settings")

    # Pressing this button will trigger the on_click_generate function defined above
    # This will generate a description and extract the features
    submit = st.button("New conversation example", on_click=lambda: on_click_generate_conversation())
    options = st.multiselect(
    "Select the rules that you want to evaluate",
    options=[rule["rule_name"] for rule in evaluation_rules],
    default=[rule["rule_name"] for rule in evaluation_rules],
)
    submit = st.button("Evaluate conversation", on_click=lambda: on_click_evaluate_conversation())



# Here we create two columns to render the description and the extracted features side by side
col1, col2 = st.columns(2)


# This is the left column where we render the generated description
with col1:
    is_there_a_description = "conversation" in st.session_state
    if is_there_a_description:
        st.write(st.session_state["conversation"])
    else:
        st.write("No conversation yet")


def render_rule_evaluation(
    rule_name: str, 
    rule_description: str, 
    level: list[str] | None, 
    evaluation_result: str| None, 
    thoughts: str| None, 
):

    st.write(f"### {rule_name} {level}")
    st.write(rule_description)
    if evaluation_result == "successful":
        st.write(f"✅ Passed: {thoughts}")
    elif rule_description == "not_successful":
        st.write(f"❌ Incorrect: {thoughts}")
    else:
        st.write(f"❓ Unknown: {thoughts}")
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
    
    return evaluation_results


# This is the right column where we render the extracted features
with col2:
    if "evaluation_results" in st.session_state:
        evaluation_results = st.session_state["evaluation_results"].dict()["evaluation_results"]
        if evaluation_results is None:
            st.write("Features could not be extracted.")
        else:
            if type(evaluation_results) is list and type(evaluation_results[0]) is dict:
                print("test 1")
                for rule_result in evaluation_results:
                     print("test 2", rule_result)
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
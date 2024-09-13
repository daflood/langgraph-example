import json
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from typing import List, Dict, Any, Literal, Union, Annotated, Set, TypedDict
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the sample profile
from my_agent.sample_profile import sample_profile

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define all profile questions globally
topics_and_questions = [
    {"topic": "Personal Background and Identity", "question": "What's your full name? Do you have any nicknames?"},
    {"topic": "Personal Background and Identity", "question": "When and where were you born?"},
    {"topic": "Personal Background and Identity", "question": "What is your nationality and ethnicity?"},
    {"topic": "Personal Background and Identity", "question": "What are your gender and preferred pronouns?"},
    {"topic": "Personal Background and Identity", "question": "Which languages do you speak?"},
    {"topic": "Family and Relationships", "question": "Can you tell me about your family history?"},
    {"topic": "Family and Relationships", "question": "Are you married or in a relationship? Do you have any children?"},
    {"topic": "Family and Relationships", "question": "Do you have any siblings or extended family members you're close to?"},
    {"topic": "Family and Relationships", "question": "Who are some key people in your life, like friends or mentors?"},
    {"topic": "Education and Professional Life", "question": "What is your educational background? Where did you go to school?"},
    {"topic": "Education and Professional Life", "question": "Can you share about your professional career and accomplishments?"},
    {"topic": "Education and Professional Life", "question": "Have you been involved in volunteer work or community activities?"},
    {"topic": "Education and Professional Life", "question": "Who have been key mentors or influencers in your professional development?"},
    {"topic": "Health and Well-being", "question": "Can you share about your physical and mental health history?"},
    {"topic": "Health and Well-being", "question": "Have you experienced any major illnesses or health challenges? If so, how did they impact your life?"},
    {"topic": "Health and Well-being", "question": "What lifestyle choices have you made related to health, such as diet or exercise?"},
    {"topic": "Major Life Events and Experiences", "question": "Can you share about any defining moments in your life, such as childhood, adolescence, or adulthood?"},
    {"topic": "Major Life Events and Experiences", "question": "Have you faced any personal challenges? How did you overcome them?"},
    {"topic": "Major Life Events and Experiences", "question": "Can you share about any impactful historical events you've experienced, such as wars or political movements?"},
    {"topic": "Major Life Events and Experiences", "question": "What are your religious or spiritual beliefs?"},
    {"topic": "Cultural and Social Environment", "question": "Can you share about your social class and economic background?"},
    {"topic": "Cultural and Social Environment", "question": "How have cultural influences, such as art, music, or media, shaped your life?"},
    {"topic": "Cultural and Social Environment", "question": "Have you moved to different geographical locations? If so, how did they impact your life?"},
    {"topic": "Cultural and Social Environment", "question": "Can you share about the social or political context of the time in which you lived?"},
    {"topic": "Hobbies, Interests, and Passions", "question": "What are your major hobbies and personal interests?"},
    {"topic": "Hobbies, Interests, and Passions", "question": "Have you had any significant travel experiences? If so, what were they like?"},
    {"topic": "Hobbies, Interests, and Passions", "question": "Are you a member of any organizations or clubs? If so, how have they influenced your life?"},
    {"topic": "Hobbies, Interests, and Passions", "question": "Have you contributed to your community or society in any meaningful way, such as through activism or advocacy?"},
    {"topic": "Aspirations, Values, and Legacy", "question": "What are your personal values and guiding principles?"},
    {"topic": "Aspirations, Values, and Legacy", "question": "What are your life aspirations, whether fulfilled or unfulfilled?"},
    {"topic": "Aspirations, Values, and Legacy", "question": "How do you reflect on your life achievements and regrets?"},
    {"topic": "Aspirations, Values, and Legacy", "question": "How do you wish to be remembered?"},
]

# Define the state with a built-in `messages` key
class UserState(MessagesState):
    user_profile_status: bool
    agent_introduction_status: bool
    profile_info_collected: bool
    profile_history: List[Dict[str, str]] = []
    current_question: str = ""
    current_topic: str = ""
    topics_covered: Set[str] = set()
    follow_up_counts: Dict[str, int] = {}

# Define the state with a built-in messages key
class BiographerState(MessagesState):
    contacts: List[str]  # List of known contacts
    chapter_complete: bool  # Flag to indicate if a chapter is complete

   # Define a new configuration schema if needed
class Config(TypedDict):
    start_node: str

# Set the entry point based on configuration
def set_entry_point_based_on_config(graph, config):
    start_node = config.get("start_node", "User Status Node")
    if start_node == "Interview Question Prep":
        graph.add_edge(START, "Interview Question Prep")
    else:
        graph.add_edge(START, "User Status Node")

# Define the logic for each node
def user_status_node(state: UserState):
    logger.info(f"user_status_node received state: {state}")
    if state.get("user_profile_status") is None:
        return {"user_profile_status": False}
    else:
        return {"messages": state["messages"] + [SystemMessage(content="User status checked.")]}

# Modify the initialize_node to handle different start points
def initialize_node(state: UserState, config: Config):
    logger.info(f"initialize_node received state: {state} with config: {config}")
    start_node = config.get("start_node", "User Status Node")
    if start_node == "Interview Question Prep":
        # Load the sample profile into the state
        state["profile_history"] = sample_profile
        state["topics_covered"] = {item["topic"] for item in sample_profile}
        logger.info("Loaded sample profile into state.")
    return {
        "user_profile_status": False,
        "agent_introduction_status": False,
        "messages": state["messages"] + [SystemMessage(content="State initialized.")],
        "start_node": start_node
    }

def agent_introduction_check_node(state: UserState):
    logger.info(f"agent_introduction_check_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Checking if agent introduction is needed.")]}

def agent_introduction_node(state: UserState):
    logger.info(f"agent_introduction_node received state: {state}")
    
    introduction = (
        "Hello! I'm your virtual biographer. My purpose is to assist you in creating your autobiography. "
        "I'll guide you through the process of recalling and organizing your life experiences, "
        "helping you craft a meaningful narrative of your life story. "
        "Let's begin this journey of self-discovery and storytelling."
    )
    
    return {
        "agent_introduction_status": True,
        "messages": state["messages"] + [
            SystemMessage(content="Agent is introducing itself as a virtual biographer."),
            AIMessage(content=introduction)
        ]
    }

def profile_check_node(state: UserState):
    logger.info(f"profile_check_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Profile status checked.")]}

def profile_introduction_node(state: UserState):
    logger.info(f"profile_introduction_node received state: {state}")
    
    profile_intro = (
        "To help me write your biography effectively, I'd like to start by asking you some general questions "
        "about your life. This will give me a better understanding of your background and experiences. "
        "With this information, I'll be able to ask more appropriate and insightful questions as we delve "
        "deeper into your life story. You can take your time with this and come back at any time. Most people take a few days to get through all of these questions. Because we're telling the story of your life, many of these questions will be personal. You don't need to answer anything that makes you uncomfortable and we can always skip questions or come back to them later. Are you read to get started?"
    )
    
    return {
        "messages": state["messages"] + [
            SystemMessage(content="Introducing the profile gathering process."),
            AIMessage(content=profile_intro)
        ]
    }

def profile_node(state: UserState):
    logger.info(f"profile_node received state: {state}")
    logger.info("Conversation history in profile_node:")
    for msg in state["messages"]:
        logger.info(f"  {type(msg).__name__}: {msg.content}")
    
    conversation_history = state["messages"]
    last_message = conversation_history[-1] if conversation_history else None

    # Initialize topics_covered and profile_history from state
    topics_covered = set(state.get("topics_covered", []))
    profile_history = state.get("profile_history", [])

    # If waiting for user response
    if state.get("awaiting_user_response", False) and last_message and isinstance(last_message, HumanMessage):
        user_response = last_message.content
        last_question = state.get("current_question", "")
        last_topic = state.get("current_topic", "")
        
        # Update profile history
        if last_question:
            profile_history.append({
                "question": last_question,
                "answer": user_response
            })
            logger.info(f"Updated profile history: {profile_history}")
    
            # Optionally perform analysis
            # ...
        
        # Reset awaiting_user_response
        state["awaiting_user_response"] = False

    # Find the next unanswered question
    next_question = None
    for item in topics_and_questions:
        if item["question"] not in {qa["question"] for qa in profile_history}:
            next_question = item
            break

    if next_question:
        # Rephrase the question
        prompt = f"""Rephrase the following question in a conversational and natural way:

Topic: {next_question['topic']}
Question: {next_question['question']}

Rephrased question:"""
        try:
            response = llm.invoke(prompt)
            question = response.content.strip()
        except Exception as e:
            logger.error(f"Error during LLM invocation for rephrasing: {e}")
            question = next_question['question']
    
        # Update state
        state["current_question"] = next_question['question']
        state["current_topic"] = next_question['topic']
        state["awaiting_user_response"] = True

        # Add to topics_covered
        topics_covered.add(next_question['topic'])
    
        # Append the question to messages
        return {
            "messages": state["messages"] + [AIMessage(content=question)],
            "current_question": next_question['question'],
            "current_topic": next_question['topic'],
            "awaiting_user_response": True,
            "profile_history": profile_history,
            "topics_covered": list(topics_covered)
        }
    else:
        # No more questions
        closing_message = "Thank you for sharing so much about yourself. I feel like I know you better now."
        return {
            "messages": state["messages"] + [AIMessage(content=closing_message)],
            "profile_history": profile_history,
            "topics_covered": list(topics_covered)
        }

def profile_completion_check_node(state: dict):
    logger.info(f"profile_completion_check_node received state: {state}")
    profile_history = state.get("profile_history", [])
    total_questions = len(topics_and_questions)

    answered_questions = len(profile_history)
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Answered questions: {answered_questions}")

    if answered_questions >= total_questions:
        return {
            "next_step": "Interview Question Prep",
            "messages": state["messages"] + [SystemMessage(content="All profile questions have been answered. Moving to interview question preparation.")]
        }
    else:
        return {
            "next_step": "Profile Node",
            "messages": state["messages"]
        }

def profile_question_validation_node(state: dict):
    logger.info(f"profile_question_validation_node received state: {state}")
    logger.info("Conversation history in profile_question_validation_node:")
    for msg in state["messages"]:
        logger.info(f"  {type(msg).__name__}: {msg.content}")
    
    conversation_history = state["messages"]
    
    # Safely retrieve the last AI and Human messages
    last_ai_message = next((msg for msg in reversed(conversation_history) if isinstance(msg, AIMessage)), None)
    last_human_message = next((msg for msg in reversed(conversation_history) if isinstance(msg, HumanMessage)), None)
    
    # Initialize variables
    last_question = ""
    last_answer = ""
    
    if last_ai_message:
        last_question = last_ai_message.content
    if last_human_message:
        last_answer = last_human_message.content
    
    if not last_question:
        logger.info("No last_question found. Skipping validation.")
        # Handle the case where there is no last_question
        return {
            "messages": state["messages"],
            "next_step": "Profile Node"  # Or any appropriate next step
        }
    
    logger.info(f"Last question: {last_question}")
    logger.info(f"Last answer: {last_answer}")
    
    # Update the profile history
    profile_history = state.get("profile_history", [])
    if last_question and last_answer:
        profile_history.append({
            "question": last_question,
            "answer": last_answer
        })
        logger.info(f"Updated profile history: {profile_history}")
        
        # Update the state with the new profile history
        state["profile_history"] = profile_history
    
    # Determine if a follow-up question is appropriate
    prompt = f"""You are a biographer interviewing a subject to create a demographic profile. Here are the most recent question and answer:
Question: {last_question}
Answer: {last_answer}

If the answer suggests an interesting area for further exploration, respond with 'Follow Up'.  
Otherwise, respond with 'Complete'. 
"""
    
    try:
        analysis_response = llm.invoke(prompt)
        decision = analysis_response.content.strip().lower()
    except Exception as e:
        logger.error(f"Error during LLM invocation for validation: {e}")
        decision = "complete"  # Default decision on error
    
    current_topic = state.get("current_topic", "")
    follow_up_counts = state.get("follow_up_counts", {})
    follow_up_count = follow_up_counts.get(current_topic, 0)
    
    if "follow up" in decision and follow_up_count < 2:
        next_step = "Profile Follow Up"
    else:
        next_step = "Profile Completion Check"
    
    logger.info(f"Decision: {decision}, Next step: {next_step}")
    
    # Update follow_up_counts in the state
    if "follow up" in decision:
        follow_up_counts[current_topic] = follow_up_count + 1
        state["follow_up_counts"] = follow_up_counts
    
    return {
        "messages": state["messages"],
        "next_step": next_step
    }

def next_step(state):
    decision = state.get("next_step", "Profile Completion Check")
    logger.info(f"Next step decision: {decision}")  # Add this line for debugging
    return decision

def profile_follow_up_node(state: UserState):
    logger.info(f"profile_follow_up_node received state: {state}")
    logger.info("Conversation history in profile_follow_up_node:")
    for msg in state["messages"]:
        logger.info(f"  {type(msg).__name__}: {msg.content}")
    
    conversation_history = state["messages"]
    last_question = state.get("current_question", "")
    last_answer = next((msg.content for msg in reversed(conversation_history) if isinstance(msg, HumanMessage)), "")
    
    logger.info(f"Last question: {last_question}")
    logger.info(f"Last answer: {last_answer}")
    
    # Update the profile history with the previous question-answer pair
    profile_history = state.get("profile_history", [])
    if last_question and last_answer:
        try:
            if profile_history and profile_history[-1]["question"] == last_question:
                profile_history[-1]["answer"] = last_answer
            else:
                profile_history.append({
                    "question": last_question,
                    "answer": last_answer
                })
            logger.info(f"Updated profile history item: {profile_history[-1]}")
        except Exception as e:
            logger.error(f"Error updating profile history: {e}")
    
    logger.info(f"Updated profile history: {profile_history}")
    
    # Prepare the profile history for the prompt
    try:
        history_str = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in profile_history])
    except Exception as e:
        logger.error(f"Error preparing profile history string: {e}")
        history_str = ""
    
    # Generate a new question using the LLM
    prompt = f"""You are a biographer working with a new subject and you have just asked the user a question and received an answer. Generate a follow-up question based on the previous question and answer that logically extends the conversation and shows genuine interest in learning more about the user. Do not repeat previous questions. 

Previous question: {last_question}
User's answer: {last_answer}

Here's the conversation history so far:
{history_str}

Use clear and concise language. Avoid exclamation marks and excessive enthusiasm. 
Follow-up question:"""
    
    try:
        follow_up_response = llm.invoke(prompt)
        follow_up_question = follow_up_response.content.strip()
    except Exception as e:
        logger.error(f"Error during LLM invocation for follow-up: {e}")
        follow_up_question = "Can you tell me more about that?"
    
    # Update follow_up_counts in the state
    try:
        current_topic = state.get("current_topic", "")
        follow_up_counts = state.get("follow_up_counts", {})
        follow_up_counts[current_topic] = follow_up_counts.get(current_topic, 0) + 1
        state["follow_up_counts"] = follow_up_counts
    except Exception as e:
        logger.error(f"Error updating follow_up_counts: {e}")
    
    # Append the agent's question to the conversation history
    try:
        updated_messages = state["messages"] + [AIMessage(content=follow_up_question)]
    except Exception as e:
        logger.error(f"Error appending new message to conversation history: {e}")
        updated_messages = state["messages"]
    
    return {
        "messages": updated_messages,
        "current_question": follow_up_question,
        "awaiting_user_response": True,
        "profile_history": profile_history,
        "follow_up_counts": follow_up_counts
    }

def interview_question_prep_node(state: UserState):
    logger.info(f"interview_question_prep_node received state: {state}")
    
    profile_history = state.get("profile_history", [])
    if not profile_history:
        logger.warning("No profile history found.")
        return {"messages": state["messages"] + [SystemMessage(content="No profile data available.")]}

    for qa in profile_history:
        question = qa["question"]
        answer = qa["answer"]
        state["messages"] += [AIMessage(content=question), HumanMessage(content=answer)]
    
    logger.info("Loaded profile questions and answers into the conversation history.")
    
    # Proceed to the next node
    return {
        "next_step": "Interview Question Prep",
        "messages": state["messages"] + [SystemMessage(content="Loaded all profile data.")]
    }

def questioning_node(state: BiographerState):
    logger.info(f"questioning_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="What is your story?")]}

def contact_check_node(state: BiographerState):
    logger.info(f"contact_check_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Checking for contacts.")]}

def contact_validation_node(state: BiographerState):
    logger.info(f"contact_validation_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Validating contact.")]}

def contact_store_node(state: BiographerState):
    logger.info(f"contact_store_node received state: {state}")
    return {
        "contacts": state.get("contacts", []) + ["new_contact"],
        "messages": state["messages"] + [SystemMessage(content="New contact stored.")]
    }

def reach_validation_node(state: BiographerState):
    logger.info(f"reach_validation_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Validating reach.")]}

def contact_node(state: BiographerState):
    logger.info(f"contact_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Contact processed.")]}

def conversation_validation_node(state: BiographerState):
    logger.info(f"conversation_validation_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Conversation validated.")]}

def follow_up_node(state: BiographerState):
    logger.info(f"follow_up_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Can you elaborate?")]}

def chapter_check_node(state: BiographerState):
    logger.info(f"chapter_check_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Checking chapter status.")]}

def chapter_writer_node(state: BiographerState):
    logger.info(f"chapter_writer_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Writing chapter...")]}

def chapter_save_node(state: BiographerState):
    logger.info(f"chapter_save_node received state: {state}")
    return {
        "chapter_complete": True,
        "messages": state["messages"] + [SystemMessage(content="Saving chapter...")]
    }

def congratulations_node(state: BiographerState):
    logger.info(f"congratulations_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Congratulations!")]}

# Create the state graph
graph = StateGraph(UserState)

# Add nodes to the graph
graph.add_node("User Status Node", user_status_node)
graph.add_node("Initialize Node", initialize_node)
graph.add_node("Agent Introduction Check Node", agent_introduction_check_node)
graph.add_node("Agent Introduction Node", agent_introduction_node)
graph.add_node("Profile Check Node", profile_check_node)
graph.add_node("Profile Introduction Node", profile_introduction_node)
graph.add_node("Profile Node", profile_node)
graph.add_node("Profile Question Validation", profile_question_validation_node)
graph.add_node("Profile Follow Up", profile_follow_up_node)
graph.add_node("Profile Completion Check", profile_completion_check_node)
graph.add_node("Interview Question Prep", interview_question_prep_node)
graph.add_node("Questioning Node", questioning_node)
graph.add_node("contact_check_node", contact_check_node)
graph.add_node("contact_validation_node", contact_validation_node)
graph.add_node("contact_store_node", contact_store_node)
graph.add_node("reach_validation_node", reach_validation_node)
graph.add_node("contact_node", contact_node)
graph.add_node("conversation_validation_node", conversation_validation_node)
graph.add_node("follow_up_node", follow_up_node)
graph.add_node("chapter_check_node", chapter_check_node)
graph.add_node("chapter_writer_node", chapter_writer_node)
graph.add_node("chapter_save_node", chapter_save_node)
graph.add_node("congratulations_node", congratulations_node)

# Define edges between nodes
graph.add_conditional_edges(
    "User Status Node",
    lambda x: "Initialize Node" if x.get("user_profile_status") is False else "Agent Introduction Check Node",
    {
        "Initialize Node": "Initialize Node",
        "Agent Introduction Check Node": "Agent Introduction Check Node"
    }
)
graph.add_edge("Initialize Node", "Agent Introduction Check Node")
graph.add_conditional_edges(
    "Agent Introduction Check Node",
    lambda x: "Agent Introduction Node" if not x.get("agent_introduction_status") else "Profile Check Node",
    {
        "Agent Introduction Node": "Agent Introduction Node",
        "Profile Check Node": "Profile Check Node"
    }
)
graph.add_edge("Agent Introduction Node", "Profile Check Node")
graph.add_conditional_edges(
    "Profile Check Node",
    lambda x: "Profile Introduction Node" if not x.get("user_profile_status") else "Profile Node",
    {
        "Profile Introduction Node": "Profile Introduction Node",
        "Profile Node": "Profile Node"
    }
)
graph.add_edge("Profile Introduction Node", "Profile Node")
graph.add_edge("Profile Node", "Profile Question Validation")

graph.add_conditional_edges(
    "Profile Question Validation",
    next_step,
    {
        "Profile Follow Up": "Profile Follow Up",
        "Profile Completion Check": "Profile Completion Check"
    }
)

graph.add_edge("Profile Follow Up", "Profile Question Validation")


# Update edges after Profile Completion Check Node
graph.add_conditional_edges(
    "Profile Completion Check",
    lambda x: x["next_step"],
    {
        "Interview Question Prep": "Interview Question Prep",
        "Profile Node": "Profile Node"
    }
)

graph.add_edge("Interview Question Prep", "Questioning Node")
graph.add_conditional_edges(
    "Interview Question Prep",
    lambda x: "User Status Node",  # Define logic to route back if necessary
    {
        "User Status Node": "User Status Node",
    }
)
graph.add_edge("Questioning Node", "contact_check_node")
graph.add_conditional_edges(
    "contact_check_node",
    lambda x: "contact_validation_node" if "someone" in x["messages"][-1].content else "conversation_validation_node",
    {
        "contact_validation_node": "contact_validation_node",
        "conversation_validation_node": "conversation_validation_node"
    }
)
graph.add_conditional_edges(
    "contact_validation_node",
    lambda x: "contact_store_node" if "new_contact" in x["messages"][-1].content else "reach_validation_node",
    {
        "contact_store_node": "contact_store_node",
        "reach_validation_node": "reach_validation_node"
    }
)
graph.add_edge("contact_store_node", "reach_validation_node")
graph.add_edge("reach_validation_node", "contact_node")
graph.add_edge("contact_node", "conversation_validation_node")
graph.add_conditional_edges(
    "conversation_validation_node",
    lambda x: "follow_up_node" if "follow-up" in x["messages"][-1].content else "chapter_check_node",
    {
        "follow_up_node": "follow_up_node",
        "chapter_check_node": "chapter_check_node"
    }
)
graph.add_edge("follow_up_node", "contact_check_node")
graph.add_conditional_edges(
    "chapter_check_node",
    lambda x: "chapter_writer_node" if x.get("chapter_complete") else "Questioning Node",
    {
        "chapter_writer_node": "chapter_writer_node",
        "Questioning Node": "Questioning Node"
    }
)
graph.add_edge("chapter_writer_node", "chapter_save_node")
graph.add_edge("chapter_save_node", "congratulations_node")
graph.add_edge("congratulations_node", END)

# Example configuration. Set "start_node" to "Interview Question Prep" to start with interview questions. Set "start_node" to "User Status Node" to start with the user's status.
config = {"start_node": "Interview Question Prep"}


# Set the entry point before compiling
set_entry_point_based_on_config(graph, config)

# Compile the graph with the configuration
app = graph.compile(interrupt_before=["Profile Node"])

# Prepare the input data
input_data = {
    "messages": [],
    "user_profile_status": False,
    "agent_introduction_status": False,
    "profile_info_collected": False,
    "profile_history": [],
    "current_question": "",
    "current_topic": "",
    "topics_covered": set(),
    "follow_up_counts": {}
}
final_state = app.invoke(input=input_data)
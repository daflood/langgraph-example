from langgraph.graph import StateGraph, MessagesState, END
from typing import List, Dict, Any, Literal, Union, Annotated
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define the state with a built-in `messages` key
class UserState(MessagesState):
    user_profile_status: bool
    agent_introduction_status: bool
    profile_info_collected: bool
    profile_history: Annotated[List[Dict[str, str]], "profile_history"] = []

# Define the state with a built-in messages key
class BiographerState(MessagesState):
    contacts: List[str]  # List of known contacts
    chapter_complete: bool  # Flag to indicate if a chapter is complete

# Define the logic for each node
def user_status_node(state: UserState):
    print(f"user_status_node received state: {state}")
    if state.get("user_profile_status") is None:
        return {"user_profile_status": False}
    else:
        return {"messages": state["messages"] + [SystemMessage(content="User status checked.")]}

def initialize_node(state: UserState):
    print(f"initialize_node received state: {state}")
    return {
        "user_profile_status": False,
        "agent_introduction_status": False,
        "messages": state["messages"] + [SystemMessage(content="State initialized.")]
    }

def agent_introduction_check_node(state: UserState):
    print(f"agent_introduction_check_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Checking if agent introduction is needed.")]}

def agent_introduction_node(state: UserState):
    print(f"agent_introduction_node received state: {state}")
    
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
    print(f"profile_check_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Profile status checked.")]}

def profile_introduction_node(state: UserState):
    print(f"profile_introduction_node received state: {state}")
    
    profile_intro = (
        "To help me write your biography effectively, I'd like to start by asking you some general questions "
        "about your life. This will give me a better understanding of your background and experiences. "
        "With this information, I'll be able to ask more appropriate and insightful questions as we delve "
        "deeper into your life story. Are you ready to begin with some general questions about yourself?"
    )
    
    return {
        "messages": state["messages"] + [
            SystemMessage(content="Introducing the profile gathering process."),
            AIMessage(content=profile_intro)
        ]
    }

def profile_node(state: UserState):
    print(f"profile_node received state: {state}")
    
    # Generate a question using the LLM
    prompt = "As a helpful biographer, ask a general question to gather information about someone's life for their biography."
    response = llm.invoke(prompt)
    question = response.content
    
    print(f"Generated question: {question}")
    
    # Return the state with the question and update the messages
    return {
        "messages": state["messages"] + [AIMessage(content=question)],
        "current_question": question,
        "awaiting_user_response": True
    }

def profile_question_validation_node(state: MessagesState):
    print(f"profile_question_validation_node received state: {state}")
    
    last_question = state.get("current_question")
    last_answer = state["messages"][-1].content if state["messages"] else ""
    
    # Determine if a follow-up question is appropriate
    prompt = f"""Based on the following question and answer, determine if a follow-up question would be appropriate to extend the conversation and gather more information about the user's life story. 
    Question: {last_question}
    Answer: {last_answer}
    
    If a follow-up is appropriate, respond with 'Follow Up'. Otherwise, respond with 'Complete'.
    """
    
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    
    if "follow up" in decision:
        return {
            "messages": state["messages"],
            "next_step": "Profile Follow Up"
        }
    else:
        return {
            "messages": state["messages"],
            "next_step": "Profile Completion Check"
        }

def next_step(state):
    return state.get("next_step", "Profile Completion Check")

def profile_follow_up_node(state: UserState):
    print(f"profile_follow_up_node received state: {state}")
    
    conversation_history = state["messages"]
    last_question = state.get("current_question", "")
    last_answer = conversation_history[-1].content if conversation_history else ""
    
    prompt = f"""As a conversational agent interested in the user's life story, generate a follow-up question based on the previous question and answer. The question should logically extend the conversation and show genuine interest in learning more about the user.

Previous question: {last_question}
User's answer: {last_answer}

Follow-up question:"""
    
    response = llm.invoke(prompt)
    follow_up_question = response.content.strip()
    
    # Append the agent's question to the conversation history
    updated_messages = state["messages"] + [AIMessage(content=follow_up_question)]
    
    return {
        "messages": updated_messages,
        "current_question": follow_up_question,
        "awaiting_user_response": True
    }

def human_feedback(state: UserState) -> UserState:
    # This function will be interrupted by LangGraph Studio
    # The user's response will be added to the state here
    user_response = input("User: ")  # This is a placeholder for LangGraph Studio interruption
    state["messages"].append(HumanMessage(content=user_response))
    state["awaiting_user_response"] = False
    return state

def profile_completion_check_node(state: UserState):
    print(f"profile_completion_check_node received state: {state}")
    profile_history = state.get("profile_history", [])
    
    # Check if we have a pending question and user response
    if state.get("current_question") and state.get("awaiting_user_response") is False:
        last_question = state["current_question"]
        last_answer = state["messages"][-1].content if state["messages"] else ""
        
        # Append the question-answer pair to the profile history
        profile_history.append({
            "question": last_question,
            "answer": last_answer
        })
        
        # Clear the current question
        state["current_question"] = None
    
    if len(profile_history) >= 3:  # Collect at least 3 pieces of information
        return {
            "user_profile_status": True,
            "profile_info_collected": True,
            "profile_history": profile_history,
            "messages": state["messages"] + [SystemMessage(content="Profile validated.")]
        }
    else:
        # If we don't have enough information, go back to the profile node
        return {
            "profile_history": profile_history,
            "messages": state["messages"] + [SystemMessage(content="Continuing profile collection.")]
        }

def questioning_node(state: BiographerState):
    print(f"questioning_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="What is your story?")]}

def contact_check_node(state: BiographerState):
    print(f"contact_check_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Checking for contacts.")]}

def contact_validation_node(state: BiographerState):
    print(f"contact_validation_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Validating contact.")]}

def contact_store_node(state: BiographerState):
    print(f"contact_store_node received state: {state}")
    return {
        "contacts": state.get("contacts", []) + ["new_contact"],
        "messages": state["messages"] + [SystemMessage(content="New contact stored.")]
    }

def reach_validation_node(state: BiographerState):
    print(f"reach_validation_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Validating reach.")]}

def contact_node(state: BiographerState):
    print(f"contact_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Contact processed.")]}

def conversation_validation_node(state: BiographerState):
    print(f"conversation_validation_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Conversation validated.")]}

def follow_up_node(state: BiographerState):
    print(f"follow_up_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Can you elaborate?")]}

def chapter_check_node(state: BiographerState):
    print(f"chapter_check_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Checking chapter status.")]}

def chapter_writer_node(state: BiographerState):
    print(f"chapter_writer_node received state: {state}")
    return {"messages": state["messages"] + [SystemMessage(content="Writing chapter...")]}

def chapter_save_node(state: BiographerState):
    print(f"chapter_save_node received state: {state}")
    return {
        "chapter_complete": True,
        "messages": state["messages"] + [SystemMessage(content="Saving chapter...")]
    }

def congratulations_node(state: BiographerState):
    print(f"congratulations_node received state: {state}")
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
graph.add_node("Human Feedback", human_feedback)
graph.add_node("Profile Completion Check", profile_completion_check_node)
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

graph.add_edge("Profile Follow Up", "Human Feedback")
graph.add_edge("Profile Completion Check", "Human Feedback")
graph.add_edge("Human Feedback", "Profile Question Validation")
graph.add_conditional_edges(
    "Profile Question Validation",
    lambda x: "Profile Node" if not x.get("profile_info_collected") else "Questioning Node",
    {
        "Profile Node": "Profile Node",
        "Questioning Node": "Questioning Node"
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

# Set the entry point before compiling
graph.set_entry_point("User Status Node")

# Compile the graph with an interrupt before the Human Feedback node
app = graph.compile(interrupt_before=["Human Feedback"])
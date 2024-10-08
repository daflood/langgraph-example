import json
import os
import logging
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from typing import List, Dict, Any, Literal, Union, Annotated, Set, TypedDict
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver


memory = MemorySaver()
# set true and true to run in testing mode
TESTING_MODE = os.getenv("TESTING_MODE", "True") == "True"

# Use this code for testing in LangGraph Studio
# Import the sample profile and interview for testing
from my_agent.sample_profile import sample_profile
# logger.info(f"sample_profile contains {len(sample_profile)} items")
from my_agent.sample_interview import sample_interview

# # Use relative imports
# from .sample_profile import sample_profile
# # logger.info(f"sample_profile contains {len(sample_profile)} items")
# from .sample_interview import sample_interview

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the compiled graph
app = None

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
    user_profile_status: bool = False
    agent_introduction_status: bool = False
    profile_info_collected: bool = False
    profile_history: List[Dict[str, str]] = Field(default_factory=list)
    interaction_history: List[Dict[str, str]] = Field(default_factory=list)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    current_question: Dict[str, Any] = Field(default_factory=dict)
    current_topic: str = ""
    current_chapter_index: int = 0
    current_question_index: int = 0
    topics_covered: Set[str] = Field(default_factory=set)
    follow_up_counts: Dict[str, int] = Field(default_factory=dict)
    interview_questions: List[Dict[str, Any]] = Field(default_factory=list)
    awaiting_user_response: bool = False
    is_initial_interaction: bool = True
    next_step: str = Field(default="")
    # messages: List[Dict[str, Any]] = Field(default_factory=list)
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
    elif start_node == "Questioning Node":
        graph.add_edge(START, "Questioning Node")
    else:
        graph.add_edge(START, "User Status Node")

#This is the single entry point for processing incoming messages
def process_message(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the incoming message and updates the state.
    
    Args:
        input_data (Dict[str, Any]): Contains 'messages' and 'state'.
    
    Returns:
        Dict[str, Any]: Updated 'messages' and 'state'.
    """
    try:
        # Invoke the compiled graph with input data
        result = app.invoke(input=input_data)
        
        # Extract updated state and messages
        updated_state = result.get("state", {})
        new_messages = result.get("messages", [])
        
        return {
            "state": updated_state,
            "messages": new_messages
        }
    except Exception as e:
        logger.exception(f"Error processing message in agent: {e}")
        return {
            "state": input_data.get("state", {}),
            "messages": [
                SystemMessage(content="An error occurred while processing your message.")
            ]
        }

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

def agent_introduction_node(state: UserState) -> Dict[str, Any]:
    logger.info(f"agent_introduction_node received state: {state}")
    
    introduction = (
        "Hello! I'm your virtual biographer. My purpose is to assist you in creating your autobiography. "
        "I'll guide you through the process of recalling and organizing your life experiences, "
        "helping you craft a meaningful narrative of your life story. "
        "Let's begin this journey of self-discovery and storytelling."
    )
    
    # Update the state
    updated_state = state.copy()
    updated_state["agent_introduction_status"] = True
    
    # Prepare messages
    new_messages = updated_state["messages"] + [
        SystemMessage(content="Agent is introducing itself as a virtual biographer."),
        AIMessage(content=introduction)
    ]
    
    return {
        "state": updated_state,
        "messages": new_messages
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
            "topics_covered": topics_covered
        }
    else:
        # No more questions
        closing_message = "Thank you for sharing so much about yourself. I feel like I know you better now."
        return {
            "messages": state["messages"] + [AIMessage(content=closing_message)],
            "profile_history": profile_history,
            "topics_covered": topics_covered
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

Use the following guidelines:
1. If the answer is brief, deflective, or includes language that suggests the subject wants to move on (e.g., "That's all," "I don't know," "It's not important"), respond with 'Complete.'
2. If the answer opens up an interesting or unexplored area and the subject shows engagement or enthusiasm, respond with 'Follow Up.'
3. Avoid over-prompting follow-ups when the subject appears uninterested or finished with the topic.

Based on these guidelines, respond with 'Follow Up' or 'Complete.'
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
    logger.info(f"Current state keys: {list(state.keys())}")
    logger.info(f"TESTING_MODE is set to: {TESTING_MODE}")

    updated_state = state.copy()  # Create a copy of the state to modify

    if TESTING_MODE:
        # Load sample interview questions for testing
        from my_agent.sample_interview import sample_interview
        updated_state["interview_questions"] = sample_interview
        logger.info(f"Loaded {len(sample_interview)} sample interview questions for testing.")
        messages = updated_state.get("messages", []) + [SystemMessage(content="Sample interview questions loaded for testing.")]
    else:
        profile_history = updated_state.get("profile_history", [])
        if not profile_history:
            logger.warning("No profile history found.")
            messages = updated_state.get("messages", []) + [SystemMessage(content="No profile data available.")]
        else:
            # Prepare the demographic profile as a string
            profile_string = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in profile_history])
            
            # Prepare the system prompt
            system_prompt = """You are an expert biographer working on a 6 to 12 chapter biography. You have been provided a complete demographic profile of the subject based on previous question and answer sessions. Prepare about 100-120 interview questions based on that demographic profile. Favor open-ended questions to encourage detailed responses. Be respectful of sensitive topics. Categorize questions into chapters."""
            
            # Prepare the user prompt
            user_prompt = f"""Here is the demographic profile of the subject:
            
    {profile_string}

    Based on this profile, generate 100-120 interview questions categorized into chapters. Please format your response as a JSON array of objects, where each object has a 'chapter' field and a 'questions' field containing an array of questions for that chapter."""
            
            # Generate interview questions using the LLM
            try:
                response = llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                )
                interview_data = json.loads(response.content)
                updated_state["interview_questions"] = interview_data
                logger.info(f"Generated {len(updated_state['interview_questions'])} interview questions across chapters.")
                messages = updated_state.get("messages", []) + [SystemMessage(content="Interview questions prepared.")]
            except Exception as e:
                logger.error(f"Error generating interview questions: {e}")
                updated_state["interview_questions"] = []
                messages = updated_state.get("messages", []) + [SystemMessage(content="Failed to prepare interview questions.")]

    # Log the updated state for debugging
    logger.info(f"Updated state after interview_question_prep_node: {updated_state}")

    return {
        "state": updated_state,
        "messages": messages
    }

def questioning_node(state):
    logger.info(f"questioning_node received state: {state}")
    logger.info(f"interview_questions in questioning_node: {state.get('interview_questions', [])}")

    chapter_index = state.get("current_chapter_index", 0)
    question_index = state.get("current_question_index", 0)
    interview_questions = state.get("interview_questions", [])

    logger.debug(f"chapter_index: {chapter_index}")
    logger.debug(f"question_index: {question_index}")
    logger.debug(f"interview_questions: {interview_questions}")

    # This code was messing up the node. Moving the loading of the sample interview to the interview_question_prep_node
    # if TESTING_MODE and not interview_questions:
    #     # remove my_agent from .sample_interview for django
    #     from .sample_interview import sample_interview
    #     state["interview_questions"] = sample_interview
    #     interview_questions = sample_interview
    #     logger.info("Loaded sample interview questions for testing mode.")
    
    if not interview_questions:
        logger.warning("No interview questions available.")
        return {
            "messages": state["messages"] + [SystemMessage(content="No interview questions available.")],
            "awaiting_user_response": False,
            "state": state
        }
    if chapter_index is None:
        chapter_index = 0
        state["current_chapter_index"] = 0

    if chapter_index >= len(interview_questions):
        logger.info("All chapters completed.")
        return {
            "messages": state["messages"] + [SystemMessage(content="Interview completed.")],
            "awaiting_user_response": False,
            "state": state
        }

    current_chapter = interview_questions[chapter_index]
    questions = current_chapter.get("questions", [])

    if question_index >= len(questions):
        logger.info("Moving to next chapter.")
        state["current_chapter_index"] = chapter_index + 1
        state["current_question_index"] = 0
        return questioning_node(state)

    next_question = questions[question_index]
    current_question = {
        "chapter": current_chapter.get("chapter", ""),
        "question": next_question
    }

    state["current_chapter_index"] = chapter_index
    state["current_question_index"] = question_index + 1
    state["current_question"] = current_question
    state["awaiting_user_response"] = True

    return {
        "messages": [AIMessage(content=next_question)],
        "current_question": current_question,
        "awaiting_user_response": True,
        "state": state
    }

# Handle user responses by appending the question and answer to the conversation_history and then move onto the contact check node.
def questioning_response_node(state: UserState) -> Dict[str, Any]:
    logger.info(f"questioning_response_node received state: {state}")
    current_question = state.get("current_question", {})
    
    if state.get("awaiting_user_response", False):
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            user_response = last_message.content

            # Update conversation_history
            conversation_entry = {
                "chapter": current_question.get("chapter", ""),
                "question": current_question.get("question", ""),
                "answer": user_response
            }
            state["conversation_history"].append(conversation_entry)
            logger.info(f"Appended to conversation history: {current_question.get('question')} - {user_response}")

            # Update state
            state["awaiting_user_response"] = False

    updated_state = state.copy()
    
    return {
        "state": updated_state,
        "messages": updated_state["messages"],
    }

def contact_check_node(state: UserState):
    logger.info(f"contact_check_node received state: {state}")
    # Implement any contact checking logic here if needed
    return {
        "messages": state["messages"],
        "state": state,
    }

def contact_validation_node(state: UserState):
    logger.info(f"contact_validation_node received state: {state}")
    return {
        "state": updated_state,
        "messages": new_messages
    }

def contact_store_node(state: UserState) -> Dict[str, Any]:
    logger.info(f"contact_store_node received state: {state}")

    # Update contacts
    updated_contacts = state.get("contacts", []) + ["new_contact"]
    updated_state = state.copy()
    updated_state["contacts"] = updated_contacts

    # Prepare messages
    new_messages = updated_state["messages"] + [
        SystemMessage(content="New contact stored.")
    ]

    return {
        "state": updated_state,
        "messages": new_messages
    }

def reach_validation_node(state: UserState):
    logger.info(f"reach_validation_node received state: {state}")
    return {"messages": state.messages + [SystemMessage(content="Validating reach.")]}

def contact_node(state: UserState):
    logger.info(f"contact_node received state: {state}")
    return {"messages": state.messages + [SystemMessage(content="Contact processed.")]}

def question_validation_node(state: UserState):
    logger.info(f"question_validation_node received state: {state}")

    # Create a copy of the state to update
    updated_state = state.copy()
    
    conversation_history = state["messages"]

    # Safely retrieve the last AI and Human messages
    last_ai_message = next((msg for msg in reversed(conversation_history) if isinstance(msg, AIMessage)), None)
    last_human_message = next((msg for msg in reversed(conversation_history) if isinstance(msg, HumanMessage)), None)

    # Initialize variables
    last_question = last_ai_message.content if last_ai_message else ""
    last_answer = last_human_message.content if last_human_message else ""

    if not last_question:
        logger.info("No last_question found. Proceeding to Chapter Check.")
        updated_state["next_step"] = "chapter_check_node"
        return {
            "state": updated_state,
            "messages": []
        }

    logger.info(f"Last question: {last_question}")
    logger.info(f"Last answer: {last_answer}")

    # Update the interaction history
    if last_question and last_answer:
        updated_state["interaction_history"].append({
            "question": last_question,
            "answer": last_answer
        })
        logger.info(f"Updated interaction history: {updated_state['interaction_history']}")

    # Determine if a follow-up question is appropriate
    prompt = f"""You are an intelligent agent conducting an interview. Based on the latest question and answer:

Question: {last_question}
Answer: {last_answer}

Determine whether a follow-up question would logically extend the conversation and gather more relevant information.

Respond with 'Follow Up' if a follow-up question is appropriate.
Otherwise, respond with 'Proceed to Chapter Check'.

Ensure your response is either 'Follow Up' or 'Proceed to Chapter Check' with no additional text.
"""

    try:
        # Invoke the LLM to make the decision
        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
    except Exception as e:
        logger.error(f"Error during LLM invocation for question validation: {e}")
        decision = "proceed to chapter check"  # Default decision on error

    if "follow up" in decision:
        updated_state["next_step"] = "Question Follow Up"
    else:
        updated_state["next_step"] = "Chapter Check Node"

    logger.info(f"Decision: {decision}, Next step: {updated_state['next_step']}")
    
    # Update follow_up_counts in the state
    current_question = updated_state.get("current_question", {})
    topic = current_question.get('topic', 'default_topic')
    if "follow up" in decision:
        updated_state["follow_up_counts"][topic] = updated_state["follow_up_counts"].get(topic, 0) + 1

    # No new messages to send to the user from this node
    new_messages = []

    return {
        "state": updated_state,
        "messages": new_messages
    }


# No idea what this does. 
# def next_question_step(state):
#     decision = state.get("next_step", "Chapter Check Node")
#     logger.info(f"Next question step decision: {decision}")  # Add this line for debugging
#     return decision

def question_follow_up_node(state: UserState):
    logger.info(f"question_follow_up_node received state: {state}")
    logger.info("Conversation history in question_follow_up_node:")
    for msg in state["messages"]:
        logger.info(f"  {type(msg).__name__}: {msg.content}")

    # Generate a follow-up question using the LLM
    last_interaction = state["interaction_history"][-1] if state["interaction_history"] else None
    if last_interaction:
        last_question = last_interaction["question"]
        last_answer = last_interaction["answer"]
        current_chapter = state["current_question"]["chapter"]
        
        prompt = f"""Based on the previous interaction, generate a thoughtful follow-up question.

Previous Question: {last_question}
User's Answer: {last_answer}
Current Chapter: {current_chapter}

Follow-up Question:
"""
        try:
            follow_up_response = llm.invoke(prompt)
            follow_up_question = follow_up_response.content.strip()
            logger.info(f"Generated follow-up question: {follow_up_question}")
        except Exception as e:
            logger.error(f"Error during LLM invocation for follow-up question: {e}")
            follow_up_question = "Can you tell me more about that?"

        # Update the state with the new question
        updated_state = state.copy()
        updated_state["current_question"] = {
            "chapter": current_chapter,
            "question": follow_up_question
        }
        updated_state["awaiting_user_response"] = True
        updated_state["messages"] = state["messages"] + [AIMessage(content=follow_up_question)]
        
        # Update interaction_history with the follow-up question
        updated_state["interaction_history"].append({
            "question": follow_up_question,
            "answer": ""  # This will be filled in the questioning_response_node
        })
        
        logger.info(f"Updated state: {updated_state}")
        return {
            "state": updated_state,
            "messages": updated_state["messages"]
        }
    else:
        # No previous interaction, proceed to next step
        logger.info("No previous interaction found, proceeding to next step")
        return {
            "state": state,
            "messages": state["messages"]
        }

def chapter_check_node(state: UserState):
    logger.info(f"chapter_check_node received state: {state}")
    
    # Create a copy of the state to update
    updated_state = state.copy()
    
    # Perform any necessary checks or updates here
    # For example:
    # updated_state['chapter_complete'] = check_if_chapter_complete(updated_state)
    
    return {
        "state": updated_state,
        "messages": state['messages']  # or updated_state['messages'] if you've modified messages
    }

def chapter_writer_node(state: UserState) -> Dict[str, Any]:
    logger.info(f"chapter_writer_node received state: {state}")

    # Example: Update state to indicate chapter writing
    updated_state = state.copy()
    updated_state["is_writing_chapter"] = True

    # Prepare messages
    new_messages = updated_state["messages"] + [
        SystemMessage(content="Writing chapter...")
    ]

    return {
        "state": updated_state,
        "messages": new_messages
    }

def chapter_save_node(state: UserState):
    logger.info(f"chapter_save_node received state: {state}")
    return {
        "chapter_complete": True,
        "messages": state.messages + [SystemMessage(content="Saving chapter...")]
    }

def congratulations_node(state: UserState):
    logger.info(f"congratulations_node received state: {state}")
    return {"messages": state.messages + [SystemMessage(content="Congratulations!")]}

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
graph.add_node("Questioning Response Node", questioning_response_node)
graph.add_node("contact_check_node", contact_check_node)
graph.add_node("contact_validation_node", contact_validation_node)
graph.add_node("contact_store_node", contact_store_node)
graph.add_node("reach_validation_node", reach_validation_node)
graph.add_node("contact_node", contact_node)
graph.add_node("Question Validation Node", question_validation_node)
graph.add_node("Question Follow Up", question_follow_up_node)
graph.add_node("Chapter Check Node", chapter_check_node)
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
# graph.add_conditional_edges(
#     "Interview Question Prep",
#     lambda x: "User Status Node",  # Define logic to route back if necessary
#     {
#         "User Status Node": "User Status Node",
#     }
# )

graph.add_edge("Questioning Node", "Questioning Response Node")
graph.add_edge("Questioning Response Node", "contact_check_node")
# graph.add_edge("Questioning Node", "contact_check_node")

graph.add_conditional_edges(
    "contact_check_node",
    lambda x: "contact_validation_node" if "someone" in x["messages"][-1].content else "Question Validation Node",
    {
        "contact_validation_node": "contact_validation_node",
        "Question Validation Node": "Question Validation Node"
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
graph.add_edge("contact_node", "Question Validation Node")

graph.add_conditional_edges(
    "Question Validation Node",
    lambda x: x["state"].get("next_step", "Chapter Check Node"),
    {
        "Question Follow Up": "Question Follow Up",
        "Chapter Check Node": "Chapter Check Node"
    }
)
# logging info for the conditional edge
logger.info(f"Conditional edges added: {graph.edges}")

graph.add_edge("Question Follow Up", "Questioning Response Node")
graph.add_conditional_edges(
    "Chapter Check Node",
    lambda x: "chapter_writer_node" if x.get("chapter_complete") else "Questioning Node",
    {
        "chapter_writer_node": "chapter_writer_node",
        "Questioning Node": "Questioning Node"
    }
)
graph.add_edge("chapter_writer_node", "chapter_save_node")
graph.add_edge("chapter_save_node", "congratulations_node")
graph.add_edge("congratulations_node", END)
graph.add_conditional_edges( # I don't understand why this is needed but the graph doesn't compile without it. I moved it all the way to the end. Make sure to test this once the agent is finalized.
    "congratulations_node",
    lambda x: "User Status Node",  # Define logic to route back if necessary
    {
        "User Status Node": "User Status Node",
    }
)

# Example configuration. Set "start_node" to "Interview Question Prep" to start with interview questions. Set "start_node" to "User Status Node" to start with the user's status.
config = {"start_node": "Interview Question Prep"}


# Set the entry point before compiling
set_entry_point_based_on_config(graph, config)

# Compile the graph with the configuration
app = graph.compile(interrupt_before=["Questioning Response Node"])


def agent_with_state(input_data):
    # Initialize state with default values, including an empty messages list
    default_state = {
        "messages": [],
        "user_profile_status": False,
        "agent_introduction_status": False,
        "profile_info_collected": False,
        "profile_history": [],
        "interaction_history": [],
        "conversation_history": [],
        "current_question": {},
        "current_topic": "",
        "current_chapter_index": 0,
        "current_question_index": 0,
        "topics_covered": set(),
        "follow_up_counts": {},
        "interview_questions": [],
        "awaiting_user_response": False,
        "is_initial_interaction": True
    }
    
    # Update default_state with input_data
    if isinstance(input_data.get("state"), dict):
        default_state.update(input_data["state"])
    elif input_data.get("state") == "state":
        # Handle the case where state is the string "state"
        pass  # You might want to log this or handle it differently
    
    # Keep state as a dictionary
    state = default_state
    
    # Update messages from input_data
    state["messages"] = input_data.get("messages", [])
    
    return {
        "messages": state["messages"],
        "state": state  # Keep state as a dictionary
    }
 
 
def initialize_biography_agent():
     global app
     # Prepare the input data
     input_data = {
         "messages": [],
         "user_profile_status": False,
         "agent_introduction_status": False,
         "profile_info_collected": False,
         "profile_history": [],
         "interaction_history": [],
         "conversation_history": [],
         "current_question": {},
         "current_topic": "",
         "current_chapter_index": 0,
         "current_question_index": 0,
         "topics_covered": set(),
         "follow_up_counts": {},
         "interview_questions": [],
         "awaiting_user_response": False,
         "state": {},
         "is_initial_interaction": True
     }
 
     # Compile the graph and store it in the global variable
     app = graph.compile(interrupt_before=["Questioning Response Node"])
 
     # Invoke the agent with initial input
     final_state = app.invoke(input=input_data)
     logger.info(f"Initial agent state: {final_state}")
 
     return agent_with_state
 
 # Initialize the agent
biography_agent = initialize_biography_agent()
# This worked in langchain studio following code was added for django
    # # Invoke the agent with initial input
    # final_state = compiled_graph.invoke(input=input_data)
    # logger.info(f"Initial agent state: {final_state}")
    # return biography_agent



# This was the original code from the Langgraph studio file.
# # Prepare the input data
# input_data = {
#     "messages": [],
#     "user_profile_status": False,
#     "agent_introduction_status": False,
#     "profile_info_collected": False,
#     "profile_history": [],
#     "interaction_history": [],
#     "conversation_history": [],
#     "current_question": {},
#     "current_topic": "",
#     "current_chapter_index": 0,
#     "current_question_index": 0,
#     "topics_covered": set(),
#     "follow_up_counts": {},
#     "interview_questions": [],          # Added Key
#     "awaiting_user_response": False     # Added Key
# }
# final_state = app.invoke(input=input_data)
import logging
# Create a formatter and set it for the console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the root logger
logging.getLogger().addHandler(console_handler)

logging.debug("Starting the agent script")

# Define the state for the graph
class UserState(TypedDict):
    user_status: Optional[Literal["active"]]  # Can be None (undefined) or "active"
    profile_status: str  # 'blank', 'incomplete', 'complete'
    conversation_history: List[str]  # Store conversation history
    contacts: List[str]  # Store contacts
    biography_chapters: List[str]  # Store biography chapters
    intro_completed: bool  # Add this line to track intro completion

# Define the nodes
def user_status_check(state: UserState) -> dict:
    logging.debug(f"Entering user_status_check with state: {state}")
    if state.get("user_status") is None:
        next_node = "initialize"
    elif state["user_status"] == "active":
        next_node = "agent_intro_check"
    else:
        next_node = "initialize"
    logging.debug(f"Exiting user_status_check with state: {state}")
    return {"next": next_node}  # Remove the user_status from the return value

def initialize(state: UserState) -> dict:
    logging.debug(f"Entering initialize with state: {state}")
    # Initialize user profile and set user status to "active"
    return {
        "next": "agent_intro_check",
        "user_status": "active",
        "conversation_history": [],
        "contacts": [],
        "biography_chapters": [],
        "intro_completed": False,
        "profile_status": "blank"
    }

def agent_intro_check(state: UserState) -> dict:
    logging.debug(f"Entering agent_intro_check with state: {state}")
    next_node = "profile_check" if state.get("intro_completed") else "agent_intro"
    state["intro_completed"] = state.get("intro_completed", False)  # Ensure intro_completed is set
    logging.debug(f"Exiting agent_intro_check with state: {state}")
    return {"next": next_node, **state}

def agent_intro(state: UserState) -> dict:
    logging.debug(f"Entering agent_intro with state: {state}")
    # Introduce the agent and mark intro as completed
    state["intro_completed"] = True
    logging.debug(f"Exiting agent_intro with state: {state}")
    return {
        "next": "profile_check",
        **state
    }

def profile_check(state: UserState) -> dict:
    logging.debug(f"Entering profile_check with state: {state}")
    if state["profile_status"] == "complete":
        next_node = "questioning"
    elif state["profile_status"] == "blank":
        next_node = "profile_intro"
    else:
        next_node = "profile"
    logging.debug(f"Exiting profile_check with state: {state}")
    return {"next": next_node, "profile_status": state["profile_status"]}

def profile_intro(state: UserState) -> dict:
    logging.debug(f"Entering profile_intro with state: {state}")
    # Explain how the agent will ask questions
    state["profile_status"] = "incomplete"
    logging.debug(f"Exiting profile_intro with state: {state}")
    return {"next": "profile", **state}

def profile(state: UserState) -> dict:
    logging.debug(f"Entering profile with state: {state}")
    # Collect user profile information
    state["profile_status"] = "complete"  # Assume profile is now complete
    logging.debug(f"Exiting profile with state: {state}")
    return {"next": "questioning", **state}

def questioning(state: UserState) -> dict:
    logging.debug(f"Entering questioning with state: {state}")
    # Ask questions based on user profile
    state["conversation_history"].append("Thought-provoking question")
    logging.debug(f"Exiting questioning with state: {state}")
    return {"next": "contact_check", **state}

def contact_check(state: UserState) -> dict:
    logging.debug(f"Entering contact_check with state: {state}")
    # Check if a mentioned person is in the contact list
    next_node = "reach_validation" if "mentioned_person" in state["contacts"] else "store_contact"
    logging.debug(f"Exiting contact_check with state: {state}")
    return {"next": next_node, "contacts": state["contacts"]}

def store_contact(state: UserState) -> dict:
    logging.debug(f"Entering store_contact with state: {state}")
    # Store the mentioned person in contacts
    state["contacts"].append("mentioned_person")
    logging.debug(f"Exiting store_contact with state: {state}")
    return {"next": "reach_validation", **state}

def reach_validation(state: UserState) -> dict:
    logging.debug(f"Entering reach_validation with state: {state}")
    # Check if follow-up questions can be asked
    next_node = "follow_up" if True else "chapter_check"
    logging.debug(f"Exiting reach_validation with state: {state}")
    return {"next": next_node, "conversation_history": state["conversation_history"]}

def follow_up(state: UserState) -> dict:
    logging.debug(f"Entering follow_up with state: {state}")
    # Continue the conversation with a probing question
    state["conversation_history"].append("Follow-up question")
    logging.debug(f"Exiting follow_up with state: {state}")
    return {"next": "contact_check", **state}

def chapter_check(state: UserState) -> dict:
    logging.debug(f"Entering chapter_check with state: {state}")
    # Check if enough information has been collected to complete a chapter
    next_node = "chapter_writer" if True else "questioning"
    logging.debug(f"Exiting chapter_check with state: {state}")
    return {"next": next_node, "biography_chapters": state["biography_chapters"]}

def chapter_writer(state: UserState) -> dict:
    logging.debug(f"Entering chapter_writer with state: {state}")
    # Placeholder for chapter writing functionality
    logging.debug(f"Exiting chapter_writer with state: {state}")
    return {"next": END, **state}

# Build the graph
graph = StateGraph(UserState)
graph.add_node("user_status_check", user_status_check)
graph.add_node("initialize", initialize)
graph.add_node("agent_intro_check", agent_intro_check)
graph.add_node("agent_intro", agent_intro)
graph.add_node("profile_check", profile_check)
graph.add_node("profile_intro", profile_intro)
graph.add_node("profile", profile)
graph.add_node("questioning", questioning)
graph.add_node("contact_check", contact_check)
graph.add_node("store_contact", store_contact)
graph.add_node("reach_validation", reach_validation)
graph.add_node("follow_up", follow_up)
graph.add_node("chapter_check", chapter_check)
graph.add_node("chapter_writer", chapter_writer)

# Define the edges
graph.set_entry_point("user_status_check")
graph.add_edge("user_status_check", "initialize")
graph.add_edge("user_status_check", "agent_intro_check")
graph.add_edge("initialize", "user_status_check")
graph.add_edge("agent_intro_check", "profile_check")
graph.add_edge("agent_intro_check", "agent_intro")
graph.add_edge("agent_intro", "profile_check")
graph.add_edge("profile_check", "questioning")
graph.add_edge("profile_check", "profile_intro")
graph.add_edge("profile_check", "profile")  # Add this line
graph.add_edge("profile_intro", "profile")  # Add this line
graph.add_edge("profile", "questioning")
graph.add_edge("questioning", "contact_check")
graph.add_edge("contact_check", "reach_validation")
graph.add_edge("contact_check", "store_contact")
graph.add_edge("store_contact", "reach_validation")
graph.add_edge("reach_validation", "follow_up")
graph.add_edge("reach_validation", "chapter_check")
graph.add_edge("follow_up", "contact_check")
graph.add_edge("chapter_check", "chapter_writer")
graph.add_edge("chapter_writer", END)

# Compile the graph
app = graph.compile()
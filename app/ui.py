# app/ui.py
"""Gradio interface for the Asterline AI Research Assistant.

Two tabs are provided:
1. Research Assistant – chat with the multi‑agent workflow.
2. Memory Management – CRUD operations on per‑user memories.
"""

from __future__ import annotations

import gradio as gr

from app.agents.workflow import run_workflow

from app.memory.memory import MemoryStore

# ---------------------------------------------------------------------------
# Helper functions for the Research Assistant tab
# ---------------------------------------------------------------------------

def answer_question(question: str) -> tuple[str, str, list[str]]:
    """Run the LangGraph workflow and format the result.

    Returns a tuple of (answer, verification_status, citations).
    """
    try:
        state = run_workflow(question)
        answer = state.answer or "I could not find an answer."
        verification = state.verification or "unknown"
        # Build citation strings "[chunk_id] title"
        citations: list[str] = []
        if state.retrieved:
            for chunk in state.retrieved:
                title = chunk.metadata.get("title", "")
                citations.append(f"[{chunk.chunk_id}] {title}")
        return answer, verification, citations
    except Exception:
        # Do not expose internal details to the user
        return "An error occurred while processing your request.", "error", []

# ---------------------------------------------------------------------------
# Helper functions for the Memory Management tab
# ---------------------------------------------------------------------------

def list_memories(user_id: str) -> list[tuple[int, str, str]]:
    """Return a list of memories for *user_id*.
    Each entry is (id, content, timestamp).
    """
    return MemoryStore.list_memories(user_id)


def add_memory(user_id: str, content: str) -> str:
    try:
        MemoryStore.add_memory(user_id, content)
        return "Memory added."
    except Exception:
        return "Failed to add memory."


def update_memory(user_id: str, memory_id: int, new_content: str) -> str:
    try:
        success = MemoryStore.update_memory(user_id, memory_id, new_content)
        return "Memory updated." if success else "Memory not found."
    except Exception:
        return "Failed to update memory."


def delete_memory(user_id: str, memory_id: int) -> str:
    try:
        success = MemoryStore.delete_memory(user_id, memory_id)
        return "Memory deleted." if success else "Memory not found."
    except Exception:
        return "Failed to delete memory."


def delete_all_memories(user_id: str) -> str:
    try:
        count = MemoryStore.delete_all(user_id)
        return f"Deleted {count} memories."
    except Exception:
        return "Failed to delete memories."

# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

def launch_ui() -> None:
    """Create and launch the Gradio app.
    Synthetic user IDs are provided for demonstration purposes.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Asterline AI Research Assistant")
        with gr.Tabs():
            # ------------------- Research Assistant Tab -------------------
            with gr.TabItem("Research Assistant"):
                user_selector = gr.Dropdown(
                    choices=["user_1", "user_2", "user_3"],
                    label="Select User",
                    value="user_1",
                )
                chat_history = gr.Chatbot(label="Conversation")
                txt_input = gr.Textbox(label="Your question", placeholder="Ask a question...")
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear Conversation")

                def submit_message(user, message, history):
                    answer, verification, citations = answer_question(message)
                    # Append user message and assistant response to the chat history
                    history = history + [(message, f"{answer}\n\n**Citations:** {'; '.join(citations)}\n**Verification:** {verification}")]
                    return history, ""

                submit_btn.click(submit_message, inputs=[user_selector, txt_input, chat_history], outputs=[chat_history, txt_input])
                clear_btn.click(lambda: [], None, chat_history)

            # ------------------- Memory Management Tab -------------------
            with gr.TabItem("Memory Management"):
                mem_user = gr.Dropdown(
                    choices=["user_1", "user_2", "user_3"],
                    label="Select User",
                    value="user_1",
                )
                mem_list = gr.Dataframe(
                    headers=["ID", "Content", "Timestamp"],
                    datatype=["number", "str", "str"],
                    label="Your Memories",
                )
                refresh_btn = gr.Button("Refresh")
                add_input = gr.Textbox(label="New memory", placeholder="Enter memory text...")
                add_btn = gr.Button("Add Memory")
                del_id = gr.Number(label="Memory ID to delete", precision=0)
                del_btn = gr.Button("Delete Memory")
                del_all_btn = gr.Button("Delete All Memories")
                status = gr.Label()

                def refresh(user):
                    return list_memories(user)

                def add_mem(user, content):
                    msg = add_memory(user, content)
                    return msg, refresh(user)

                def del_mem(user, mem_id):
                    msg = delete_memory(user, int(mem_id))
                    return msg, refresh(user)

                def del_all(user):
                    msg = delete_all_memories(user)
                    return msg, refresh(user)

                refresh_btn.click(refresh, inputs=mem_user, outputs=mem_list)
                add_btn.click(add_mem, inputs=[mem_user, add_input], outputs=[status, mem_list])
                del_btn.click(del_mem, inputs=[mem_user, del_id], outputs=[status, mem_list])
                del_all_btn.click(del_all, inputs=mem_user, outputs=[status, mem_list])

    demo.launch()

if __name__ == "__main__":
    launch_ui()

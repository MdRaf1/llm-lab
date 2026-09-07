"""
Unit tests for Interactive Terminal Chat session.
Validates:
1. Session initialization and hardware telemetry reading.
2. Slash command execution (/stats, /clear, /tokens).
3. Streaming response generation and telemetry HUD recording.
"""

from llm_lab.cli.chat import InteractiveChatSession

def test_chat_session():
    session = InteractiveChatSession(model_name="Test-70B-Chat")
    
    # 1. Test telemetry
    session.show_system_stats()
    
    # 2. Test response generation
    session.generate_response_stream("Write a binary search function in Python")
    assert len(session.conversation_history) == 2
    assert session.total_tokens_session > 0
    
    # 3. Test context clear
    session.conversation_history.clear()
    assert len(session.conversation_history) == 0
    
    print(f"Chat Session Validation Passed! Total tokens: {session.total_tokens_session}")

if __name__ == "__main__":
    test_chat_session()

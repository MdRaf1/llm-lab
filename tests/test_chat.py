"""
Test for the interactive chat session.

Only `local-llama` generates real text. This test therefore checks the honest
contract: the simulated engines must REFUSE to generate rather than emit canned text
(which is what the previous version of this test asserted as a pass).

The real `local-llama` path is not exercised here — it needs an 4.7 GB GGUF file on
disk and would make the suite depend on a model download.
"""

from llm_lab.cli.chat import InteractiveChatSession


def test_sim_engine_refuses_to_generate():
    session = InteractiveChatSession(mode="sim-frontier-70b")

    # Telemetry reads real process/system memory, so it is allowed to run.
    session.show_system_stats()

    try:
        session.generate_response_stream("Write a binary search function in Python")
    except RuntimeError as exc:
        assert "cannot generate text" in str(exc)
        assert "local-llama" in str(exc)
    else:
        raise AssertionError(
            "sim-frontier-70b generated text; simulated engines must refuse"
        )


def test_unknown_mode_rejected():
    try:
        InteractiveChatSession(mode="frontier-70b")  # the old, misleading name
    except ValueError as exc:
        assert "Unknown engine mode" in str(exc)
    else:
        raise AssertionError("unknown engine mode was silently accepted")


if __name__ == "__main__":
    test_sim_engine_refuses_to_generate()
    test_unknown_mode_rejected()
    print("chat session honesty checks passed")

from session_manager import SessionManager


def test_create_session_returns_unique_ids():
    manager = SessionManager()
    assert manager.create_session() == "session_1"
    assert manager.create_session() == "session_2"


def test_no_history_for_new_session():
    manager = SessionManager()
    session_id = manager.create_session()
    assert manager.get_conversation_history(session_id) is None


def test_history_formats_exchange_in_order():
    manager = SessionManager()
    session_id = manager.create_session()

    manager.add_exchange(session_id, "What is lesson 1?", "It's an introduction.")

    history = manager.get_conversation_history(session_id)
    assert history == "User: What is lesson 1?\nAssistant: It's an introduction."


def test_history_trims_to_max_history_exchanges():
    manager = SessionManager(max_history=1)
    session_id = manager.create_session()

    manager.add_exchange(session_id, "Q1", "A1")
    manager.add_exchange(session_id, "Q2", "A2")

    history = manager.get_conversation_history(session_id)
    assert "Q1" not in history
    assert "Q2" in history and "A2" in history


def test_clear_session_empties_history():
    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_exchange(session_id, "Q1", "A1")

    manager.clear_session(session_id)

    assert manager.get_conversation_history(session_id) is None

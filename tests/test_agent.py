from agent import execute_tool, get_price


def test_get_price_works():
    result = get_price("AAPL")
    assert result == {"ticker": "AAPL", "price": 231.5}

    result = get_price("MSFT")
    assert result == {"ticker": "MSFT", "price": 420.1}

    # Test unknown ticker
    result = get_price("UNKNOWN")
    assert result == {"ticker": "UNKNOWN", "price": "unknown"}


def test_execute_tool_success():
    """Test successful tool execution"""
    result = execute_tool("get_price", {"ticker": "AAPL"})
    assert result == {"ticker": "AAPL", "price": 231.5}

    result = execute_tool("calculate", {"a": 100, "b": 50, "op": "add"})
    assert result == {"result": 150.0}


def test_execute_tool_errors():
    """Test error handling in tool execution"""
    # Unknown tool
    result = execute_tool("fake_tool", {})
    assert "error" in result
    assert "Unknown tool" in result["error"]

    # Invalid arguments
    result = execute_tool("get_price", {"ticker": 123})
    assert "error" in result

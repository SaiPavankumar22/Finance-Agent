from eval.metrics import Prediction, expected_first_call, score_prediction, summarize


def row(tool="get_stock_quote"):
    return {
        "id": "ts_1",
        "intent": "overview",
        "gold_tools": [tool, "finish"],
        "tickers": ["AAPL"],
        "period": "1mo",
        "expect_abstain": False,
    }


def test_expected_first_call_includes_history_period():
    tool, args = expected_first_call(row("get_historical_prices"))
    assert tool == "get_historical_prices"
    assert args == {"ticker": "AAPL", "period": "1mo"}


def test_score_prediction_requires_exact_tool_and_arguments():
    result = score_prediction(
        row(),
        Prediction(
            '{"tool":"get_stock_quote","args":{"ticker":"AAPL"}}',
            "get_stock_quote",
            {"ticker": "AAPL"},
            True,
        ),
    )
    assert result["tool_correct"] is True
    assert result["args_correct"] is True
    assert result["schema_valid"] is True


def test_summary_reports_rates():
    scored = [
        {
            "intent": "overview",
            "schema_valid": True,
            "tool_correct": True,
            "args_correct": True,
            "abstention_correct": True,
            "predicted_tool": "get_stock_quote",
        }
    ]
    summary = summarize(scored)
    assert summary["examples"] == 1
    assert summary["tool_selection_accuracy"] == 1.0

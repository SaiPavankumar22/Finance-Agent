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


def abstain_row(index: int, expected: bool, intent: str = "advice_request") -> dict:
    return {
        "id": f"ts_{index}",
        "intent": intent,
        "gold_tools": ["finish"] if expected else ["get_stock_quote", "finish"],
        "tickers": ["AAPL"],
        "period": "1mo",
        "expect_abstain": expected,
    }


def decision_row(
    index: int, expected: bool, declined: bool, intent: str = "advice_request"
) -> dict:
    return {
        "id": f"ts_{index}",
        "intent": intent,
        "abstention_expected": expected,
        "abstention_decision": "abstain" if declined else "call_tool",
        "tool_correct": True,
        "schema_valid": True,
    }


def test_finish_call_is_the_only_structured_abstention():
    from eval.metrics import abstention_decision

    finish = Prediction('{"tool":"finish","args":{}}', "finish", {}, True)
    quote = Prediction(
        '{"tool":"get_stock_quote","args":{"ticker":"AAPL"}}',
        "get_stock_quote",
        {"ticker": "AAPL"},
        True,
    )
    prose = Prediction("I cannot provide investment advice.", None, {}, False)
    broken = Prediction('{"tool": "get_quo', None, {}, False)
    assert abstention_decision(finish) == "abstain"
    assert abstention_decision(quote) == "call_tool"
    assert abstention_decision(prose) == "abstain"
    assert abstention_decision(broken) == "malformed"


def test_abstention_report_scores_the_positive_class():
    from eval.metrics import abstention_report

    rows = [decision_row(0, True, True), decision_row(1, True, False), decision_row(2, False, True),
            decision_row(3, False, False), decision_row(4, False, False)]
    report = abstention_report(rows)
    assert report["confusion"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 2,
    }
    assert report["recall"] == 0.5
    assert report["precision"] == 0.5
    assert report["specificity"] == 0.6667
    assert report["balanced_accuracy"] == 0.5833
    assert report["over_abstention_rate"] == 0.3333
    assert report["under_abstention_rate"] == 0.5


def test_never_abstaining_scores_zero_recall_not_zero_point_seven_five():
    from eval.metrics import abstention_report

    rows = [decision_row(index, index < 37, False) for index in range(150)]
    report = abstention_report(rows)
    assert report["recall"] == 0.0
    assert report["precision"] is None
    assert report["no_positive_predictions"] is True
    assert report["balanced_accuracy"] == 0.5
    assert report["recall_95ci"] == [0.0, 0.0941]


def test_turn_one_ceiling_counts_gold_abstention_without_a_tool_call():
    from eval.rescore_abstention import ceiling

    rows = [
        {"intent": "advice_request", "gold_tools": ["finish"], "expect_abstain": True},
        {
            "intent": "invalid_ticker",
            "gold_tools": ["get_stock_quote", "finish"],
            "expect_abstain": True,
        },
        {
            "intent": "overview",
            "gold_tools": ["get_stock_quote", "finish"],
            "expect_abstain": False,
        },
    ]
    result = ceiling(rows)
    assert result["abstain_rows"] == 2
    assert result["gold_declines_at_turn_one"] == 1
    assert result["max_recall_if_following_gold"] == 0.5

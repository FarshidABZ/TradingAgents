import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold. End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation.
                \n\nAlways include open and close position points with take-profit and stop-loss levels. Conclude your response with JSON in the format:\n"
                {{\n  "action": "OPEN" | "CLOSE" | "HOLD",\n  "symbol": "<ticker>",\n  "side": "BUY" | "SELL",\n  "entry_price": <float>,\n  "take_profit": <float>,\n  "stop_loss": <float>,\n  "exit_price": <float>,\n  "confidence": "<low|medium|high>",\n  "reason": "<rationale>"\n}}
                \nIf opening or closing, include entry_price or exit_price respectively. Ensure take_profit is at least 1.5 times the stop_loss distance and stop_loss is below purchase for BUY trades and above for SELL trades. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situations you traded in and the lessons learned: {past_memory_str}"""
            },
            context,
        ]

        result = llm.invoke(messages)

        def extract_json_block(text: str):
            """Extract JSON object from text if present."""
            start = text.find("{")
            if start == -1:
                return None
            for end in range(len(text) - 1, start - 1, -1):
                if text[end] == "}":
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        continue
            return None

        trade_json = extract_json_block(result.content)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "trade_signal": trade_json,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")

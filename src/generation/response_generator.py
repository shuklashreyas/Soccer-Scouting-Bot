def generate_response(intent, entities, result):
    if intent == "profile":
        player = entities['player'][0]
        stat = entities['stat'][0]
        value = result
        return f"{player} has {value} {stat} this season."
    elif intent == "compare":
        return f"{result[0]} leads over {result[1]} in {entities['stat'][0]}."
    elif intent == "similarity":
        return f"Players similar to {entities['player'][0]}: {', '.join(result)}."
    return "Sorry, I didn’t understand that."

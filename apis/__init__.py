import json as _json


def extract_token_ids(market: dict) -> tuple[str, str]:
    """
    Return (yes_token_id, no_token_id) from a Gamma market dict.

    Gamma API format:
      "clobTokenIds": ["<yes_id>", "<no_id>"]  (JSON string or list)
      "outcomes":     '["Yes", "No"]'           (JSON string or list)

    Falls back to legacy `tokens` array format (CLOB API).
    """
    clob_ids = market.get("clobTokenIds")
    if clob_ids:
        if isinstance(clob_ids, str):
            clob_ids = _json.loads(clob_ids)
        outcomes_raw = market.get("outcomes", '["Yes","No"]')
        if isinstance(outcomes_raw, str):
            outcomes_raw = _json.loads(outcomes_raw)
        outcome_map = {o.lower(): tid for o, tid in zip(outcomes_raw, clob_ids)}
        return outcome_map.get("yes", ""), outcome_map.get("no", "")

    tokens = market.get("tokens", [])
    yes_tok = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), None)
    no_tok  = next((t for t in tokens if t.get("outcome", "").lower() == "no"),  None)
    if yes_tok and no_tok:
        return yes_tok.get("token_id", ""), no_tok.get("token_id", "")

    return "", ""

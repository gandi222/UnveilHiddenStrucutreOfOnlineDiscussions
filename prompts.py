"""
Prompt builders for each classification strategy.

Strategy A — binary:      only predicts attack            (pred_attack)
Strategy B — two-class:   predicts attack or support      (pred_attack, pred_support)
Strategy C — three-class: predicts attack, support, neither (pred_attack, pred_support, pred_neither)
"""


def _format_pairs(pairs: list) -> str:
    lines = []
    for i, (arg1, arg2) in enumerate(pairs, start=1):
        lines.append(f"Pair {i}:\nArg1: {arg1}\nArg2: {arg2}")
    return "\n\n".join(lines)


def prompt_a(pairs: list) -> str:
    # Binary — only predicts whether arg2 attacks arg1.
    # Explicit label definition reduces ambiguity; exact count {n} tells the model
    # how many objects to output; both valid JSON objects are shown so the model
    # doesn't pattern-match on a single example.
    n = len(pairs)
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"whether Arg2 attacks Arg1 based on the definition below.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n\n"
        f"For each pair, output 1 if Arg2 attacks Arg1, or 0 if it does not.\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must be either {{\"attack\": 0}} or {{\"attack\": 1}}.\n"
    )


def prompt_b(pairs: list) -> str:
    # Two-class — predicts attack or support.
    # Schema shows all zeros as a key-structure template; "exactly one must be 1"
    # tells the model what to fill in, avoiding confusion from a fixed example.
    n = len(pairs)
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"the relation between them as either \"Support\" or \"Attack\" based on the definitions below.\n"
        f"'Support': Arg2 is in favour of or agrees with Arg1.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n\n"
        f"For each pair, set the matching field to 1 and the other to 0.\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must follow this schema: {{\"attack\": 0, \"support\": 0}}\n"
        f"Exactly one field per object must be 1."
    )


def prompt_c(pairs: list) -> str:
    # Three-class — predicts attack, support, or neither.
    # Same schema/count design as B; adds "No Relation" class with its own definition.
    n = len(pairs)
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"the relation between them as either \"Support\", \"Attack\", or \"No Relation\" based on the definitions below.\n"
        f"'Support': Arg2 is in favour of or agrees with Arg1.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n"
        f"'No Relation': Arg2 has no meaningful relation to Arg1.\n\n"
        f"For each pair, set the matching field to 1 and all others to 0.\n\n"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must follow this schema: {{\"attack\": 0, \"support\": 0, \"neither\": 0}}\n"
        f"Exactly one field per object must be 1."
    )


STRATEGIES = {"A": prompt_a, "B": prompt_b, "C": prompt_c}

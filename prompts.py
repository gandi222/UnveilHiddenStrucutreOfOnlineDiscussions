"""
Prompt builders for each classification strategy.

Strategy A — binary:      only predicts attack            (pred_attack)
Strategy B — two-class:   predicts attack or support      (pred_attack, pred_support)
Strategy C — three-class: predicts attack, support, neither (pred_attack, pred_support, pred_neither)

Each prompt function accepts an optional `few_shot` argument: a list of
(arg1, arg2, support_int) tuples drawn from the dataset. When provided, labeled
examples are inserted between the task description and the pairs to classify.
"""


def _format_pairs(pairs: list) -> str:
    lines = []
    for i, (arg1, arg2) in enumerate(pairs, start=1):
        lines.append(f"Pair {i}:\nArg1: {arg1}\nArg2: {arg2}")
    return "\n\n".join(lines)


def _answer_a(support: int) -> str:
    return '{"attack": 1}' if support == 0 else '{"attack": 0}'


def _answer_b(support: int) -> str:
    if support == 0:
        return '{"attack": 1, "support": 0}'
    return '{"attack": 0, "support": 1}'


def _answer_c(support: int) -> str:
    if support == 0:
        return '{"attack": 1, "support": 0, "neither": 0}'
    if support == 1:
        return '{"attack": 0, "support": 1, "neither": 0}'
    return '{"attack": 0, "support": 0, "neither": 1}'


def _format_few_shot(examples: list, answer_fn) -> str:
    """Format labeled examples as a block to insert into the prompt."""
    lines = [f"Here are {len(examples)} labeled example(s) to guide your classification:\n"]
    for i, (arg1, arg2, support) in enumerate(examples, start=1):
        lines.append(
            f"Example {i}:\nArg1: {arg1}\nArg2: {arg2}\nLabel: {answer_fn(support)}"
        )
    lines.append("Now classify the following pairs:\n")
    return "\n\n".join(lines)


def prompt_a(pairs: list, few_shot: list = None) -> str:
    # Binary — only predicts whether arg2 attacks arg1.
    # Explicit label definition reduces ambiguity; exact count {n} tells the model
    # how many objects to output; both valid JSON objects are shown so the model
    # doesn't pattern-match on a single example.
    n = len(pairs)
    few_shot_block = (
        f"\n{_format_few_shot(few_shot, _answer_a)}\n" if few_shot else "\n"
    )
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"whether Arg2 attacks Arg1 based on the definition below.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n\n"
        f"For each pair, output 1 if Arg2 attacks Arg1, or 0 if it does not."
        f"{few_shot_block}"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must be either {{\"attack\": 0}} or {{\"attack\": 1}}.\n"
    )


def prompt_b(pairs: list, few_shot: list = None) -> str:
    # Two-class — predicts attack or support.
    # Schema shows all zeros as a key-structure template; "exactly one must be 1"
    # tells the model what to fill in, avoiding confusion from a fixed example.
    n = len(pairs)
    few_shot_block = (
        f"\n{_format_few_shot(few_shot, _answer_b)}\n" if few_shot else "\n"
    )
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"the relation between them as either \"Support\" or \"Attack\" based on the definitions below.\n"
        f"'Support': Arg2 is in favour of or agrees with Arg1.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n\n"
        f"For each pair, set the matching field to 1 and the other to 0."
        f"{few_shot_block}"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must follow this schema: {{\"attack\": 0, \"support\": 0}}\n"
        f"Exactly one field per object must be 1."
    )


def prompt_c(pairs: list, few_shot: list = None) -> str:
    # Three-class — predicts attack, support, or neither.
    # Same schema/count design as B; adds "No Relation" class with its own definition.
    n = len(pairs)
    few_shot_block = (
        f"\n{_format_few_shot(few_shot, _answer_c)}\n" if few_shot else "\n"
    )
    return (
        f"In this task, you will be given two arguments and your goal is to classify "
        f"the relation between them as either \"Support\", \"Attack\", or \"No Relation\" based on the definitions below.\n"
        f"'Support': Arg2 is in favour of or agrees with Arg1.\n"
        f"'Attack': Arg2 contradicts or opposes Arg1.\n"
        f"'No Relation': Arg2 has no meaningful relation to Arg1.\n\n"
        f"For each pair, set the matching field to 1 and all others to 0."
        f"{few_shot_block}"
        f"{_format_pairs(pairs)}\n\n"
        f"Respond with ONLY a JSON array of exactly {n} objects, one per pair, in order.\n"
        f"Each object must follow this schema: {{\"attack\": 0, \"support\": 0, \"neither\": 0}}\n"
        f"Exactly one field per object must be 1."
    )


STRATEGIES = {"A": prompt_a, "B": prompt_b, "C": prompt_c}

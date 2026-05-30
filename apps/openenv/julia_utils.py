# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Julia task-specific utilities using GenericAction.

This version uses OpenEnv's GenericAction (a simple dict wrapper) instead of
the environment-specific JuliaAction class. This means you don't need to
install the julia_env package locally.

Usage:
    # In your YAML config:
    task:
      env_name: "julia"
      build_action: !function apps.openenv.julia_utils.build_julia_action
      evaluate_response: !function apps.openenv.julia_utils.evaluate_julia_response
      transform_sample: !function apps.openenv.julia_utils.transform_julia_sample
"""

import re
from typing import Any, Dict

from openenv import GenericAction

from forge.observability.metrics import record_metric, Reduce


def get_julia_system_prompt() -> str:
    """Get system prompt for Julia coding tasks."""
    return """You are a precise and pragmatic Julia programmer.

Write a **single Julia function** that correctly solves the problem described below.

CRITICAL - Julia is NOT Python! Use correct Julia syntax:
- Use `lowercase()` NOT `tolower()`
- Use `uppercase()` NOT `upper()`
- Use `reverse()` NOT `rev()` or `reversed()`
- Use `parse(Int, x)` or `Int(x)` for type conversion, NOT `int(x)`
- Use `string()` for string conversion, NOT `str()`
- Use `filter()` NOT `subset()`
- Use `length()` NOT `len()`
- Use `push!()` to append to arrays, NOT `append()`
- String indexing: `str[i]` returns a Char, use `str[i:i]` for single-char String
- Arrays are 1-indexed, NOT 0-indexed
- Use `println()` NOT `print()` for line output
- Use `Dict()` NOT `dict()`
- Boolean operators: `&&` for AND, `||` for OR, `!` for NOT
- Check string contains: `occursin(needle, haystack)` NOT `in` or `contains(haystack, needle)`

Example - Convert string to uppercase and reverse:
```julia
function process_text(text::String)
    upper_text = uppercase(text)  # NOT upper()
    reversed_text = reverse(upper_text)  # NOT rev()
    return reversed_text
end
```

Example - Work with integers and arrays:
```julia
function sum_digits(n::Int)
    total = 0
    digits_arr = Int[]  # Empty array
    while n > 0
        digit = n % 10
        push!(digits_arr, digit)  # NOT append()
        total += digit
        n = div(n, 10)
    end
    return total
end
```

Rules:
- The code must be syntactically correct and runnable as is
- Use only the Julia standard library
- Do **not** wrap the code in a module or add a `main` function
- Do **not** include any test code in your response
- Do **not** hardcode specific test cases or outputs — the function must work for general inputs
- The **function name must exactly match** the one used in the provided tests
- Respond with **only the Julia function** and nothing else (no explanations, no comments, no extra text)
- Character literal should not contain multiple characters
- Take care of object types and mind that spaces matter in Julia

Passing tests and clean, compilable code are rewarded. Hardcoding or failing tests is penalized.

FORMAT YOUR RESPONSE AS:

```julia
function <function_name>(<argument_list>)
    <function_body>
end
```
""".strip()


def build_julia_prompt(sample: Dict[str, Any], tokenizer) -> str:
    """
    Build prompt for Julia code generation.

    Args:
        sample: Dataset sample with 'julia_prompt', 'julia_test', 'first_test_case', 'task_id'
        tokenizer: HuggingFace tokenizer for chat template

    Returns:
        Formatted prompt string ready for model generation
    """
    system_prompt = get_julia_system_prompt()
    request = sample.get("julia_prompt", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]

    formatted_request = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return formatted_request


def build_julia_action(response: str, sample: Dict[str, Any]) -> GenericAction:
    """
    Build GenericAction from model response and dataset sample.

    This uses GenericAction (a simple dict wrapper) instead of JuliaAction,
    so you don't need to install julia_env locally.

    Args:
        response: Model's generated response
        sample: Dataset sample with 'target' field containing test code

    Returns:
        GenericAction instance with core_code and test_code fields
    """
    # Extract code from markdown if present
    code = extract_julia_code(response)

    # Get test code from dataset
    test_code = sample.get("target", "")

    # GenericAction is just a dict wrapper - same fields as JuliaAction
    return GenericAction(
        core_code=code,
        test_code=test_code,
    )


def evaluate_julia_response(result, response: str, sample: Dict[str, Any]) -> float:
    """
    Evaluate Julia code execution result and return reward.

    Works with both typed observations (JuliaObservation) and raw dicts
    returned by GenericEnvClient.

    Args:
        result: StepResult from environment execution
        response: Model's response (for logging)
        sample: Dataset sample (for logging)

    Returns:
        Reward score (0.0 to 1.0)
    """
    try:
        print("=" * 80)
        print("RAW RESPONSE FROM MODEL:")
        print("-" * 80)
        print(response)
        print("-" * 80)

        # Extract code for validation
        code = extract_julia_code(response)

        if not code:
            print("No Julia code extracted - Reward: 0.0")
            print("=" * 80)
            record_metric("reward/julia/no_code_extracted", 1, Reduce.SUM)
            return 0.0

        # Get test code from sample for logging
        test_code = sample.get("target", "")

        # Log both code and test_code together for extraction
        print("EXTRACTED JULIA CODE:")
        print("-" * 80)
        print(code)
        print("-" * 80)
        print("TEST CODE:")
        print("-" * 80)
        print(test_code)
        print("-" * 80)
        print("END OF SAMPLE")
        print("=" * 80)

        # Validate for common Python-like syntax errors
        is_valid, validation_warnings = validate_julia_syntax(code)
        if not is_valid:
            print("SYNTAX VALIDATION WARNINGS:")
            for warning in validation_warnings:
                print(f"  {warning}")
            print("-" * 80)
            record_metric("reward/julia/syntax_warnings", len(validation_warnings), Reduce.SUM)

        # Extract reward from result
        reward = result.reward if result.reward is not None else 0.0

        # Handle both typed observation and dict observation (from GenericEnvClient)
        obs = result.observation
        if isinstance(obs, dict):
            # GenericEnvClient returns dicts
            passed = obs.get("tests_passed", 0)
            failed = obs.get("tests_failed", 0)
            exit_code = obs.get("exit_code", -1)
            code_compiles = obs.get("code_compiles", False)
            stderr = obs.get("stderr", "")
            stdout = obs.get("stdout", "")
        else:
            # Typed observation (JuliaObservation)
            passed = obs.tests_passed
            failed = obs.tests_failed
            exit_code = obs.exit_code
            code_compiles = obs.code_compiles
            stderr = obs.stderr
            stdout = obs.stdout

        total = passed + failed

        # Log execution details
        print("JuliaEnv Execution Result:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Tests Passed: {passed}")
        print(f"  Tests Failed: {failed}")
        print(f"  Total Tests: {total}")
        print(f"  Exit Code: {exit_code}")
        print(f"  Code Compiles: {code_compiles}")

        if stderr:
            print(f"  Stderr: {stderr[:500]}")
            record_metric("reward/julia/has_errors", 1, Reduce.SUM)

        if stdout:
            print(f"  Stdout (first 200 chars): {stdout[:200]}")

        # Log metrics
        pass_rate = passed / total if total > 0 else 0.0
        record_metric("reward/julia/pass_rate", pass_rate, Reduce.MEAN)

        print(f"Final Reward: {reward:.3f}")
        print("=" * 80)

        return reward

    except Exception as e:
        print(f"✗ Error evaluating response: {e} - Reward: 0.0")
        print("=" * 80)
        record_metric("reward/julia/evaluation_errors", 1, Reduce.SUM)
        return 0.0


def extract_julia_code(response: str) -> str:
    """
    Extract Julia code from markdown code blocks.

    Args:
        response: Model's response text

    Returns:
        Extracted Julia code
    """
    text = re.sub(r"^```julia\s*\n?", "", response, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def validate_julia_syntax(code: str) -> tuple[bool, list[str]]:
    """
    Validate Julia code for common Python-like syntax errors.

    Args:
        code: Julia code string to validate

    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []

    python_functions = {
        r'\btolower\(': 'tolower() -> use lowercase()',
        r'\bupper\(': 'upper() -> use uppercase()',
        r'\brev\(': 'rev() -> use reverse()',
        r'\bint\(': 'int() -> use parse(Int, x) or Int(x)',
        r'\bstr\(': 'str() -> use string()',
        r'\blen\(': 'len() -> use length()',
        r'\bsubset\(': 'subset() -> use filter()',
        r'\bappend\(': 'append() -> use push!()',
        r'\bdict\(': 'dict() -> use Dict()',
        r'\breversed\(': 'reversed() -> use reverse()',
        r'\.append\(': '.append() -> use push!()',
        r'\.lower\(': '.lower() -> use lowercase()',
        r'\.upper\(': '.upper() -> use uppercase()',
    }

    for pattern, suggestion in python_functions.items():
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(f"⚠ Found Python-like syntax: {suggestion}")

    if re.search(r'\[\s*0\s*\]', code):
        warnings.append("⚠ Found [0] indexing - Julia arrays are 1-indexed")

    if 'function' in code and not re.search(r'\bend\b', code):
        warnings.append("⚠ Function missing 'end' keyword")

    is_valid = len(warnings) == 0
    return is_valid, warnings


def transform_julia_sample(sample: Dict[str, Any], tokenizer) -> Dict[str, Any] | None:
    """
    Transform raw dataset sample into training format.

    Args:
        sample: Raw dataset sample
        tokenizer: HuggingFace tokenizer

    Returns:
        Transformed sample with 'request', 'target', 'task_id' or None if invalid
    """
    if not sample.get("julia_test") or not sample.get("first_test_case"):
        if not hasattr(transform_julia_sample, "_warned"):
            print(
                f"WARNING: Sample rejected - missing 'julia_test' or 'first_test_case' field. Sample keys: {list(sample.keys())}"
            )
            transform_julia_sample._warned = True
        return None

    formatted_request = build_julia_prompt(sample, tokenizer)

    return {
        "request": formatted_request,
        "target": sample.get("julia_test", ""),
        "task_id": sample.get("task_id", ""),
    }

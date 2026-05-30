# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python coding task-specific utilities using GenericAction.

This version uses OpenEnv's GenericAction (a simple dict wrapper) instead of
the environment-specific CodeAction class. This means you don't need to
install the coding_env package locally.

Usage:
    # In your YAML config:
    task:
      env_name: "coding"
      build_action: !function apps.openenv.python_utils.build_python_action
      evaluate_response: !function apps.openenv.python_utils.evaluate_python_response
      transform_sample: !function apps.openenv.python_utils.transform_python_sample
"""

import re
from typing import Any, Dict

from openenv import GenericAction

from forge.observability.metrics import record_metric, Reduce


def get_python_system_prompt() -> str:
    """Get system prompt for Python coding tasks."""
    return """You are an expert Python programmer.

Write a Python function that correctly solves the problem described below.

Rules:
- The code must be syntactically correct and runnable
- Use proper Python conventions and best practices
- Include necessary imports
- Do not include test code in your response
- Return only the Python code

FORMAT YOUR RESPONSE AS:

```python
def function_name(args):
    # implementation
    return result
```
""".strip()


def build_python_prompt(sample: Dict[str, Any], tokenizer) -> str:
    """
    Build prompt for Python code generation.

    Args:
        sample: Dataset sample with 'prompt' field (e.g., from HumanEval)
        tokenizer: HuggingFace tokenizer for chat template

    Returns:
        Formatted prompt string ready for model generation
    """
    system_prompt = get_python_system_prompt()

    # Support both HumanEval and AceCode formats
    request = sample.get("prompt") or sample.get("question", "")

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


def build_python_action(response: str, sample: Dict[str, Any]) -> GenericAction:
    """
    Build GenericAction from model response and dataset sample.

    This uses GenericAction (a simple dict wrapper) instead of CodeAction,
    so you don't need to install coding_env locally.

    The coding environment only accepts GenericAction(code=...), so we combine
    the model's generated code with the test code into a single code string.

    Args:
        response: Model's generated response
        sample: Dataset sample with test information

    Returns:
        GenericAction instance with combined code (model code + test code)
    """
    # Extract code from markdown if present
    model_code = extract_python_code(response)

    # Get test code if available
    test_code = sample.get("target", "")

    # Combine model code and test code into a single executable script
    # The test code typically contains assertions or function calls that test the model's code
    if test_code:
        combined_code = f"{model_code}\n\n# Test code\n{test_code}"
    else:
        combined_code = model_code

    # GenericAction only accepts 'code' field (maps to CodeAction)
    return GenericAction(code=combined_code)


def evaluate_python_response(result, response: str, sample: Dict[str, Any]) -> float:
    """
    Evaluate Python code execution result and return reward.

    Since the coding environment executes combined code (model code + test code),
    we determine success based on the execution output:
    - exit_code == 0 means all tests passed -> reward = 1.0
    - exit_code != 0 means tests failed or code error -> reward = 0.0

    Works with both typed observations (CodeObservation) and raw dicts
    returned by GenericEnvClient.

    Args:
        result: StepResult from environment execution
        response: Model's response (for logging)
        sample: Dataset sample (for logging)

    Returns:
        Reward score: 1.0 if all tests pass (exit_code == 0), 0.0 otherwise
    """
    try:
        print("=" * 80)
        print("RAW RESPONSE FROM MODEL:")
        print("-" * 80)
        print(response)
        print("-" * 80)

        # Extract code for validation
        code = extract_python_code(response)

        if not code:
            print("No Python code extracted - Reward: 0.0")
            print("=" * 80)
            record_metric("reward/python/no_code_extracted", 1, Reduce.SUM)
            return 0.0

        print("EXTRACTED PYTHON CODE:")
        print("-" * 80)
        print(code)
        print("-" * 80)

        # Handle both typed observation and dict observation (from GenericEnvClient)
        obs = result.observation
        if isinstance(obs, dict):
            # GenericEnvClient returns dicts
            exit_code = obs.get("exit_code", -1)
            stderr = obs.get("stderr", "")
            stdout = obs.get("stdout", "")
        else:
            # Typed observation (CodeObservation)
            exit_code = getattr(obs, "exit_code", -1)
            stderr = getattr(obs, "stderr", "")
            stdout = getattr(obs, "stdout", "")

        # Log execution details
        print("CodingEnv Execution Result:")
        print(f"  Exit Code: {exit_code}")

        if stdout:
            print("  Stdout (first 500 chars):")
            print("-" * 40)
            print(stdout[:500])
            print("-" * 40)

        if stderr:
            print("  Stderr (first 500 chars):")
            print("-" * 40)
            print(stderr[:500])
            print("-" * 40)

        # Compute reward based on exit code
        # exit_code == 0 means the combined code (model code + tests) ran successfully
        # This indicates all assertions passed
        if exit_code == 0:
            reward = 1.0
            record_metric("reward/python/tests_passed", 1, Reduce.SUM)
        else:
            reward = 0.0
            record_metric("reward/python/tests_failed", 1, Reduce.SUM)
            if "AssertionError" in stderr:
                record_metric("reward/python/assertion_errors", 1, Reduce.SUM)
            elif "SyntaxError" in stderr:
                record_metric("reward/python/syntax_errors", 1, Reduce.SUM)
            elif "Error" in stderr or "error" in stderr:
                record_metric("reward/python/other_errors", 1, Reduce.SUM)

        record_metric("reward/python/reward", reward, Reduce.MEAN)

        print(f"Final Reward: {reward:.3f}")
        print("=" * 80)

        return reward

    except Exception as e:
        print(f"✗ Error evaluating response: {e} - Reward: 0.0")
        print("=" * 80)
        record_metric("reward/python/evaluation_errors", 1, Reduce.SUM)
        return 0.0


def extract_python_code(response: str) -> str:
    """
    Extract Python code from markdown code blocks.

    Args:
        response: Model's response text

    Returns:
        Extracted Python code
    """
    # Try to find ```python code block
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    pattern = r"```\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No markdown block, return as-is
    return response.strip()


def transform_python_sample(sample: Dict[str, Any], tokenizer) -> Dict[str, Any] | None:
    """
    Transform raw dataset sample into training format.

    Args:
        sample: Raw dataset sample (e.g., from HumanEval)
        tokenizer: HuggingFace tokenizer

    Returns:
        Transformed sample with 'request', 'target', 'task_id' or None if invalid
    """
    # Validate required fields - support both HumanEval and AceCode formats
    prompt_text = sample.get("prompt") or sample.get("question")
    if not prompt_text:
        # Debug: log why sample was rejected (only for first few)
        if not hasattr(transform_python_sample, '_warned'):
            print(f"WARNING: Sample rejected - missing 'prompt' or 'question' field. Sample keys: {list(sample.keys())}")
            transform_python_sample._warned = True
        return None

    # Build prompt
    formatted_request = build_python_prompt(sample, tokenizer)

    # Get test code - support both formats
    test_code = sample.get("test") or sample.get("test_cases", "")
    if isinstance(test_code, list):
        # AceCode format: list of test cases
        test_code = "\n".join(test_code)

    return {
        "request": formatted_request,
        "target": test_code,  # Test code for reward function
        "task_id": sample.get("task_id") or sample.get("id", ""),
    }

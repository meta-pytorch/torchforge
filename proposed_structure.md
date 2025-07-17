forge/
    apps/
        sft/
            main.py
            example.yaml
        grpo/
            main.py
            example.yaml
        dpo/
            main.py
            example.yaml
        ...
    src/
        cli/
            config.py
            entrypoint.py
            ...
        envs/
            chat.py
            browser.py
            ...
        data/
            tokenizer.py
            message.py
            sft_dataset.py
            packing.py
            iterable.py
            dataset_metrics.py
            ...
        training/
            monarch_utils.py
            logging.py
            profiling.py
            ...
    tests/
    README.md
    ...

Assumptions:
* `uv` for package management (Eli and Ed believe it's ready for primetime)
* Users will work with a fork of the repo
    `uv run python apps/sft/main.py --config=apps/sft/example.yaml`
* Launcher is Python w/ Monarch
   * Can use CLI args or config w/ overrides
* Don't over utilize submodules right now, can always go back and add them later
* Don't over split files right now, can always go back and split them later

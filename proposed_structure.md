forge/
    apps/
        sft/
            sft.py
            example.yaml
        grpo/
            grpo.py
            example.yaml
        dpo/
            dpo.py
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
* Users will work with a fork of the repo
    `pip install -e .`
* Launcher is Python w/ Monarch
   * Can use CLI args or config w/ overrides

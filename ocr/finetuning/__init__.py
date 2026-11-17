"""Fine-tuning the recognizer on this project's own cells.

Needs torch only, and is imported in the VLM's virtualenv by
`tools/finetune_recognizer.py` (training) and `tools/qwen_worker.py` (loading a
tuned model), so nothing here may import the service's other dependencies.
"""

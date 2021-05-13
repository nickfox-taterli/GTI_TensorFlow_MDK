# Addon: Full .model Conversion & Inference
Description: this addon contains example that enable user to convert entire TensorFlow model, including both chip and host layers into a GTI SDK compatible format and perform inference.

## Full Model Definition JSONs
- [CHIP ID]_[MODEL NAME]_fullmodel.json

## Conversion
Run from project root folder:
```python addon/[MODEL NAME]_convert_full.py --help```

## Inference
Run from project root folder:
```python addon/[MODEL NAME]_infer_full.py --help```

# Setup

Have Pipenv make the virtualenv for you:

```
pipenv install
```

# Model: Unique

A TensorFlow model that removes duplicate tensor elements.

This supports strings, and doesn't require custom operators.

```
pipenv run python src/unique_model.py
pipenv run python -m tf2onnx.convert --saved-model models/unique_model --output unique_model.onnx --opset 11
```

# Model: Regex (uses `ort_customops`)

A TensorFlow model that applies a regex, which requires the onnxruntime custom ops in `ort-customops`.

```
pipenv run python src/regex_model.py
pipenv run python -m tf2onnx.convert --saved-model models/regex_model --output regex_model.onnx --extra_opset ai.onnx.contrib:1
```

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

# brew install libomp
# python -m venv .venv
# source .venv/bin/activate
# python ./issue22.py

import onnxruntime # v1.2.0

session = onnxruntime.InferenceSession("model.onnx")

print("Inputs:")
for idx, inputs in enumerate(session.get_inputs()):
    print("idx:", idx)
    print("   Name:", inputs.name)
    print("   Shape:", inputs.shape)
print("Outputs:")
for idx, outputs in enumerate(session.get_outputs()):
    print("idx:", idx)
    print("   Name:", outputs.name)
    print("   Shape:", outputs.shape)

outputs = session.run(None, {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]})[0]
print(outputs.shape)

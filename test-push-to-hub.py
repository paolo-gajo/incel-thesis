import inspect
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
method = model.push_to_hub

source_file = inspect.getsourcefile(method)
source_lines, line_number = inspect.getsourcelines(method)

print(f"Source file: {source_file}")
print(f"Starting at line number: {line_number}")
print("".join(source_lines))

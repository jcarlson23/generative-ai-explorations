from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("gonglinyuan/ast_t5_base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("gonglinyuan/ast_t5_base", trust_remote_code=True)

input_text = r'''def fibonacci(n):
    """return n-th fibonacci number.
    fibonacci[0] = 0
    fibonacci[1] = 1
    """'''
inputs = tokenizer(
    [input_text + "<sen001>"],  # T5-style sentinel token for completion
    max_length=1024,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt"
).input_ids
outputs = model.generate(inputs, max_length=256, do_sample=False)

output_code = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
output_code = output_code[len("<sen001>"):]  # Remove the sentinel token
print(input_text + output_code)

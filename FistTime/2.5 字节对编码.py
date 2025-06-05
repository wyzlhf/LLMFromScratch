import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')
text = (
    "Hello, do you like tea? "
    "In the sunlit terraces"
    "of someunknownPlace."
)
integers=tokenizer.encode(text,allowed_special={" "})
print(integers)
strings=tokenizer.decode(integers)
print(strings)

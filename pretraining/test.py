from transformers import BartTokenizer, BartForCausalLM

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

input_text = ["def 0\ndef 1\nadd 0 1 <e>"]


print (tokenizer.tokenize("def 0\ndef 1\nadd 0 1 <e>"))

# Tokenize
tokens = tokenizer.add_tokens(["<e>"])


print(tokens)
print (tokenizer.tokenize("def 0\ndef 1\nadd 0 1 <e>"))

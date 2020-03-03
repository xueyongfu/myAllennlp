

from allennlp.data.tokenizers import CharacterTokenizer

tokenize = CharacterTokenizer()
a = tokenize.tokenize('how are you')
print(a)


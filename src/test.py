import enchant

text = 'lmaooooo'

dic = enchant.Dict("en-US")
text = ' '.join([dic.suggest(word)[0] if dic.check(word) == False else word for word in text.split()])

print(text)
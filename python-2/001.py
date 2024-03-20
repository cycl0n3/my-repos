def has_rus_in_english(text):
  """
  Checks if a sentence contains potential Russian words written in English alphabets.

  Args:
      text: The string to check.

  Returns:
      True if the sentence might contain Russian written in English alphabets, 
      False otherwise. (This is a possibility, not confirmation)
  """
  replacements = {
      "a": "a", "b": "b", "v": "v", "g": "g", "d": "d",
      "e": "e", "yo": "yo", "zh": "zh", "z": "z", "i": "i",
      "y": "y", "k": "k", "l": "l", "m": "m", "n": "n",
      "o": "o", "p": "p", "r": "r", "s": "s", "t": "t",
      "u": "u", "f": "f", "ch": "ch", "sh": "sh", "shch": "shch",
      "c": "ts", "yu": "yu", "ya": "ya"
  }

  # Remove punctuation and special characters before processing
  allowed_chars = set("abcdefghijklmnopqrstuvwxyz ")
  text = "".join(char for char in text.lower() if char in allowed_chars)
  words = text.split()
  for word in words:
    temp_word = word
    for cyrillic, english in replacements.items():
      temp_word = temp_word.replace(english, cyrillic)
    if len(temp_word) > 2 and 2 <= len(temp_word) <= 15:
      return True
  return False

sentence1 = "Ajayan P.M., Schadler L.S., Braun P.V. Nanocomposite Science and Technology"
sentence2 = "Anur#ev V.I. Spravochnik konstruktora-mashinostroitelja, tom 1"

print(f"Sentence 1: {has_rus_in_english(sentence1)}")
print(f"Sentence 2: {has_rus_in_english(sentence2)}")

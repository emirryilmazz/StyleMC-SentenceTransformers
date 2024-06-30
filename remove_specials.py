import re


def remove_specials(text):
    text = text.lower().replace(' ', '_')
    return re.sub("[^a-zA-Z0-9çÇğĞıİöÖşŞüÜ\$_]", '', text)
print(remove_specials("çiRkin _kadın_"))
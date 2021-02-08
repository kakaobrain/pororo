"""
Copied from https://github.com/em-shea/tones
Copyright (c) em-shea
"""

# Enables print statements useful for debugging
DEBUG_ENABLED = False

# Dictionary with lists of tonal pinyin for each vowel
pinyin = {
    "a": ["ā", "á", "ǎ", "à", "a"],
    "e": ["ē", "é", "ě", "è", "e"],
    "i": ["ī", "í", "ǐ", "ì", "i"],
    "o": ["ō", "ó", "ǒ", "ò", "o"],
    "u": ["ū", "ú", "ǔ", "ù", "u"],
    "ü": ["ǖ", "ǘ", "ǚ", "ǜ", "ü"],
}


# Function to enable/disable debugging print statements
def debug(*args, **kwargs):
    if DEBUG_ENABLED:
        print(*args, **kwargs)


# Function that converts numerical pinyin (ni3) to tone marked pinyin (nǐ)
def convert_from_numerical_pinyin(word):
    finished_word = []

    # Splits word into individual character strings and calls convert_indiv_character for each
    split_word = word.split(" ")
    for indiv_character in split_word:
        try:
            finished_char = convert_indiv_character(indiv_character)
        except:
            continue
        finished_word.append(finished_char)

    # Joins the returned indiv char back into one string
    finished_string = " ".join(finished_word)
    debug("Joined individual characters into finished word:", finished_string)
    return finished_string


# Converts indiv char to tone marked chars
def convert_indiv_character(indiv_character):
    debug("")
    debug("------")
    debug("Starting loop for word:", indiv_character)

    # Convert indiv char string into list of letters
    letter_list = list(indiv_character)

    # Identify v letters, convert to ü
    for index, letter in enumerate(letter_list):
        if letter == "v":
            letter_list[index] = "ü"
            debug("Letter v converted to 'ü' at index:", index)

    # Start an empty counter and list in case of multiple vowels
    counter = 0
    vowels = []

    # Find and count vowels, and use tone mark logic if multiple found
    for index, char in enumerate(letter_list):
        if char in "aeiouü":
            counter = counter + 1
            vowels.append(char)
    debug("Found vowels:", vowels)

    # If multiple vowels are found, use this logic to choose vowel for tone mark
    # a, e, or o takes tone mark - a takes tone in 'ao'
    # else, second vowel takes tone mark
    if counter > 1:
        debug("Found multiple vowels, count:", counter)

        if "a" in vowels:
            tone_vowel = "a"
        elif "o" in vowels:
            tone_vowel = "o"
        elif "e" in vowels:
            tone_vowel = "e"
        else:
            tone_vowel = vowels[1]

        debug("Selected vowel:", tone_vowel)
    elif counter == 0:
        # try:

        # If the character is r5 (儿), remove tone number and return
        if letter_list == ["r", "5"]:
            return "".join(letter_list[:-1])
        else:
            raise ValueError(
                "Invalid numerical pinyin. Input does not contain a vowel.")

    else:
        tone_vowel = vowels[0]
        debug("Only one vowel found:", tone_vowel)

    # Select tone number, which is last item in letter_list
    tone = letter_list[-1]

    # Set integer to use as pinyin dict/list index
    # Select tonal vowel from pinyin dict/list using tone_vowel and tone index
    try:
        tone_int = int(tone) - 1
        tonal_pinyin = pinyin[tone_vowel][tone_int]

    except Exception as e:
        raise ValueError(
            "Invalid numerical pinyin. The last letter must be an integer between 1-5."
        )

    debug("Found tone:", tone)
    debug("Tone vowel converted:", tonal_pinyin)

    # Cal replace_tone_vowel to replace and reformat the string
    return replace_tone_vowel(letter_list, tone_vowel, tonal_pinyin)


def replace_tone_vowel(letter_list, tone_vowel, tonal_pinyin):
    # Replace the tone vowel with tone marked vowel
    letter_list = [w.replace(tone_vowel, tonal_pinyin) for w in letter_list]
    debug("Replaced tone vowel with tone mark:", letter_list)

    # Remove tone number
    tone_number_removed = letter_list[:-1]
    debug("Removed now unnecessary tone number:", tone_number_removed)

    # Reform string
    finished_char = "".join(tone_number_removed)
    debug("Made the letters list into a string:", finished_char)
    return finished_char

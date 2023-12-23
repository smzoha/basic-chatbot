marker_tokens = {
    'pad_token': '<PAD>',
    'start_token': '<SOS>',
    'end_token': '<EOS>',
    'unk_token': '<UNK>'
}


def construct_vocab(data, attrs):
    word_counter = get_word_count(data, attrs)
    vocab = {marker_tokens['pad_token']: 1}
    counter = 2

    for attr in attrs:
        for line in data[attr]:
            for word in line.split():
                if word not in vocab and word_counter[word] > 5:
                    vocab[word] = counter
                    counter += 1

    for token in list(marker_tokens.values())[1:]:
        vocab[token] = counter
        counter += 1

    return vocab


def get_word_count(data, attrs):
    word_counter = {}

    for attr in attrs:
        for line in data[attr]:
            for word in line.split():
                if word not in word_counter:
                    word_counter[word] = 1
                else:
                    word_counter[word] = word_counter[word] + 1

    return word_counter


def get_formatted_answers(answers):
    return [marker_tokens['start_token'] + ' ' + answer + ' ' + marker_tokens['end_token'] for answer in answers]


def get_tokenized_texts(sentences, vocab):
    tokenized_text = []

    for sentence in sentences:
        tokenized_text.append(
            [vocab[word] if word in vocab else vocab[marker_tokens['unk_token']] for word in sentence.split()])

    return tokenized_text

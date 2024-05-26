import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除標點符號
    text = text.lower()  # 轉換為小寫
    text = re.sub(r'\d+', '', text)  # 去除數字
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]  # 去除停用詞
    return ' '.join(words)

def tokenize(text):
    return word_tokenize(text)

def build_vocab(texts):
    counter = Counter()
    for text in texts:
        words = tokenize(text)
        counter.update(words)
    vocab = {word: idx for idx, (word, _) in enumerate(counter.items(), start=1)}
    vocab['<pad>'] = 0  # 填充標誌
    return vocab

def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab['<unk>']) for word in tokenize(text)]

def pad_sequences(sequences, maxlen, pad_value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < maxlen:
            seq = seq + [pad_value] * (maxlen - len(seq))
        else:
            seq = seq[:maxlen]
        padded_sequences.append(seq)
    return padded_sequences

# 假設texts是原始文本數據列表
texts = ["This is an example sentence.", "Here is another one!"]

# 數據清洗
cleaned_texts = [clean_text(text) for text in texts]

# 建立詞典
vocab = build_vocab(cleaned_texts)

# 轉換為數字序列
sequences = [text_to_sequence(text, vocab) for text in cleaned_texts]

# 填充序列
maxlen = 10  # 設定最大序列長度
padded_sequences = pad_sequences(sequences, maxlen)

print(padded_sequences)
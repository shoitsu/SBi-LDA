import numpy as np
from tqdm import tqdm
from collections import Counter
from janome.tokenizer import Tokenizer
from gensim.parsing.preprocessing import STOPWORDS
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud

class SBiLDA:
    def __init__(self, n_user_topics=10, n_item_topics=10, alpha_u=0.1, alpha_m=0.1, beta_u=0.1, beta_m=0.1, gamma=0.1):
        # モデルのパラメータを初期化
        self.n_user_topics = n_user_topics  # ユーザートピックの数
        self.n_item_topics = n_item_topics  # アイテムトピックの数
        self.alpha_u = alpha_u  # ユーザートピックのDirichlet事前分布のパラメータ
        self.alpha_m = alpha_m  # アイテムトピックのDirichlet事前分布のパラメータ
        self.beta_u = beta_u  # ユーザートピック-単語分布のDirichlet事前分布のパラメータ
        self.beta_m = beta_m  # アイテムトピック-単語分布のDirichlet事前分布のパラメータ
        self.gamma = gamma  # スイッチング変数のBeta事前分布のパラメータ
        self.tokenizer = Tokenizer("user_dictionary.csv", udic_enc="utf8")  # 日本語分かち書きのためのトークナイザー
        self.stopwords = set(STOPWORDS)  # ストップワードのセット

    def update_stopwords(self, new_stopwords):
        # ストップワードを追加更新するメソッド
        self.stopwords.update(new_stopwords)

    def preprocess_text(self, text):
        # テキストを前処理するメソッド
        if isinstance(text, str):
            tokens = self.tokenizer.tokenize(text)
            words = [token.surface for token in tokens
                     if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']]
            words = [word for word in words
                     if word not in self.stopwords]
            return words
        return []

    def process_texts(self, texts):
        # 複数のテキストを処理するメソッド
        results = []
        for text in tqdm(texts, desc="Processing records"):
            results.append(self.preprocess_text(text))

        all_words = [word for doc in results for word in doc]
        word_counts = Counter(all_words)

        filtered_results = []
        for doc in results:
            # 出現頻度が2以上の単語のみを保持
            filtered_doc = [word for word in doc if word_counts[word] > 1]
            filtered_results.append(filtered_doc)

        return filtered_results

    def prepare_corpus(self, df):
        # コーパスを準備するメソッド
        texts = self.process_texts(df["review_text"])

        vocabulary = set(word for doc in texts for word in doc)
        self.word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        corpus = [[self.word_to_idx[word] for word in doc] for doc in texts]

        self.n_users = df['user_id'].nunique()
        self.n_items = df['item_id'].nunique()
        self.n_vocab = len(self.word_to_idx)

        return corpus, df['user_id'].tolist(), df['item_id'].tolist()

    def fit(self, df, max_iter=100):
        # モデルを学習するメソッド
        corpus, users, items = self.prepare_corpus(df)

        self.docs = corpus
        self.users = users
        self.items = items

        # カウンタの初期化
        self.nsv = np.zeros((self.n_user_topics, self.n_vocab))
        self.ns = np.zeros(self.n_user_topics)
        self.nus = np.zeros((self.n_users, self.n_user_topics))
        self.nu = np.zeros(self.n_users)
        self.nu0 = np.zeros(self.n_users)
        self.nu1 = np.zeros(self.n_users)

        self.ntv = np.zeros((self.n_item_topics, self.n_vocab))
        self.nt = np.zeros(self.n_item_topics)
        self.nmt = np.zeros((self.n_items, self.n_item_topics))
        self.nm = np.zeros(self.n_items)

        # トピックとスイッチ変数の初期化
        self.z = []
        self.y = []
        for doc, user, item in zip(self.docs, self.users, self.items):
            z_doc = []
            y_doc = []
            for word in doc:
                y = np.random.randint(2)
                if y == 0:
                    z = np.random.randint(self.n_user_topics)
                    self.nsv[z, word] += 1
                    self.ns[z] += 1
                    self.nus[user, z] += 1
                    self.nu0[user] += 1
                else:
                    z = np.random.randint(self.n_item_topics)
                    self.ntv[z, word] += 1
                    self.nt[z] += 1
                    self.nmt[item, z] += 1
                    self.nu1[user] += 1
                self.nu[user] += 1
                self.nm[item] += 1
                z_doc.append(z)
                y_doc.append(y)
            self.z.append(z_doc)
            self.y.append(y_doc)

        # ギブスサンプリング
        for iteration in tqdm(range(max_iter), desc="Gibbs Sampling"):
            for i, (doc, user, item) in enumerate(zip(self.docs, self.users, self.items)):
                for j, word in enumerate(doc):
                    # 現在の割り当てを削除
                    z = self.z[i][j]
                    y = self.y[i][j]
                    if y == 0:
                        self.nsv[z, word] -= 1
                        self.ns[z] -= 1
                        self.nus[user, z] -= 1
                        self.nu0[user] -= 1
                    else:
                        self.ntv[z, word] -= 1
                        self.nt[z] -= 1
                        self.nmt[item, z] -= 1
                        self.nu1[user] -= 1
                    self.nu[user] -= 1
                    self.nm[item] -= 1

                    # 新しい割り当てをサンプリング
                    probs = np.zeros(self.n_user_topics + self.n_item_topics)
                    for s in range(self.n_user_topics):
                        probs[s] = ((self.nsv[s, word] + self.beta_u) / (self.ns[s] + self.n_vocab * self.beta_u)) * \
                                   ((self.nus[user, s] + self.alpha_u) / (self.nu[user] + self.n_user_topics * self.alpha_u)) * \
                                   ((self.nu0[user] + self.gamma) / (self.nu[user] + 2 * self.gamma))
                    for t in range(self.n_item_topics):
                        probs[self.n_user_topics + t] = ((self.ntv[t, word] + self.beta_m) / (self.nt[t] + self.n_vocab * self.beta_m)) * \
                                                        ((self.nmt[item, t] + self.alpha_m) / (self.nm[item] + self.n_item_topics * self.alpha_m)) * \
                                                        ((self.nu1[user] + self.gamma) / (self.nu[user] + 2 * self.gamma))

                    # 正規化
                    probs /= np.sum(probs)

                    # 新しい割り当てを選択
                    new_topic = np.random.choice(self.n_user_topics + self.n_item_topics, p=probs)
                    if new_topic < self.n_user_topics:
                        y = 0
                        z = new_topic
                        self.nsv[z, word] += 1
                        self.ns[z] += 1
                        self.nus[user, z] += 1
                        self.nu0[user] += 1
                    else:
                        y = 1
                        z = new_topic - self.n_user_topics
                        self.ntv[z, word] += 1
                        self.nt[z] += 1
                        self.nmt[item, z] += 1
                        self.nu1[user] += 1
                    self.nu[user] += 1
                    self.nm[item] += 1

                    self.z[i][j] = z
                    self.y[i][j] = y

    def get_user_topic_distribution(self):
        # ユーザーごとのトピック分布を取得
        return (self.nus + self.alpha_u) / (self.nu[:, np.newaxis] + self.n_user_topics * self.alpha_u)

    def get_item_topic_distribution(self):
        # アイテムごとのトピック分布を取得
        return (self.nmt + self.alpha_m) / (self.nm[:, np.newaxis] + self.n_item_topics * self.alpha_m)

    def get_user_topic_word_distribution(self):
        # ユーザートピックごとの単語分布を取得
        return (self.nsv + self.beta_u) / (self.ns[:, np.newaxis] + self.n_vocab * self.beta_u)

    def get_item_topic_word_distribution(self):
        # アイテムトピックごとの単語分布を取得
        return (self.ntv + self.beta_m) / (self.nt[:, np.newaxis] + self.n_vocab * self.beta_m)

    def get_switch_probability(self):
        # スイッチング確率を取得
        return (self.nu0 + self.gamma) / (self.nu + 2 * self.gamma)

    def visualize_topics(self, n_words=10, n_topics=None):
        # トピックを可視化するメソッド
        # 日本語フォントの設定
        japanese_font = 'BIZ UDGothic'
        plt.rcParams['font.family'] = japanese_font

        user_topic_word_dist = self.get_user_topic_word_distribution()
        item_topic_word_dist = self.get_item_topic_word_distribution()

        if n_topics is None:
            n_topics = max(self.n_user_topics, self.n_item_topics)

        for topic_type, topic_word_dist in [("User", user_topic_word_dist), ("Item", item_topic_word_dist)]:
            for i in range(min(n_topics, len(topic_word_dist))):
                topic_words = self.get_top_words(topic_word_dist[i], n_words)

                print(f"{topic_type} Topic {i}:")
                print(", ".join(list(topic_words.keys())[:10]))  # 上位10単語を表示

                self.generate_wordcloud(topic_words, i, f"{topic_type} Topic")
                self.plot_top_words(topic_words, i, f"{topic_type} Topic", n_words)

    def get_top_words(self, topic_word_dist, n=30):
        # トピックの上位n個の単語を取得
        topic_words = {}
        for idx, prob in enumerate(topic_word_dist):
            if idx in self.idx_to_word:
                topic_words[self.idx_to_word[idx]] = prob
        return dict(sorted(topic_words.items(), key=lambda x: x[1], reverse=True)[:n])

    def generate_wordcloud(self, topic_words, topic_id, title):
        # ワードクラウドを生成して表示
        font_path = fm.findfont(fm.FontProperties(family='BIZ UDGothic'))
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              font_path=font_path, regexp=r"[\w']+").generate_from_frequencies(topic_words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{title} {topic_id} - WordCloud')
        plt.show()

    def plot_top_words(self, topic_words, topic_id, title, n_words=10):
        # トピックの上位単語を棒グラフで表示
        words = list(topic_words.keys())[:n_words]
        weights = list(topic_words.values())[:n_words]

        plt.figure(figsize=(10, 5))
        plt.bar(range(n_words), weights, align='center')
        plt.xticks(range(n_words), words, rotation=45, ha='right')
        plt.title(f'{title} {topic_id} - Top {n_words} Words')
        plt.xlabel('Words')
        plt.ylabel('Weight')
        plt.tight_layout()
        plt.show()
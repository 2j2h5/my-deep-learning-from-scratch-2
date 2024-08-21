import sys
sys.path.append("..")
from common.util import preprocess, create_co_matrix, most_similar, ppmi

text = "One sunny afternoon, Sarah decided to visit the park near her house. She packed a small bag with a bottle of water, a book, and some snacks. When she arrived, she found a nice spot under a big oak tree. The sound of birds chirping and children playing filled the air. Sarah took out her book and began to read, enjoying the peaceful atmosphere. After a while, she took a break to eat some snacks and watched the ducks swimming in the pond. It was a perfect way to spend a relaxing afternoon."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

most_similar('tree', word_to_id, id_to_word, W, top=5)
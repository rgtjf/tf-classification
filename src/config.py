category2id = {'entertainment': 0, 'sports': 1, 'car': 2, 'society': 3, 'tech': 4, 'world': 5, 'finance': 6, 'game': 7,
               'travel': 8, 'military': 9, 'history': 10, 'baby': 11, 'fashion': 12, 'food': 13, 'discovery': 14,
               'story': 15, 'regimen': 16, 'essay': 17}

id2category = {index: label for label, index in category2id.items()}


max_sent_len = 30
num_class = 18


ROOT = '../data/nlpcc_data'

DATA_DIR = ROOT + '/word'
train_file = DATA_DIR + '/train.txt'
dev_file = DATA_DIR + '/dev.txt'
test_file = DATA_DIR + '/test.txt'
word_embed_file = ROOT + '/embed/emb_wd/embedding.100'
word_dim = 100

OUTPUT_DIR = '../output'
w2i_file = OUTPUT_DIR + '/w2i.p'
we_file = OUTPUT_DIR + '/we.p'
dev_predict_file = OUTPUT_DIR + '/dev-predicts.txt'
test_predict_file = OUTPUT_DIR + '/test-predicts.txt'

SAVE_DIR = '../save'
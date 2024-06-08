def input_data_en():
    import pandas as pd
    import nltk
    import numpy as np
    from nltk.corpus import stopwords
    from nltk.stem.lancaster import LancasterStemmer
    articles = pd.read_csv(
        'E:\\others\\book  programing\\Arshad danesh knowledge\\Rcommender system\\data\\sentiment\\sentiment\\sentiment_warmup_articles.csv',
        low_memory=False , encoding='latin-1')
    article_list = articles.values.tolist()  # matrix 2d # matrix 2d
    terms = pd.read_csv(
        'E:\\others\\book  programing\\Arshad danesh knowledge\\Rcommender system\\data\\sentiment\\sentiment\\sentiment_warmup_terms.csv',
        low_memory=False, encoding='latin-1')
    terms_list = terms.values.tolist()  # matrix 2d which each word is in one list
    trm = [[] for i in range(0, len(terms_list))]  # matrix 2d
    for i in range(0, len(terms_list)):
        if type(terms_list[i][1]) == str:
            trm[i].append(terms_list[i][1])
            #trm[i].append(terms_list[i][1].decode('unicode_escape').encode('utf-8').decode('utf-8'))  # trm list for decode matrixterm_list

    tag = []  # matrix 3d which tag list elements of trm matrix
    for i in range(0, len(trm)):
        if trm[i] != []:
            tag.append(nltk.pos_tag(trm[i]))
    roles_stop_words = ['RES', 'PUNC', 'NUM', 'POSTP', 'DETe', 'CL', 'P', 'PRO', 'ADV', 'V', 'AJ', 'CONJ', 'Ne', 'DET',
                        'VBE', 'CD', 'VBD', 'NNS', 'PRP', 'IN', 'RB', 'PRO', 'VBG',
                        'V']  # mtrix elements is roles which has in tag matrix ,and must be remove
    row = len(tag)
    i = 0
    while i < row:  # this loop remove stop-words(adv,pp,.... ) from trm list
        a = tag[i][0][1]
        if a in roles_stop_words:
            trm[i] = []
        i += 1

    #from stop_words import get_stop_words
    #en_stop = get_stop_words('en')
    en_stop = stopwords.words('english')

    j = 0
    i = 0
    while j < len(trm):
        trm[j] = [i for i in trm[j] if not i in en_stop]
        j += 1
    stem = [[] for i in range(0, len(trm))]  # this matrix is stem of trm
    st = LancasterStemmer()
    for i in range(0, len(trm)):
        if trm[i] != [] and type(trm[i][0]) == str:
            stem[i].append(st.stem(trm[i][0]))

    j = 0
    while trm[j]:  # this loop is equal to [] must be remove
        if trm[j] == []:
            del trm[j]
        j += 1

    i = 0
    while stem[i]:  # this loop is equal to [] must be remove
        if stem[i] == []:
            del stem[i]
        i += 1

    trm_1d = []  # 1d matrix is term matrix and dont have none value
    for i in range(0, len(trm)):  # this loop is to term_1d
        if trm[i] != [] and trm[i][0] not in trm_1d:
            trm_1d.append(trm[i][0])

    stem_1d = []  # 1d matrix is stem matrix and dont have none value
    for i in range(0, len(stem)):
        if stem[i] != [] and stem[i][0] not in stem_1d:
            stem_1d.append(stem[i][0])

    real_id_terms = []
    for i in range(0, len(terms_list)):
        real_id_terms.append(terms_list[i][0])

    dc_term_2 = [[0 for i in range(len(stem_1d))] for j in range(len(article_list))]  # 2d matrix document-matrix
    corpus = [[] for j in range(len(article_list))]  # 2d matrix document-matrix
    real_id_news = []  # 1d matrix is id of news
    date_news_2 = []  # 1d matrix  is date of news
    for i in range(0, len(article_list)):  # this loop is initialized matix document-terms
        real_id_news.append(article_list[i][0])
        date_news_2.append(article_list[i][1])
        j = 3
        while j < len(article_list[i]):
            if type(article_list[i][j]) == str:
                index = article_list[i][j].find(':')
                id_term = int(article_list[i][j][0:index])
                TF_IDF = int(article_list[i][j][index + 1:index + 3])
                if id_term in real_id_terms:
                    index_term = real_id_terms.index(id_term)
                    if terms_list[index_term][1] in trm_1d:
                        index_trm = trm_1d.index(terms_list[index_term][1])
                        if st.stem(trm_1d[index_trm]) in stem_1d:
                            index_stem = stem_1d.index(st.stem(trm_1d[index_trm]))
                            dc_term_2[i][index_stem] = TF_IDF
                            list_1 = [index_stem]
                            list_1.append(TF_IDF)
                            corpus[i].append(tuple(list_1))
                else:
                    break
            j += 1

    # end of loop

    return trm_1d, stem_1d, stem, date_news_2, real_id_news, corpus, dc_term_2


#trm_1d returns all terms in 1-d matrix
#stem_1d returns all stem in 1-d matrix
#date_news_2  returns all date related to news
#real_id_news returns all id related to news
#corpus is matrix which consist of (ID_NEWS , iD_USER)

def lda_funtion(matrix_document_terms):
    import gensim
    ldamodel = gensim.models.LdaModel(matrix_document_terms, num_topics=40, id2word=dictionary, passes=500)
    Matrix_topic_document = ldamodel.get_document_topics(matrix_document_terms)
    Matrix_topic_terms = ldamodel.get_topic_terms(matrix_document_terms)
    return Matrix_topic_document,Matrix_topic_terms

from TREC_IR import IRLucene
from ExtractiveSummary import ExtractiveSummary
from trec_car.format_runs import *
from orderedset import OrderedSet

from collections import OrderedDict

import sys
import base64
import hashlib

import numpy as np

FILENAMES  = {
    'Random': 'random',
    'Luhn': 'luhn',
    'Edmundson': 'edmundson' ,
    'Text_Rank':'text_rank',
    'Lex_Rank': 'lex_rank',
    'Lsa': 'lsa',
    'Sum_Basic': 'sum_basic',
    'Kl': 'kl' 
}

class TRECExtractiveSummary():

    def __init__(self, dictionary=FILENAMES):
        assert(type(dictionary) == dict)
        self.summary_filenames = dictionary
        self.info_ret = IRLucene()
        self.directory_name_qrels = 'benchmarkTest/outputs/ir_es/'

    def save_relevance_passage_by_ir(self, article_file, paragraph_file, outline_file):
        topics, dict_topics = self.get_sorted_topics(outline_file)
        self.save_passage_article(article_file, paragraph_file, topics)
        self.save_passage_toplevel(article_file, paragraph_file, topics, dict_topics)
        self.save_passage_hierarchical(article_file, paragraph_file, topics, dict_topics)

    def save_relevance_passage_by_summary(self, article_file, paragraph_file, outline_file):
        topics, dict_topics = self.get_sorted_topics(outline_file)
        self.save_passage_article_by_summary(article_file, paragraph_file, topics)
        self.save_passage_toplevel_by_summary(article_file, paragraph_file, topics, dict_topics)
        self.save_passage_hierarchical_by_summary(article_file, paragraph_file, topics, dict_topics)

    def save_relevance_passage_by_ir_and_summary(self, article_file, paragraph_file, outline_file):
        topics, dict_topics = self.get_sorted_topics(outline_file)
        # self.save_passage_toplevel_by_ir_summary(article_file, paragraph_file, topics, dict_topics)
        self.save_passage_hierarchical_by_ir_summary(article_file, paragraph_file, topics, dict_topics)

    def save_passage_article(self, article_file, paragraph_file, topics):
        for topic in topics:
            tuples = self.get_passage_id_tuples(topic, article_file, paragraph_file, 'article')
            self.save_passage_ranking('article.qrels', tuples)
    
    def save_passage_toplevel(self, article_file, paragraph_file, topics, dict_topics):
        for topic in topics:
            section_toplevels = dict_topics.get(topic).get('toplevel')
            for toplevel in section_toplevels:
                tuples = self.get_passage_id_tuples(toplevel, article_file, paragraph_file, 'toplevel')
                self.save_passage_ranking('toplevel.qrels', tuples) 

    def save_passage_hierarchical(self, article_file, paragraph_file, topics, dict_topics):
        for topic in topics:
            section_hierarchical = dict_topics.get(topic).get('hierarchical')
            for hierarchical in section_hierarchical:
                tuples = self.get_passage_id_tuples(hierarchical, article_file, paragraph_file, 'hierarchical')
                self.save_passage_ranking('hierarchical.qrels', tuples)  

    def save_passage_article_by_summary(self, article_file, paragraph_file, topics):
        analyzer, index = self.info_ret.index_articles_paragraphs(article_file, paragraph_file)
        ext_sum = ExtractiveSummary()
        for topic in topics:
            text, array = self.get_text_ids(index, analyzer, topic, 'article')
            dictionary = ext_sum.extract_text(text, 'all') 
            self.save_extractive_summary(dictionary, array, topic)
    
    def save_passage_toplevel_by_summary(self, article_file, paragraph_file, topics, dict_topics):
        analyzer, index = self.info_ret.index_articles_paragraphs(article_file, paragraph_file)
        ext_sum = ExtractiveSummary()
        for topic in topics:
            text, array = self.get_text_ids(index, analyzer, topic, 'article')
            dictionary = ext_sum.extract_text(text, 'all') 
            id_path_dict = self.get_id_path_dictionary(index, analyzer, dict_topics.get(topic).get('toplevel'), 'toplevel')
            self.save_extractive_summary_by_granularity(dictionary, array, id_path_dict, '_toplevel.qrels')

    def save_passage_hierarchical_by_summary(self, article_file, paragraph_file, topics, dict_topics):
        analyzer, index = self.info_ret.index_articles_paragraphs(article_file, paragraph_file)
        ext_sum = ExtractiveSummary()
        for topic in topics:
            text, array = self.get_text_ids(index, analyzer, topic, 'article')
            dictionary = ext_sum.extract_text(text, 'all') 
            id_path_dict = self.get_id_path_dictionary(index, analyzer, dict_topics.get(topic).get('hierarchical'), 'hierarchical')
            self.save_extractive_summary_by_granularity(dictionary, array, id_path_dict, '_hierarchical.qrels')

    def save_passage_toplevel_by_ir_summary(self, article_file, paragraph_file, topics, dict_topics):
        analyzer, index = self.info_ret.index_articles_paragraphs(article_file, paragraph_file)
        ext_sum = ExtractiveSummary()
        for topic in topics:
            section_toplevels = dict_topics.get(topic).get('toplevel')
            for toplevel in section_toplevels:
                id_path_dict = self.get_id_path_dictionary(index, analyzer, [toplevel], 'toplevel')
                text, array = self.get_text_ids(index, analyzer, toplevel, 'toplevel')
                if(len(text) > 0 and len(array) > 0):
                    dictionary = ext_sum.extract_text(text, 'all')
                    self.save_extractive_summary_by_granularity(dictionary, array, id_path_dict, '_toplevel_ir.qrels')
    
    def save_passage_hierarchical_by_ir_summary(self, article_file, paragraph_file, topics, dict_topics):
        analyzer, index = self.info_ret.index_articles_paragraphs(article_file, paragraph_file)
        ext_sum = ExtractiveSummary()
        for topic in topics:
            path_sections = dict_topics.get(topic).get('hierarchical')
            for path in path_sections:
                id_path_dict = self.get_id_path_dictionary(index, analyzer, [path], 'hierarchical')
                text, array = self.get_text_ids(index, analyzer, path, 'hierarchical')
                if(len(text) > 0 and len(array) > 0):
                    dictionary = ext_sum.extract_text(text, 'all')
                    self.save_extractive_summary_by_granularity(dictionary, array, id_path_dict, '_hierarchical_ir.qrels')

    def get_sorted_topics(self, outline_file):
        dict_topics = self.info_ret.get_topics_by_granularity(outline_file)
        topics = list(dict_topics.keys())
        topics.sort()
        return topics, dict_topics

    def get_passage_id_tuples(self, path, article_file, paragraph_file, granularity):
        passage_ids = self.info_ret.retrieve_paragraph_ids(article_file, paragraph_file, granularity, path)
        return tuple(zip([path]* len(passage_ids), passage_ids))

    def get_text_ids(self, index, analyzer, path, granularity):
        query = hashlib.sha256(path.encode()).hexdigest()
        array = np.array(self.info_ret.search_multiple_options(query, index, analyzer, granularity, ['id_paragraph', 'text']))
        text = ''
        if(len(array) > 0):
            texts = [base64.b64decode(base64_text.encode('utf-8')).decode('utf-8') for base64_text in array[:, 1]]
            array = np.array([array[:, 0], np.array(texts)]).transpose()
            text = text.join(texts)
        return text, array

    def get_id_path_dictionary(self, index, analyzer, paths, granularity):
        id_path_dict = {}
        for path in paths:
            encoded_path = hashlib.sha256(path.encode()).hexdigest()
            ids = self.info_ret.search(encoded_path, index, analyzer, granularity, 'id_paragraph')
            for id in ids:
                id_path_dict[id] = path
        return id_path_dict

    def save_extractive_summary(self, dictionary, array, topic):
        for methodName, text in dictionary.items():
            passage_ids = self.match_summary_paragraph_ids(text, array)
            tuples = tuple(zip([topic] * len(passage_ids), passage_ids))
            self.save_passage_ranking(self.summary_filenames.get(methodName)+'_article.qrels', tuples)     

    def save_extractive_summary_by_granularity(self, dictionary, array, id_path_dict, filename):
        for method_name, text in dictionary.items():
            passage_ids = self.match_summary_paragraph_ids(text, array)
            tuples = []
            for passage_id in passage_ids:
                path = id_path_dict.get(passage_id, None)
                if(path is not None):
                    tuples.append([path, passage_id])
            self.save_passage_ranking(self.summary_filenames.get(method_name) + filename, tuples)

    def save_passage_ranking(self, filename, tuples):
        ranking = [(ids[0], ids[1], 1.0 / (r + 1), (r + 1)) for ids, r in zip(tuples, range(0, len(tuples)))]
        # ranking = [(ids[0], ids[1], 1.0, (r + 1)) for ids, r in zip(tuples, range(0, len(tuples)))]
        with open(self.directory_name_qrels + filename, mode='a', encoding='UTF-8') as file:
            writer = file   
            # for query_id, parph_id in tuples:
            #     rankings = [RankingEntry(query_id, parph_id, r, s) for s, r in ranking]
            #     format_run(writer, rankings, exp_name='test')
            rankings = [RankingEntry(query_id, parph_id, r, s) for query_id, parph_id, s, r in ranking]
            format_run(writer, rankings, exp_name='test')
            file.close()   
       
    def match_summary_paragraph_ids(self, text, array):
        paragraphs = text.split('\n')
        ids = []
        for paragraph in paragraphs:
            index = self.search_index(paragraph, array[:, 1])
            if(index >= 0):
                ids.append(array[index][0])
        return list(OrderedSet(ids))

    def search_index(self, paragraph, texts):
        found = False
        i = 0
        while(not found and i < len(texts)):
            found = texts[i].find(paragraph) >= 0
            i += 1
        return i - 1 if found else -1

if __name__ == "__main__":
    if len(sys.argv) < 1 or len(sys.argv) > 5:
        print('usage ', sys.argv[0], ' articleFile paragraphFile outlineFile')
        exit()
    TREC_summary = TRECExtractiveSummary()
    # TREC_summary.save_relevance_passage_by_ir(sys.argv[1], sys.argv[2], sys.argv[3])
    # TREC_summary.save_relevance_passage_by_summary(sys.argv[1], sys.argv[2], sys.argv[3])
    TREC_summary.save_relevance_passage_by_ir_and_summary(sys.argv[1], sys.argv[2], sys.argv[3])
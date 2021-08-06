
import lucene
from org.apache.lucene.index import IndexWriter, DirectoryReader, IndexWriterConfig, IndexOptions
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, FieldType, StoredField
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import RAMDirectory

from trec_car.read_data import *

from orderedset import OrderedSet
import sys
import hashlib
import base64

class IRLucene:

    def __init__(self):
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    def index_articles_paragraphs(self, article_file, paragraph_file):
        analyzer, index = self.index_paragraphs(paragraph_file)
        article_analyzer, article_index = self.index_articles(article_file, index, analyzer)
        return article_analyzer, article_index

    def retrieve_paragraph_ids(self, article_file, paragraph_file, granularity, query):
        article_analyzer, article_index = self.index_articles_paragraphs(article_file, paragraph_file)
        query = hashlib.sha256(query.encode()).hexdigest()
        res = self.search(query, article_index, article_analyzer, granularity, 'id_paragraph')
        return res

    def get_topics_by_granularity(self, outline_file):
        topics = {}
        with open(outline_file, 'rb') as f:
            for p in iter_annotations(f):
                for section_path in p.flat_headings_list():
                    if(not p.page_id in topics):
                        topics[p.page_id] = {'hierarchical': []}
                    path = "/".join([p.page_id]+[section.headingId for section in section_path])
                    topics.get(p.page_id).get('hierarchical').append(path)
                
                for section, _ in p.deep_headings_list():
                    if(not 'toplevel' in topics.get(p.page_id)):
                        topics.get(p.page_id)['toplevel'] = []
                    path = p.page_id + '/' + section.headingId
                    topics.get(p.page_id).get('toplevel').append(path)

        return topics

    def index_paragraphs(self, paragraph_file):
        analyzer = StandardAnalyzer()
        index = RAMDirectory()
        config = IndexWriterConfig(analyzer)
        w = IndexWriter(index, config)
        self.add_paragraph_at_document(w, paragraph_file)
        w.close()
        return analyzer, index

    def add_paragraph_at_document(self, w, paragraph_file):
        with open(paragraph_file, 'rb') as f:
            for p in iter_paragraphs(f):
                id = p.para_id
                texts = [ elem.text if isinstance(elem, ParaText)
                    else elem.anchor_text
                    for elem in p.bodies ]
                doc = Document()
                doc.add(StringField('id', id, Field.Store.YES))
                # field_type = FieldType()
                # field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
                # field_type.setStored(True)
                # field_type.setTokenized(True)
                text = ''.join(texts)
                base64_text = base64.b64encode(text.encode('utf-8')).decode('utf-8') # hashlib.sha256(text.encode()).hexdigest()
                # text_field = Field('text', base64_text, field_type)
                doc.add(StoredField('text', base64_text))
                encoded_text = hashlib.sha256(text.encode()).hexdigest()
                doc.add(StringField('encoded_text', encoded_text, Field.Store.YES))
                # doc.add(text_field)
                w.addDocument(doc)
        
    def index_articles(self, article_file, index_parph, analyzer_parph):
        analyzer = StandardAnalyzer()
        index = RAMDirectory()
        config = IndexWriterConfig(analyzer)
        w = IndexWriter(index, config)
        self.add_articles_at_document(w, article_file, index_parph, analyzer_parph)
        w.close()
        return analyzer, index

    def add_articles_at_document(self, w, article_file, index_parph, analyzer_parph):
        with open(article_file, 'rb') as f:
            for p in iter_pages(f):
                page_id = p.page_id
                all_children = self.get_all_children(p)
                for toplevel_id, hierarchical_id, child in all_children:
                    text = child.get_text()
                    base64_text = base64.b64encode(text.encode('utf-8')).decode('utf-8') 
                    encoded_text = hashlib.sha256(text.encode()).hexdigest()
                    ids = self.search(encoded_text, index_parph, analyzer_parph, 'encoded_text', 'id')
                    toplevel_id = page_id + '/' + toplevel_id
                    hierarchical_id = page_id + '/' + hierarchical_id
                    encoded_toplevel_id = hashlib.sha256(toplevel_id.encode()).hexdigest()
                    encoded_hierarchical_id = hashlib.sha256(hierarchical_id.encode()).hexdigest()
                    encoded_page_id = hashlib.sha256(page_id.encode()).hexdigest()
                    for id in ids:
                        doc = Document()
                        doc.add(StringField('article', encoded_page_id, Field.Store.YES))
                        doc.add(StringField('toplevel', encoded_toplevel_id, Field.Store.YES))
                        doc.add(StringField('hierarchical', encoded_hierarchical_id, Field.Store.YES))
                        doc.add(StringField('id_paragraph', id, Field.Store.YES))
                        doc.add(StoredField('text', base64_text))
                        # field_type = FieldType()
                        # field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
                        # field_type.setStored(True)
                        # field_type.setTokenized(True)
                        # text_field = Field('text', base64_text, field_type)
                        # doc.add(text_field)
                        w.addDocument(doc)

    def search(self, query, index, analyzer, str_in, str_out):
        p = QueryParser(str_in, analyzer)
        q = p.parse(QueryParser.escape(query))
        hitsPerPage = 30
        reader = DirectoryReader.open(index)
        searcher = IndexSearcher(reader)
        docs = searcher.search(q, hitsPerPage)
        hits = docs.scoreDocs
        result = []
        for hit in hits:
            doc_id = hit.doc
            d = searcher.doc(doc_id)
            result.append(d.get(str_out))
        reader.close()
        return list(OrderedSet(result))

    def search_multiple_options(self, query, index, analyzer, str_in, str_outs):
        p = QueryParser(str_in, analyzer)
        q = p.parse(QueryParser.escape(query))
        hitsPerPage = 30
        reader = DirectoryReader.open(index)
        searcher = IndexSearcher(reader)
        docs = searcher.search(q, hitsPerPage)
        hits = docs.scoreDocs
        result = []
        for hit in hits:
            doc_id = hit.doc
            d = searcher.doc(doc_id)
            result.append([ d.get(str_out) for str_out in str_outs ])
        reader.close()
        return result

    def get_all_children(self, page):
        res = []
        if(len(page.outline()) == 0):
            res.append((page.headingId, page.headingId, page))
        else:
            for child in page.outline():
                res.extend(self.get_paragraphs(child, child.headingId, child.headingId))
        return res
    
    def get_paragraphs(self, section, toplevel_id, hierarchical_id):
        res = []
        if(len(section.children) == 0):
            res.append((toplevel_id, hierarchical_id, section))
        else:
            for p in section.children:
                if(isinstance(p, Para)):
                    res.append((toplevel_id, hierarchical_id, p))
                else:
                    res.extend(self.get_paragraphs(p, toplevel_id, hierarchical_id + '/' + p.headingId))
        return res


if __name__ == '__main__':
    if len(sys.argv) < 1 or len(sys.argv) > 5:
        print('usage ', sys.argv[0], ' articleFile paragraphfile  type query')
        exit()
    hL = IRLucene()
    # r = hL.retrieve_paragraph_ids(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    # print(str(list(r)))
    analyzer, index = hL.index_articles_paragraphs(sys.argv[1], sys.argv[2])
    query = hashlib.sha256(sys.argv[4].encode()).hexdigest()
    # l = hL.search('4f413364b490793f121a5a95d48bf83ba3b39b30', index, analyzer, 'id_paragraph', 'toplevel_path')
    # print(str(l))
    l = hL.search(query, index, analyzer, sys.argv[3], 'id_paragraph')
    # analyzer, index = hL.index_paragraphs(sys.argv[2])
    # t = hL.search('4f413364b490793f121a5a95d48bf83ba3b39b30', index, analyzer, 'id', 'text')
    print(str(l))
    # print(base64.b64decode(t[0].encode('utf-8')).decode('utf-8'))
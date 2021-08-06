from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import sys

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.random import RandomSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from trec_car.read_data import *

LANGUAGE = "english"
SENTENCES_COUNT = 10
SUMMARIZER_OPTIONS = {
    'Random': RandomSummarizer,
    'Luhn': LuhnSummarizer,
    'Edmundson': EdmundsonSummarizer,
    'Text_Rank': TextRankSummarizer,
    'Lex_Rank': LexRankSummarizer,
    'Lsa': LsaSummarizer,
    'Sum_Basic': SumBasicSummarizer,
    'Kl': KLSummarizer
}

class ExtractiveSummary:

    def getArticleText(self, file, outputFilename, outputSummaryFilename):
        # newFile = open(outputFilename, 'w')
        # summaryFile = open(outputSummaryFilename, 'w')
        with open(file, 'rb') as f:
            for p in iter_pages(f):
                title = str(p.page_name)
                text = ''
                for section in p.child_sections:
                    if section:
                        text = text + '\n' + section.heading
                        text = text + '\n' + section.get_text_with_headings(False)
                # newFile.write('\n======================' + title + '========================')
                # newFile.write(text)
                self.writeExtractionOfText(title, text, outputSummaryFilename)
                # summaryFile.write('\n===========' + title + '========')
                # summaryFile.write(self.extractText(text, summaryzerName))
        # newFile.close()

    def writeExtractionOfText(self, title, text, filename):
        for tuple in SUMMARIZER_OPTIONS.items():
            # print(tuple[0])
            extractedText = self.extract_summary(text, tuple[1])
            file = open(str(tuple[0] + filename), 'a')
            file.write('\n===========' + title + '============')
            file.write('\n' + extractedText)
            file.close()

    def extract_text(self, text, summarizer_name = 'all'):
        result = {}
        if(summarizer_name == 'all'):
            for item in SUMMARIZER_OPTIONS.items():
                summarizer = self.extract_summary(text, item[1]) 
                result[item[0]] = summarizer
        elif(SUMMARIZER_OPTIONS.has_key(summarizer_name)):
            result[summarizer_name] = self.extract_summary(text, SUMMARIZER_OPTIONS[summarizer_name])
        return result
        
    def extract_summary(self, text, summarizer_class):
        summary_text = ''
        parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
        summarizer = summarizer_class(Stemmer(LANGUAGE))
        stop_words = get_stop_words(LANGUAGE)
        if summarizer_class is EdmundsonSummarizer:
            summarizer.null_words = stop_words
            summarizer.bonus_words = parser.significant_words
            summarizer.stigma_words = parser.stigma_words
        else:
            summarizer.stop_words = stop_words
        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            summary_text = summary_text + '\n' + str(sentence)
        return summary_text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

if __name__ == "__main__":
    if len(sys.argv) < 1 or len(sys.argv) > 4:
        print('usage ', sys.argv[0], ' articleFile outputFilename outputSummaryFilename')
        exit()
    extractor = ExtractiveSummary()
    extractor.main(sys.argv[1], sys.argv[2], sys.argv[3])
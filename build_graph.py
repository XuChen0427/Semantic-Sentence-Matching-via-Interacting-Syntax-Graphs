import nltk
import numpy as np
import os
import sys
import json5
from pprint import pprint
from src.utils import params
from pytorch_transformers import BertTokenizer
import copy
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordPOSTagger
from nltk.internals import find_jars_within_path
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse import CoreNLPParser
from tqdm import tqdm,trange
import json

from nltk.stem import WordNetLemmatizer
from supar import Parser


#nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 1500000 >logv2.out 2>&1 &
class TextGraphBuilder():
    def __init__(self,args):

        self.dependency_parser = CoreNLPDependencyParser(url='http://localhost:9000')
        self.data_dir = args.data_dir

        self.args = args
        self.data = self.load_data()
        self.tokenizer = BertTokenizer.from_pretrained ('modeling_bert/bert-base-uncased-vocab.txt')
        self.postager =  CoreNLPParser(url='http://localhost:9000', tagtype='pos')
        self.FirstPOSRelEncoder = {}
        self.FirstNERRelEncoder = {}
        self.SecondSynRelEncoder = {}
        self.SecondSemRelEncoder = {}
        self.FirstPOSRelCount = 0
        self.FirstNERRelCount = 0
        self.SecondSynRelCount = 0
        self.SecondSemRelCount = 0
        self.nertager =  CoreNLPParser(url='http://localhost:9000', tagtype='ner')
        self.lemmatizer = WordNetLemmatizer()
        self.semantic_parser = Parser.load('sugar_model/dm.biaffine.sdp.roberta')



    def DependencyParse(self,s):
        '''

        :param s:
        :return: 1st-2nd coomatrix, 2nd->3nd coomatrix, word_token1st, rel_token1st, rel_token2rd
        '''
        result, = self.dependency_parser.raw_parse(s.lower())
        input_format = []
        word_token = []
        rel_POS_first_token = []
        rel_Syn_second_token = []
        First2Second_Syn_coomatrx = [[],[]]
        rel_NER_first_token = []
        rel_Sem_second_token = []
        First2Second_Sem_coomatrx = [[],[]]

        parse_list = result.to_conll(4)[:-1].split("\n")
        edge_index = 0
        words = []
        ################Syntactic dependency parse and POS parse###################
        for index,i in enumerate(parse_list):
            word,POS,ids,rel = i.split("\t")
            ##semantic parses' format#########
            input_format.append((word,self.lemmatizer.lemmatize(word),POS))

            POS = POS[:2]
            rel = rel[:2]
            word_token.append(self.tokenizer.convert_tokens_to_ids(word))
            words.append(word)
            if POS not in self.FirstPOSRelEncoder.keys():
                self.FirstPOSRelEncoder[POS] = self.FirstPOSRelCount
                self.FirstPOSRelCount += 1
            rel_POS_first_token.append(self.FirstPOSRelEncoder[POS])
            if rel not in self.SecondSynRelEncoder.keys():
                self.SecondSynRelEncoder[rel] = self.SecondSynRelCount
                self.SecondSynRelCount += 1

            if int(ids) != 0:
                rel_Syn_second_token.append(self.SecondSynRelEncoder[rel])
                #second_rel_list.append((index,int(ids)-1,str(index)+":"+str(int(ids)-1)))
                First2Second_Syn_coomatrx[0].append(index)
                First2Second_Syn_coomatrx[0].append(int(ids)-1)
                First2Second_Syn_coomatrx[1].append(edge_index)
                First2Second_Syn_coomatrx[1].append(edge_index)
                edge_index += 1

        result = self.semantic_parser.predict([input_format],verbose=False)[0]
        values = result.__getattr__("values")

        #print(values)
        words,edges = values[1],values[8]
        edge_index = 0
        for i,word in enumerate(words):

            edge = edges[i]
            if edge != '_':
                edge_list = edge.split("|")
                for j in edge_list:
                    nodeid,edge_type = j.split(":")
                    nodeid = int(nodeid)
                    if nodeid != 0:

                        if edge_type not in self.SecondSemRelEncoder.keys():
                            self.SecondSemRelEncoder[edge_type] = self.SecondSemRelCount
                            self.SecondSemRelCount += 1
                        rel_Sem_second_token.append(self.SecondSemRelEncoder[edge_type])
                        #second_rel_list.append((i,int(nodeid)-1,str(i)+":"+str(int(nodeid)-1)))
                        First2Second_Sem_coomatrx[0].append(i)
                        First2Second_Sem_coomatrx[0].append(int(nodeid)-1)
                        First2Second_Sem_coomatrx[1].append(edge_index)
                        First2Second_Sem_coomatrx[1].append(edge_index)
                        edge_index += 1

        ner_tag = self.nertager.tag(words)
        for i,(word,tag) in enumerate(ner_tag):
            if tag not in self.FirstNERRelEncoder.keys():
                self.FirstNERRelEncoder[tag] = self.FirstNERRelCount
                self.FirstNERRelCount += 1
            rel_NER_first_token.append(self.FirstNERRelEncoder[tag])

        return First2Second_Syn_coomatrx,First2Second_Sem_coomatrx,word_token,rel_POS_first_token,rel_NER_first_token,rel_Syn_second_token,rel_Sem_second_token


    def WriteGraphIntoFile(self):
        #graph_file = open(os.path.join(self.args.graph_dir,self.args.graph_file),"w")
        First2Second_CoomatrixFile = open(os.path.join(self.args.graph_dir,self.args.First2Second_CoomatrixFile),'w')
        #Second2Third_CoomatrixFile = open(os.path.join(self.args.graph_dir,self.args.Second2Third_CoomatrixFile),"w")

        WordFirst_NodeFile = open(os.path.join(self.args.graph_dir,self.args.WordFirst_NodeFile),"w")
        RelFirst_NodeFile = open(os.path.join(self.args.graph_dir,self.args.RelFirst_NodeFile),"w")
        RelSecond_NodeFile = open(os.path.join(self.args.graph_dir,self.args.RelSecond_NodeFile),"w")

        graphs = []
        #index_list = []
        #for id in trange(0,len(self.data)):
        for id in trange(0, 200):

            X,Y,label = self.data[id]['text1'],self.data[id]['text2'],self.data[id]['target']

            #########parsing sentence X.........

            First2Second_Syn_coomatrx,First2Second_Sem_coomatrx,word_token,rel_POS_first_token,rel_NER_first_token,rel_Syn_second_token,rel_Sem_second_token = self.DependencyParse(X)
            for i in range(len(First2Second_Syn_coomatrx[0])):
                First2Second_CoomatrixFile.write(str(First2Second_Syn_coomatrx[0][i]))
                if i != len(First2Second_Syn_coomatrx[0]) - 1:
                    First2Second_CoomatrixFile.write(",")
            First2Second_CoomatrixFile.write(" ")


            for i in range(len(First2Second_Sem_coomatrx[0])):
                First2Second_CoomatrixFile.write(str(First2Second_Sem_coomatrx[0][i]))
                if i != len(First2Second_Sem_coomatrx[0]) - 1:
                    First2Second_CoomatrixFile.write(",")
            First2Second_CoomatrixFile.write("  ")

            for i in range(len(word_token)):
                WordFirst_NodeFile.write(str(word_token[i]))
                if i != len(word_token) - 1:
                    WordFirst_NodeFile.write(",")
            WordFirst_NodeFile.write("  ")

            for i in range(len(rel_POS_first_token)):
                RelFirst_NodeFile.write(str(rel_POS_first_token[i]))
                if i != len(rel_POS_first_token) - 1:
                    RelFirst_NodeFile.write(",")
            RelFirst_NodeFile.write(" ")

            for i in range(len(rel_NER_first_token)):
                RelFirst_NodeFile.write(str(rel_NER_first_token[i]))
                if i != len(rel_NER_first_token) - 1:
                    RelFirst_NodeFile.write(",")
            RelFirst_NodeFile.write("  ")

            for i in range(len(rel_Syn_second_token)):
                RelSecond_NodeFile.write(str(rel_Syn_second_token[i]))
                if i != len(rel_Syn_second_token) - 1:
                    RelSecond_NodeFile.write(",")
            RelSecond_NodeFile.write(" ")

            for i in range(len(rel_Sem_second_token)):
                RelSecond_NodeFile.write(str(rel_Sem_second_token[i]))
                if i != len(rel_Sem_second_token) - 1:
                    RelSecond_NodeFile.write(",")
            RelSecond_NodeFile.write("  ")


            #########parsing sentence Y.........
            First2Second_Syn_coomatrx,First2Second_Sem_coomatrx,word_token,rel_POS_first_token,rel_NER_first_token,rel_Syn_second_token,rel_Sem_second_token = self.DependencyParse(Y)

            for i in range(len(First2Second_Syn_coomatrx[0])):
                First2Second_CoomatrixFile.write(str(First2Second_Syn_coomatrx[0][i]))
                if i != len(First2Second_Syn_coomatrx[0]) - 1:
                    First2Second_CoomatrixFile.write(",")
            First2Second_CoomatrixFile.write(" ")

            for i in range(len(First2Second_Sem_coomatrx[0])):
                First2Second_CoomatrixFile.write(str(First2Second_Sem_coomatrx[0][i]))
                if i != len(First2Second_Sem_coomatrx[0]) - 1:
                    First2Second_CoomatrixFile.write(",")
            First2Second_CoomatrixFile.write("\n")

            for i in range(len(word_token)):
                WordFirst_NodeFile.write(str(word_token[i]))
                if i != len(word_token) - 1:
                    WordFirst_NodeFile.write(",")
            WordFirst_NodeFile.write("  ")
            WordFirst_NodeFile.write(str(label))
            WordFirst_NodeFile.write("\n")

            for i in range(len(rel_POS_first_token)):
                RelFirst_NodeFile.write(str(rel_POS_first_token[i]))
                if i != len(rel_POS_first_token) - 1:
                    RelFirst_NodeFile.write(",")
            RelFirst_NodeFile.write(" ")

            for i in range(len(rel_NER_first_token)):
                RelFirst_NodeFile.write(str(rel_NER_first_token[i]))
                if i != len(rel_NER_first_token) - 1:
                    RelFirst_NodeFile.write(",")
            RelFirst_NodeFile.write("\n")

            for i in range(len(rel_Syn_second_token)):
                RelSecond_NodeFile.write(str(rel_Syn_second_token[i]))
                if i != len(rel_Syn_second_token) - 1:
                    RelSecond_NodeFile.write(",")
            RelSecond_NodeFile.write(" ")

            for i in range(len(rel_Sem_second_token)):
                RelSecond_NodeFile.write(str(rel_Sem_second_token[i]))
                if i != len(rel_Sem_second_token) - 1:
                    RelSecond_NodeFile.write(",")
            RelSecond_NodeFile.write("\n")


        print(self.FirstPOSRelEncoder)
        print(self.FirstNERRelEncoder)
        print(self.SecondSynRelEncoder)
        print(self.SecondSemRelEncoder)
        print("complete!")

        First2Second_CoomatrixFile.close()
        WordFirst_NodeFile.close()
        RelFirst_NodeFile.close()
        RelSecond_NodeFile.close()

    def filterword(self,sentence):

            filtered_s = []
            for word in sentence:
                if word != "\\":
                    filtered_s.append(word)
                # else:
                #     print("filter:",word)

            return filtered_s


    def load_data (self, split=None):
        sentence_id = 0
        data_dir = self.data_dir
        data = []
        if split is None:
            files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.txt')]
        else:
            if not split.endswith('.txt'):
                split += '.txt'
            files = [os.path.join(data_dir, f'{split}')]
        print(files)
        # exit(0)
        #files = [os.path.join(data_dir,'test.txt')]
        for file in files:
            print(file)
            index = 0
            with open(file) as f:
                for line in f:
                    datas = line.rstrip().split('\t')
                    if len(datas) == self.args.file_length:
                        text1, text2, label = datas[0],datas[1],datas[2]
                        if len(text1.split()) > self.args.min_length and len(text2.split()) > self.args.min_length:

                            index += 1
                            data.append({
                                'text1': " ".join(text1.split()[:self.args.max_length]),
                                'text2': " ".join(text2.split()[:self.args.max_length]),
                                'target': label,
                                'sentence_id': sentence_id
                            })
                            sentence_id += 1
                        else:
                            print("text1 filter:",text1)
                            print("text2 filter:",text2)
            print("total num:",index)
        #exit(0)

        return data

    def process_sample (self, sample):
        text1 = sample['text1']
        text2 = sample['text2']
        if self.args.lower_case:
            text1 = text1.lower()
            text2 = text2.lower()

        text1 = self.postager.tokenize(text1)
        text2 = self.postager.tokenize(text2)

        return self.filterword(text1),self.filterword(text2)


def main():
    argv = sys.argv

    if len(argv) == 2:
        arg_groups = params.parse(sys.argv[1])
        for args, config in arg_groups:
            graphbuilder = TextGraphBuilder(args)
            #print(graphbuilder.data[0])

            graphbuilder.WriteGraphIntoFile()


            exit(0)
    elif len(argv) == 3 and '--dry' in argv:
        argv.remove('--dry')
        arg_groups = params.parse(sys.argv[1])
        pprint([args.__dict__ for args, _ in arg_groups])
    else:
        print('Usage: "python train.py configs/xxx.json5"')

if __name__ == "__main__":
    main()

import os
from pprint import pprint
from .model import Model
from .interface import Interface
from .utils.loader import load_data
import copy
from pytorch_transformers import AdamW, WarmupLinearSchedule, BertTokenizer,WarmupCosineSchedule
import torch
import torch.nn as nn
import numpy as np
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag import StanfordPOSTagger
from nltk.internals import find_jars_within_path
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse import CoreNLPParser
from nltk.stem import WordNetLemmatizer
from GMutil.factorize_graph_matching import kronecker_sparse, kronecker_torch
from GMutil.sparse_torch import CSRMatrix3d

#from .MatrixVis import matrix_visualization
#nohup java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 1500000 >logv2.out 2>&1 &
class Evaluator:
    def __init__(self, model_path):
        self.model_path = model_path[0]
        #self.data_file = data_file
        #data = load_data(args)
        print(self.model_path)
        self.model, checkpoint = Model.load(self.model_path)

        self.args = checkpoint['args']
        self.device = torch.cuda.current_device() if self.args.cuda else torch.device('cpu')


        #self.tokenizer = BertTokenizer.from_pretrained ('modeling_bert/bert-base-uncased-vocab.txt')
        self.dependency_parser = CoreNLPDependencyParser(url='http://localhost:9000')
        self.postager =  CoreNLPParser(url='http://localhost:9000', tagtype='pos')
        self.lemmatizer = WordNetLemmatizer()

        self.bert_tokenizer = BertTokenizer.from_pretrained("modeling_bert/bert-base-uncased-vocab.txt")

        self.POSdict = {'VB': 0, 'IN': 1, 'DT': 2, 'NN': 3, 'CD': 4, ',': 5, 'JJ': 6, 'CC': 7, 'PR': 8, '-L': 9, '-R': 10, 'MD': 11, 'TO': 12, 'RB': 13, 'WR': 14, 'WD': 15, 'PD': 16, 'RP': 17, 'PO': 18, ':': 19, '.': 20, 'EX': 21, 'SY': 22, 'AF': 23, 'WP': 24, 'FW': 25, 'LS': 26, 'UH': 27, "''": 28, 'GW': 29, 'NF': 30, '$': 31, '``': 32, 'HY': 33}
        #self.NERdict = {'O': 0, 'NUMBER': 1, 'COUNTRY': 2, 'CAUSE_OF_DEATH': 3, 'ORDINAL': 4, 'DATE': 5, 'TITLE': 6, 'DURATION': 7, 'SET': 8, 'NATIONALITY': 9, 'IDEOLOGY': 10, 'STATE_OR_PROVINCE': 11, 'RELIGION': 12, 'PERCENT': 13, 'MONEY': 14, 'PERSON': 15, 'ORGANIZATION': 16, 'MISC': 17, 'TIME': 18, 'LOCATION': 19, 'CRIMINAL_CHARGE': 20, 'CITY': 21, 'URL': 22}
        self.depdict = {'ca': 0, 'de': 1, 'ad': 2, 'ac': 3, 'ob': 4, 'nm': 5, 'co': 6, 'nu': 7, 'pu': 8, 'RO': 9, 'cc': 10, 'am': 11, 'ma': 12, 'ns': 13, 'au': 14, 'xc': 15, 'fi': 16, 'ap': 17, 'pa': 18, 'io': 19, 'ex': 20, 'cs': 21, 'di': 22, 'go': 23, 'or': 24}
        #self.semdict = {'ARG3': 0, 'ARG2': 1, 'BV': 2, 'ARG1': 3, 'loc': 4, 'compound': 5, 'subord': 6, '_and_c': 7, 'than': 8, '_or_c': 9, 'parenthetical': 10, 'part': 11, 'manner': 12, 'comp_less': 13, '_but_c': 14, 'neg': 15, 'poss': 16, 'appos': 17, 'ARG4': 18, 'mwe': 19, 'conj': 20, 'comp': 21, '_but+not_c': 22, '_nor_c': 23, '_versus_c': 24, 'measure': 25, 'times': 26, '_and+not_c': 27, '_not_c': 28, 'comp_so': 29, 'temp': 30, 'plus': 31, 'comp_too': 32, '_and+then_c': 33, '_as+well+as_c': 34, '_and+thus_c': 35, 'comp_enough': 36, '_then_c': 37, '_rather+than_c': 38, '_instead+of_c': 39, 'of': 40, '_but+also_c': 41, '_plus_c': 42, '_and+so_c': 43, '_and+also_c': 44, 'discourse': 45, '_if+not_c': 46, '_yet_c': 47, '_minus_c': 48, '_though_c': 49}

        self.id2POS = {v:k for k,v in zip(self.POSdict.keys(),self.POSdict.values())}
        #self.id2NER = {v:k for k,v in zip(self.NERdict.keys(),self.NERdict.values())}
        self.id2dep = {v:k for k,v in zip(self.depdict.keys(),self.depdict.values())}
        #self.id2sem = {v:k for k,v in zip(self.semdict.keys(),self.semdict.values())}


    def MakeNode2EdgeGraph(self,coomatrix,node_max_len,edge_max_len,is_bidirecional = True):
        #coomatrix = coomatrix.split(",")
        print(coomatrix)
        len_coomatrix = len(coomatrix)
        in_graph,out_graph = np.zeros((node_max_len,edge_max_len)),np.zeros((node_max_len,edge_max_len))
        #print(coomatrix)

        for i in range(0,len_coomatrix,2):

            in_graph[int(coomatrix[i])][i//2] = 1
            out_graph[int(coomatrix[i+1])][(i+1)//2] = 1
            if is_bidirecional:
                out_graph[int(coomatrix[i])][i//2] = 1
                in_graph[int(coomatrix[i+1])][(i+1)//2] = 1

        return [in_graph],[out_graph]

    def MakesingleBatch(self,sentenceX,sentenceY):

        textX_batch = [[]]
        textY_batch = [[]]


        RelPOSXid_batch = [[]]
        RelPOSYid_batch = [[]]

        RelSynXid_batch = [[]]
        RelSynYid_batch = [[]]

        # OneX_mask_batch = []
        # OneY_mask_batch = []


        bert_full_batch = []
        bert_segment_batch = []
        bert_mask_batch = []



        X_length = []
        Y_length = []

        coomatrix_X = []
        coomatrix_Y = []

        result, = self.dependency_parser.raw_parse(sentenceX.lower())
        parse_list = result.to_conll(4)[:-1].split("\n")
        edge_index = 0

        self.POS_X,self.POS_Y,self.SYN_X,self.SYN_Y,self.X,self.Y = [],[],[],[],[],[]

        for index,i in enumerate(parse_list):
            word,POS,ids,rel = i.split("\t")

            ##semantic parses' format#########
            #input_format.append((word,self.lemmatizer.lemmatize(word),POS))
            self.POS_X.append(POS)
            if int(ids) != 0:
                self.SYN_X.append(rel)
            self.X.append(word)
            POS = POS[:2]
            rel = rel[:2]



            textX_batch[0].append(self.bert_tokenizer.convert_tokens_to_ids(word))
            RelPOSXid_batch[0].append(self.POSdict[POS])


            if int(ids) != 0:
                RelSynXid_batch[0].append(self.depdict[rel])
                #second_rel_list.append((index,int(ids)-1,str(index)+":"+str(int(ids)-1)))
                coomatrix_X.append(index)
                coomatrix_X.append(int(ids)-1)
                # coomatrix_X[1].append(edge_index)
                # coomatrix_X[1].append(edge_index)
                edge_index = edge_index + 1


        result, = self.dependency_parser.raw_parse(sentenceY.lower())
        parse_list = result.to_conll(4)[:-1].split("\n")
        edge_index=0
        for index,i in enumerate(parse_list):
            word,POS,ids,rel = i.split("\t")

            ##semantic parses' format#########
            #input_format.append((word,self.lemmatizer.lemmatize(word),POS))
            self.POS_Y.append(POS)
            if int(ids) != 0:
                self.SYN_Y.append(rel)
            self.Y.append(word)

            POS = POS[:2]
            rel = rel[:2]

            textY_batch[0].append(self.bert_tokenizer.convert_tokens_to_ids(word))
            RelPOSYid_batch[0].append(self.POSdict[POS])
            #edge_index = 0
            if int(ids) != 0:
                RelSynYid_batch[0].append(self.depdict[rel])
                #second_rel_list.append((index,int(ids)-1,str(index)+":"+str(int(ids)-1)))
                coomatrix_Y.append(index)
                coomatrix_Y.append(int(ids)-1)
                # coomatrix_X[1].append(edge_index)
                # coomatrix_X[1].append(edge_index)
                edge_index = edge_index + 1

        X_length.append(len(textX_batch[0]))
        Y_length.append(len(textY_batch[0]))

        self.x_length,self.y_length = X_length[0],Y_length[0]

        print(textX_batch)
        print(textY_batch)

        SynXG1_batch, SynXG2_batch = self.MakeNode2EdgeGraph(coomatrix_X,self.args.max_one_len,self.args.max_syn_len)
        SynYG1_batch, SynYG2_batch = self.MakeNode2EdgeGraph(coomatrix_Y,self.args.max_one_len,self.args.max_syn_len)



        bert_full_batch.append([self.bert_tokenizer.convert_tokens_to_ids("[CLS]")] + textX_batch[0] + [0]*(self.args.max_one_len-X_length[0]) + [self.bert_tokenizer.convert_tokens_to_ids("[SEP]")]
                               +textY_batch[0] + [0]*(self.args.max_one_len-Y_length[0])+[self.bert_tokenizer.convert_tokens_to_ids("[SEP]")])
        bert_segment_batch.append([1]*(2+self.args.max_one_len)+[0]*(1+self.args.max_one_len))
        bert_mask_batch.append([1]*(1+X_length[0]) + [0]*(self.args.max_one_len-X_length[0]) + [1]*(1+Y_length[0]) + [0]*(self.args.max_one_len-Y_length[0]) + [1])

        OneX_mask_batch = [[1]*X_length[0] + [0]*(self.args.max_one_len-X_length[0])]
        OneY_mask_batch = [[1]*Y_length[0] + [0]*(self.args.max_one_len-Y_length[0])]


        Syn1G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(SynYG1_batch, SynXG1_batch)]
        Syn2G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(SynYG2_batch, SynXG2_batch)]
        Syn1G = CSRMatrix3d(Syn1G,device=self.device)
        Syn2G = CSRMatrix3d(Syn2G,device=self.device).transpose()

        RelPOSXid_batch[0] = RelPOSXid_batch[0] + [0]*(self.args.max_one_len-self.x_length)
        RelPOSYid_batch[0] = RelPOSYid_batch[0] + [0]*(self.args.max_one_len-self.y_length)
        RelSynXid_batch[0] = RelSynXid_batch[0] + [0]*(self.args.max_syn_len-self.x_length+1)
        RelSynYid_batch[0] = RelSynYid_batch[0] + [0]*(self.args.max_syn_len-self.y_length+1)

        RelPOSXid_batch = torch.LongTensor(RelPOSXid_batch).to(self.device)

        RelPOSYid_batch = torch.LongTensor(RelPOSYid_batch).to(self.device)


        RelSynXid_batch = torch.LongTensor(RelSynXid_batch).to(self.device)

        RelSynYid_batch = torch.LongTensor(RelSynYid_batch).to(self.device)


        OneX_mask_batch = torch.FloatTensor(OneX_mask_batch).to(self.device)
        OneY_mask_batch = torch.FloatTensor(OneY_mask_batch).to(self.device)

        print(bert_full_batch)



        inputs = {
            'bert_full_batch':torch.LongTensor(bert_full_batch).to(self.device),
            'bert_segment_batch':torch.LongTensor(bert_segment_batch).to(self.device),
            'bert_mask_batch':torch.FloatTensor(bert_mask_batch).to(self.device),

            'SynG':(Syn1G,Syn2G),

            #

            'RelPOSXid_batch': RelPOSXid_batch,

            'RelPOSYid_batch': RelPOSYid_batch,

            'RelSynXid_batch': RelSynXid_batch,

            'RelSynYid_batch': RelSynYid_batch,

            'OneX_mask_batch': OneX_mask_batch,
            'OneY_mask_batch': OneY_mask_batch,



            'X_length':X_length,
            'Y_length':Y_length,
            'batch_size':1,

        }

            # word_token.append(self.tokenizer.convert_tokens_to_ids(word))
            # words.append(word)


        return inputs

    def GetaffinityMatrices(self,inputs):
        POS_result,Syn_result,word_simmatrix,S = self.model.network.AffinitymatrixGet(inputs)
        return POS_result,Syn_result,word_simmatrix,S




    def normalize(self,a):
        min_a,max_a = np.min(a),np.max(a)
        #return list((a-min_a)/(max_a-min_a))
        t = (a-min_a)/(max_a-min_a)
        list_a = []
        for i in t:
            list_a.append(list(i))
        return list_a

    def evaluate(self):

        inputs = self.MakesingleBatch("this gas is oxygen","oxygen gas is given off by plants")
        POS_result,Syn_result,word_simmatrix,S = self.GetaffinityMatrices(inputs)
        POS_result = POS_result[:self.x_length,:self.y_length].cpu().detach().numpy()
        Syn_result = Syn_result[:self.x_length-1,:self.y_length-1].cpu().detach().numpy()
        word_simmatrix = word_simmatrix[:self.x_length,:self.y_length].cpu().detach().numpy()
        S = S[:self.x_length,:self.y_length].cpu().detach().numpy()



        print("POS:")
        print(self.normalize(POS_result))

        print("dep:")
        print(self.normalize(Syn_result))

        print("word:")
        print(self.normalize(word_simmatrix))

        print("S:")
        print(self.normalize(S))

        print("=======")
        #self.POS_X,self.POS_Y,self.SYN_X,self.SYN_Y,self.X,self.Y
        print(self.POS_X)
        print(self.POS_Y)
        print(self.SYN_X)
        print(self.SYN_Y)
        print(self.X)
        print(self.Y)
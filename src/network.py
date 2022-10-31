
import torch
import torch.nn.functional as F
from .modules import Module, ModuleList, ModuleDict
from pytorch_transformers import  BertModel, BertConfig,BertTokenizer
from .modules.prediction import registry as prediction
from .modules.prediction import Prediction_Bert,Prediction_Bert_GAT
from .modules.GCNS import *
import torch.nn as nn
import math
from GMutil.affinity_layer import *
from GMutil.gnn import *
from GMutil.factorize_graph_matching import *

import time

class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

class TextNet(nn.Module):
    def __init__(self,args): #code_length为fc映射到的维度大小

        super(TextNet, self).__init__()
        # for p in self.parameters():
        #     p.requires_grad = False
        code_length = args.hidden_size
        modelConfig = BertConfig.from_pretrained(args.bert_config_dir)
        #modelConfig.output_attentions=True
        modelConfig.output_attentions=True
        modelConfig.output_hidden_states  = True
        self.textExtractor = BertModel.from_pretrained(
            args.bert_model_dir, config=modelConfig)

        self.textExtractor.train()
        self.embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(self.embedding_dim, code_length)
        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, :, :]
        semantic_attentions = (output[3][6][:,:,:]+output[3][7][:,:,:]+output[3][8][:,:,:]+output[3][9][:,:,:]+output[3][10][:,:,:]+output[3][11][:,:,:])/6
        semantic_attentions = torch.mean(semantic_attentions,dim=1,keepdim=False)
        features = self.fc(text_embeddings)
        features = self.tanh(features)
        return features,semantic_attentions



    def do_eval(self):
        self.textExtractor.eval()

    def do_train(self):
        self.textExtractor.train()



class Network(Module):
    def __init__(self, args,device):
        super().__init__()

        self.dropout = args.dropout
        self.device = device

        self.bert_feature = TextNet(args)
        #self.bert_dim

        self.hidden_size = args.hidden_size
        self.args = args

        self.bert_dim = self.bert_feature.embedding_dim
        self.bert_laynorm = torch.nn.LayerNorm(self.bert_dim,1e-12)

        self.affinity_layerWRS = ProjectAffinity(args.hidden_size)
        #self.affinity_layerSem = ProjectAffinity(args.hidden_size)
        self.affinity_layerWAS = ProjectAffinity(args.hidden_size//2)
        #self.affinity_layerNER = ProjectAffinity(args.hidden_size//2)
        self.affinity_semantic = ProjectAffinity(args.hidden_size//2)


        self.affinity_word = PointAffinity(args.hidden_size//2)

        self.sparsity = args.GM_sparsity


        self.syn_embeddings = nn.Embedding(args.max_SynEdgeSize,args.hidden_size*2)
        self.sem_embeddings = nn.Embedding(args.max_SemEdgeSize,args.hidden_size*2)

        self.pos_embeddings = nn.Embedding(args.max_POSEdgeSize,args.hidden_size)
        self.ner_embeddings = nn.Embedding(args.max_NEREdgeSize,args.hidden_size)
        self.gnn_layer = args.GNN_LAYER

        for i in range(args.GNN_LAYER):
            tau = args.SK_TAU
            if i == 0:
                #gnn_layer = Gconv(1, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(1, 1, args.GNN_FEAT[i] + (1 if args.SK_EMB else 0), args.GNN_FEAT[i],
                                     sk_channel=args.SK_EMB, sk_tau=tau, edge_emb=True)
                #gnn_layer = HyperConvLayer(1, 1, cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            else:
                #gnn_layer = Gconv(cfg.NGM.GNN_FEAT, cfg.NGM.GNN_FEAT)
                gnn_layer = GNNLayer(args.GNN_FEAT[i - 1] + (1 if args.SK_EMB else 0), args.GNN_FEAT[i - 1],
                                     args.GNN_FEAT[i] + (1 if args.SK_EMB else 0), args.GNN_FEAT[i],
                                     sk_channel=args.SK_EMB, sk_tau=tau, edge_emb=None)
                #gnn_layer = HyperConvLayer(cfg.NGM.GNN_FEAT[i-1] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i-1],
                #                           cfg.NGM.GNN_FEAT[i] + (1 if cfg.NGM.SK_EMB else 0), cfg.NGM.GNN_FEAT[i],
                #                           sk_channel=cfg.NGM.SK_EMB, voting_alpha=alpha)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)



        #self.CRFFirst = CRF(args.max_FirstEdgeSize, batch_first=True)
        #self.CRFSecond = CRF(args.max_SecondEdgeSize, batch_first=True)
        self.cls_projection = nn.Linear(self.hidden_size,args.predict_dim)
        self.v_projection = nn.Linear(args.max_one_len*args.max_one_len*(args.GNN_FEAT[-1]+ (1 if args.SK_EMB else 0)),args.predict_dim)

        #self.test_projection = nn.Linear(args.max_one_len*args.max_one_len*1,args.predict_dim)


        #self.bert_predict = Prediction_Bert(args,hidden_size = (args.GNN_FEAT[-1] + 1 + (1 if args.SK_EMB else 0))*2)
        self.bert_predict = Prediction_Bert(args,hidden_size = args.predict_dim)
        self.bert_predict_v = Prediction_Bert(args,hidden_size = args.predict_dim)
        self.bert_predict_all = Prediction_Bert(args,hidden_size = args.predict_dim*2)

        #self.bert_test = Prediction_Bert(args,hidden_size = args.predict_dim)
        #self.Syntactic_loss = nn.CrossEntropyLoss()



    def forward(self, inputs):
        '''
            'bert_full_batch':torch.LongTensor(bert_full_batch).to(self.device),
            'bert_segment_batch':torch.LongTensor(bert_segment_batch).to(self.device),
            'bert_mask_batch':torch.FloatTensor(bert_mask_batch).to(self.device),
        :param inputs:
        :return:
        '''
        self.start_time = time.time()
        bert_feature,semantic_attentions = self.bert_feature(inputs['bert_full_batch'],inputs['bert_segment_batch'].long(),inputs['bert_mask_batch'])
        cls = self.cls_projection(bert_feature[:,0,:])

        self.bert_time = time.time()

        #print("bert time: ",bert_time-start_time)
        Semantic_X = bert_feature[:,1:1+self.args.max_one_len,:] * inputs['OneX_mask_batch'].unsqueeze(-1)
        Semantic_Y = bert_feature[:,2+self.args.max_one_len:2+2*self.args.max_one_len,:] * inputs['OneY_mask_batch'].unsqueeze(-1)
        batch_size = inputs['batch_size']
        sentence_x = torch.mean(Semantic_X,dim=1,keepdim=False)
        sentence_y = torch.mean(Semantic_Y,dim=1,keepdim=False)

        #sim = self.affinity_word(sentence_x,sentence_y)
        #sim_loss = 0.05*F.cross_entropy(torch.softmax(sim,dim=-1),(torch.range(0,batch_size-1)).long().to(self.device))


        Syn_X,Syn_Y,Sem_X,Sem_Y = inputs['RelSynXid_batch'],inputs['RelSynYid_batch'],inputs['RelSemXid_batch'],inputs['RelSemYid_batch']
        Syn_X_emb,Syn_Y_emb,Sem_X_emb,Sem_Y_emb = self.syn_embeddings(Syn_X),self.syn_embeddings(Syn_Y),self.sem_embeddings(Sem_X),self.sem_embeddings(Sem_Y)
        #
        POS_X,POS_Y,NER_X,NER_Y = inputs['RelPOSXid_batch'],inputs['RelPOSYid_batch'],inputs['RelNERXid_batch'],inputs['RelNERYid_batch']
        POS_X_emb,POS_Y_emb,NER_X_emb,NER_Y_emb = self.pos_embeddings(POS_X),self.pos_embeddings(POS_Y),self.ner_embeddings(NER_X),self.ner_embeddings(NER_Y)

        POS_g,NER_g,Syn_g,Sem_g = self.MakeBinaryMaitrx(POS_X,POS_Y),self.MakeBinaryMaitrx(NER_X,NER_Y),self.MakeBinaryMaitrx(Syn_X,Syn_Y),self.MakeBinaryMaitrx(Sem_X,Sem_Y)


        Kp = self.affinity_layerWAS(POS_X_emb, POS_Y_emb)
        WAS_loss = self.args.lambda_first_POS*F.binary_cross_entropy_with_logits(Kp,POS_g)

        #if self.args.WRS_type  == 'SYN':

        Ke = self.affinity_layerWRS(Syn_X_emb, Syn_Y_emb)
        WRS_loss = self.args.lambda_second_Syn*F.binary_cross_entropy_with_logits(Ke,Syn_g)

        G1,G2 = inputs['SynG']
        syn_loss = WAS_loss + WRS_loss
        # #

        # KpSem = self.affinity_semantic(Semantic_X,Semantic_Y)
        KpSem = semantic_attentions[:,1:1+self.args.max_one_len,2+self.args.max_one_len:2+2*self.args.max_one_len]

        K = construct_aff_mat(Ke, torch.zeros_like(KpSem).to(self.device), G1, G2)

        all_length = len(K.view(-1))
        K_pos = K[K>0].view(-1)
        pos_length = len(K_pos)
        #print("init_sparsity:", pos_length/all_length)
        K_list = torch.sort(K_pos).values
        s_value = K_list[int(pos_length*self.sparsity)-1]

        self.now_sparsity = float(pos_length*self.sparsity)/all_length

        A = (K > s_value).to(K.dtype)
        # #
        emb_K = K.unsqueeze(-1)
        emb = (KpSem+0.5*Kp).contiguous().view(KpSem.shape[0], -1,1)
        #
        # # # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_K, emb = gnn_layer(A, emb_K, emb, inputs['X_length'], inputs['Y_length']) #, norm=False)

        self.GM_time = time.time()

        emb = self.v_projection(emb.contiguous().view(batch_size,-1))

        Matching_feature = torch.cat((emb,cls),dim=-1)
        result = self.bert_predict_all(Matching_feature)
        return result, syn_loss

    def poolings(self,x):
        return x.max(dim=0)[0].unsqueeze(0)

    def mean(self,x,mask):
        mask = mask.unsqueeze(-1)
        return torch.mean(x*mask,dim=1,keepdim=False)

    def AffinitymatrixGet(self,inputs):
        bert_feature,semantic_attentions = self.bert_feature(inputs['bert_full_batch'],inputs['bert_segment_batch'].long(),inputs['bert_mask_batch'])


        batch_size = inputs['batch_size']

        ###test constractive


        Syn_X,Syn_Y = inputs['RelSynXid_batch'],inputs['RelSynYid_batch']
        Syn_X_emb,Syn_Y_emb = self.syn_embeddings(Syn_X),self.syn_embeddings(Syn_Y)
        #
        POS_X,POS_Y = inputs['RelPOSXid_batch'],inputs['RelPOSYid_batch']
        POS_X_emb,POS_Y_emb = self.pos_embeddings(POS_X),self.pos_embeddings(POS_Y)



        Kp = self.affinity_layerWAS(POS_X_emb, POS_Y_emb)


        Ke = self.affinity_layerWRS(Syn_X_emb, Syn_Y_emb)

        KpSem = semantic_attentions[:,1:1+self.args.max_one_len,2+self.args.max_one_len:2+2*self.args.max_one_len]

        G1,G2 = inputs['SynG']
        print(Ke.shape)
        #print(G1.shape)
        K = construct_aff_mat(Ke, torch.zeros_like(KpSem).to(self.device), G1, G2)

        # # #K = construct_aff_mat(KpSem, torch.zeros_like(KpSem).to(self.device), Sem1G, Sem2G)
        # #
        A = (K > 0).to(K.dtype)
        # #
        emb_K = K.unsqueeze(-1)
        emb = (KpSem+0.5*Kp).contiguous().view(KpSem.shape[0], -1,1)
        #
        # # # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb_K, emb = gnn_layer(A, emb_K, emb, inputs['X_length'], inputs['Y_length']) #, norm=False)

        emb_S = torch.mean(emb,dim=-1,keepdim=False).view(1,self.args.max_one_len,self.args.max_one_len)
        return Kp[0],Ke[0],KpSem[0],emb_S[0]


    def MakeBinaryMaitrx(self,X,Y):
        '''

        :param X: [batch,n]
        :param Y: [batch,n]
        :return:  [batch,n,n]
        '''
        return (X.unsqueeze(-1)==Y.unsqueeze(1)).float()




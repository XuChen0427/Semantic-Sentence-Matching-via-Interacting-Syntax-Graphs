

import os
import numpy as np

def TurnList(list_str,sep=',',add=0,mask_num = 0,return_mask = False):
    lists = list_str.split(sep)
    list_int = []
    for l in lists:
        list_int.append(int(l)+add)
    length = len(list_int)
    mask = [1] * length + [0] * (mask_num-length)
    list_int = list_int + [0] * (mask_num-length)
    #return np.array(list_int)
    if return_mask:
        return list_int,mask,length


    return list_int

def MakeNode2EdgeGraph(coomatrix,node_max_len,edge_max_len,is_bidirecional = True):
    coomatrix = coomatrix.split(",")
    len_coomatrix = len(coomatrix)
    in_graph,out_graph = np.zeros((node_max_len,edge_max_len)),np.zeros((node_max_len,edge_max_len))
    #print(coomatrix)

    for i in range(0,len_coomatrix,2):

        in_graph[int(coomatrix[i]),i//2] = 1
        out_graph[int(coomatrix[i+1]),(i+1)//2] = 1
        if is_bidirecional:
            out_graph[int(coomatrix[i]),i//2] = 1
            in_graph[int(coomatrix[i+1]),(i+1)//2] = 1

    return in_graph,out_graph




def load_data(args):
    print("start to loading data...")
    WordFirst_NodeFile = open(os.path.join(args.graph_dir,args.WordFirst_NodeFile),'r')
    RelFirst_NodeFile = open(os.path.join(args.graph_dir,args.RelFirst_NodeFile),'r')
    RelSecond_NodeFile = open(os.path.join(args.graph_dir,args.RelSecond_NodeFile),'r')
    First2Second_CoomatrixFile = open(os.path.join(args.graph_dir,args.First2Second_CoomatrixFile),'r')
    #Second2Third_CoomatrixFile = open(os.path.join(args.graph_dir,args.Second2Third_CoomatrixFile),'r')
    datas = []
    max_index = 0
    max_first_num = 0
    max_sem_num = 0
    max_syn_num = 0
    # 'OneX_mask_batch': OneX_mask_batch,
    # 'OneY_mask_batch': OneY_mask_batch,
    # 'SynX_mask_batch': SynX_mask_batch,
    # 'SemX_mask_batch': SemX_mask_batch,
    # 'SynY_mask_batch': SynY_mask_batch,
    # 'SemY_mask_batch': SemY_mask_batch,
    for (WordFirst,RelFirst,RelSecond,First2Second) in zip(WordFirst_NodeFile,RelFirst_NodeFile,RelSecond_NodeFile,First2Second_CoomatrixFile):
        Xid,Yid,label = WordFirst.strip().split("  ")
        Xid,OneX_mask_batch,X_length = TurnList(Xid,mask_num=args.max_one_len,return_mask=True)
        Yid,OneY_mask_batch,Y_length = TurnList(Yid,mask_num=args.max_one_len,return_mask=True)

        RelOneXid,RelOneYid = RelFirst.strip().split("  ")
        #print(RelOneXid)
        RelOneXid = RelOneXid.split(" ")

        RelOnePOSXid = TurnList(RelOneXid[0],add=0,mask_num=args.max_one_len)
        RelOneNERXid = TurnList(RelOneXid[1],add=0,mask_num=args.max_one_len)
        RelOneYid = RelOneYid.split(" ")
        RelOnePOSYid = TurnList(RelOneYid[0],add=0,mask_num=args.max_one_len)
        RelOneNERYid = TurnList(RelOneYid[1],add=0,mask_num=args.max_one_len)

        RelTwoXid,RelTwoYid = RelSecond.strip().split("  ")
        #print(RelTwoXid)
        # if len(RelOnePOSXid) != len(RelOneNERXid) or len(RelOnePOSYid) != len(RelOneNERYid):
        #     print(len(RelOnePOSXid))
        #     print(len(RelOneNERXid))
        #     print(len(RelOnePOSYid))
        #     print(len(RelOneNERYid))
        #     print("ERROR!")
            #exit(0)
        if len(RelTwoXid.split(" ")) ==2 and len(RelTwoYid.split(" ")) == 2 and len(RelOnePOSXid) == len(RelOneNERXid) and len(RelOnePOSYid) == len(RelOneNERYid):
            RelTwoXid,RelTwoYid = RelTwoXid.split(" "),RelTwoYid.split(" ")


            RelSynXid = TurnList(RelTwoXid[0],add=0,mask_num=args.max_syn_len)
            RelSemXid = TurnList(RelTwoXid[1],add=0,mask_num=args.max_sem_len)
            #RelTwoYid = RelTwoYid.split(" ")
            RelSynYid = TurnList(RelTwoYid[0],add=0,mask_num=args.max_syn_len)
            RelSemYid = TurnList(RelTwoYid[1],add=0,mask_num=args.max_sem_len)

            First2SecondCoomatrix_X,First2SecondCoomatrix_Y = First2Second.strip().split("  ")
            First2SecondCoomatrix_X = First2SecondCoomatrix_X.split(" ")

            SynXG1,SynXG2 = MakeNode2EdgeGraph(First2SecondCoomatrix_X[0],node_max_len=args.max_one_len,edge_max_len=args.max_syn_len,is_bidirecional=True)
            SemXG1,SemXG2 = MakeNode2EdgeGraph(First2SecondCoomatrix_X[1],node_max_len=args.max_one_len,edge_max_len=args.max_sem_len,is_bidirecional=True)



            #First2SecondCoomatrix_X = [TurnList(First2SecondCoomatrix_X[0]),TurnList(First2SecondCoomatrix_X[1])]
            #SynCoomatrix_X = TurnList(First2SecondCoomatrix_X[0],mask_num=args.max_syn_len*2)
            #SemCoomatrix_X = TurnList(First2SecondCoomatrix_X[1],mask_num=args.max_sem_len*2)

            First2SecondCoomatrix_Y = First2SecondCoomatrix_Y.split(" ")
            SynYG1,SynYG2 = MakeNode2EdgeGraph(First2SecondCoomatrix_Y[0],node_max_len=args.max_one_len,edge_max_len=args.max_syn_len,is_bidirecional=True)
            SemYG1,SemYG2 = MakeNode2EdgeGraph(First2SecondCoomatrix_Y[1],node_max_len=args.max_one_len,edge_max_len=args.max_sem_len,is_bidirecional=True)
          

            max_first_num = max(max_first_num,len(Xid),len(Yid))
            max_sem_num = max(max_sem_num,len(RelSemXid),len(RelSemYid))
            max_syn_num = max(max_syn_num,len(RelSynXid),len(RelSynYid))
            datas.append({
                "textX":Xid,
                "textY":Yid,
                "RelOnePOSXid":RelOnePOSXid,
                "RelOneNERXid":RelOneNERXid,
                "RelOnePOSYid":RelOnePOSYid,
                "RelOneNERYid":RelOneNERYid,
                "RelSynXid":RelSynXid,
                "RelSemXid":RelSemXid,
                "RelSynYid":RelSynYid,
                "RelSemYid":RelSemYid,

                "SynXG1":SynXG1,
                "SynXG2":SynXG2,
                "SemXG1":SemXG1,
                "SemXG2":SemXG2,
                "SynYG1":SynYG1,
                "SynYG2":SynYG2,
                "SemYG1":SemYG1,
                "SemYG2":SemYG2,

                "X_len": X_length,
                "Y_len": Y_length,


                "OneX_mask_batch":OneX_mask_batch,
                "OneY_mask_batch":OneY_mask_batch,

                "target":int(label)
            })
        else:
            print("filtering--------------")

    WordFirst_NodeFile.close()
    RelFirst_NodeFile.close()
    RelSecond_NodeFile.close()
    First2Second_CoomatrixFile.close()
    
    return datas



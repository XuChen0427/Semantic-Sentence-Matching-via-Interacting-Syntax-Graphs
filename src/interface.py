

import os
import random
from .utils.loader import *


class Interface:
    def __init__(self, args):
        self.args = args

        # self.Coomatrix = load_CooMatrix(os.path.join(args.graph_dir, args.coomatrix_dir))
        # self.Nodelist = load_Nodelist(os.path.join(args.graph_dir, args.node_dir))
        #self.data, self.file_index = load_data(self.args.data_dir, self.Coomatrix, self.Nodelist,self.args.filelength)


    def shuffle_data(self,data):
        random.shuffle(data)

    

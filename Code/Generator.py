from keras.utils import Sequence
import numpy as np 

class get_hir_train_generator(Sequence):
    def __init__(self,news_scoring,clicked_news,user_id, news_id, label, batch_size):
        self.news_emb = news_scoring
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __get_news(self,docids):
        news_emb = self.news_emb[docids]

        return news_emb
        

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        label = self.label[start:ed]
        
        doc_ids = self.doc_id[start:ed]
        title= self.__get_news(doc_ids)
        
        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        user_title = self.__get_news(clicked_ids)

        return ([title, user_title],[label])

class get_test_generator(Sequence):
    def __init__(self,news_scoring,docids,userids, clicked_news,batch_size):
        self.docids = docids
        self.userids = userids
        
        self.news_scoring = news_scoring
        self.clicked_news = clicked_news

        self.batch_size = batch_size
        self.ImpNum = self.docids.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    def __get_news(self,docids):
        news_scoring = self.news_scoring[docids]
        
        return news_scoring
            
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
            
        docids = self.docids[start:ed]
        
        userisd = self.userids[start:ed]
        clicked_ids = self.clicked_news[userisd]

        title = self.__get_news(docids)
        user_title = self.__get_news(clicked_ids)

        return [title,user_title,]
import numpy as np
import keras
from keras.layers import Embedding
from keras.layers import *
from keras import backend as K
from keras.optimizers import *
from keras.models import Model
from Hypers import *

class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = Dense(200,activation='tanh')(user_vecs)
    user_att = keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model


def get_doc_encoder(title_word_embedding_matrix,entity_word_embedding_matrix,category_dict):

    sentence_input = Input(shape=(MAX_TITLE+MAX_ENTITY+1,), dtype='int32')
    
    title_input = keras.layers.Lambda(lambda x:x[:,:MAX_TITLE])(sentence_input)
    entity_input = keras.layers.Lambda(lambda x:x[:,MAX_TITLE:MAX_TITLE+MAX_ENTITY])(sentence_input)
    vert_input = keras.layers.Lambda(lambda x:x[:,MAX_TITLE+MAX_ENTITY:MAX_TITLE+MAX_ENTITY+1])(sentence_input)

    title_word_embedding_layer = Embedding(title_word_embedding_matrix.shape[0], title_word_embedding_matrix.shape[1], weights=[title_word_embedding_matrix],trainable=True)
    word_vecs = title_word_embedding_layer(title_input)
    droped_vecs = Dropout(0.2)(word_vecs)
    word_rep = Attention(20,20)([droped_vecs]*3)
    droped_rep = Dropout(0.2)(word_rep)
    title_vec = AttentivePooling(MAX_TITLE,400)(droped_rep)
    
    entity_word_embedding_layer = Embedding(entity_word_embedding_matrix.shape[0], entity_word_embedding_matrix.shape[1], weights=[entity_word_embedding_matrix],trainable=False)
    entity_vecs = entity_word_embedding_layer(entity_input)
    droped_vecs = Dropout(0.2)(entity_vecs)
    entity_rep = Attention(4,40)([droped_vecs]*3)
    droped_rep = Dropout(0.2)(entity_rep)
    entity_vec = AttentivePooling(MAX_ENTITY,160)(droped_rep)
    
    vert_word_embedding_layer = Embedding(len(category_dict)+1, 100,trainable=False)
    vert_emb = vert_word_embedding_layer(vert_input)
    vert_emb = keras.layers.Reshape((100,))(vert_emb)
    vert_emb = Dropout(0.2)(vert_emb)
    vert_vec = Dense(100)(vert_emb)
    
    vec = keras.layers.Concatenate(axis=-1)([title_vec,entity_vec,vert_vec])

    vec = Dense(400)(vec)
    
    sentEncodert = Model(sentence_input, vec)
    return sentEncodert

def DenseAtt():
    vec_input = Input(shape=(400*2,),dtype='float32')
    vec = Dense(400,activation='tanh')(vec_input)
    vec = Dense(256,activation='tanh')(vec)
    score = Dense(1)(vec)
    return Model(vec_input,score)

def get_inter_model():
    user_vecs_input = Input(shape=(MAX_CLICK,400))
    cand_vec_input = Input(shape=(400,))
    
    dense_att = DenseAtt()
    
    can_vec = Dropout(0.2)(cand_vec_input)
    user_vecs = Dropout(0.2)(user_vecs_input)


    user_vecs_can = keras.layers.RepeatVector(MAX_CLICK)(cand_vec_input)
        
    user_vecs_left1 = keras.layers.Lambda(lambda x:x[:,:-1,:])(user_vecs)
    user_vecs_left2 = keras.layers.Lambda(lambda x:x[:,-1:,:])(user_vecs)
    user_vecs_left = keras.layers.Concatenate(axis=-2)([user_vecs_left2,user_vecs_left1])
    user_vecs_right1 = keras.layers.Lambda(lambda x:x[:,:1,:])(user_vecs)
    user_vecs_right2 = keras.layers.Lambda(lambda x:x[:,1:,:])(user_vecs)
    user_vecs_right = keras.layers.Concatenate(axis=-2)([user_vecs_right2,user_vecs_right1])
    user_vecs_cnn = keras.layers.Concatenate(axis=-1)([user_vecs_left,user_vecs,user_vecs_right,user_vecs_can])
    user_vecs_cnn = Dense(400)(user_vecs_cnn)

    user_vecs = keras.layers.Concatenate(axis=-1)([user_vecs_can,user_vecs])
    user_vecs = keras.layers.Dense(400)(user_vecs)
    user_vecs_self = Attention(20,20)([user_vecs]*3)
    user_vecs = keras.layers.Concatenate(axis=-1)([user_vecs_cnn,user_vecs_self])
    user_vecs = Dropout(0.2)(user_vecs)
    user_vecs = Dense(400)(user_vecs)
        
    att_vecs = keras.layers.Concatenate(axis=-1)([user_vecs,user_vecs_can])
    att_score = TimeDistributed(dense_att)(att_vecs)
    att_score = keras.layers.Reshape((50,))(att_score)
    att = keras.layers.Activation('softmax')(att_score)
    user_vec = keras.layers.Dot(axes=[-1,-2])([att,user_vecs])
        
    score = keras.layers.Dot(axes=-1)([user_vec,can_vec])

    return Model([cand_vec_input,user_vecs_input,],score)


def create_model(title_word_embedding_matrix,entity_word_embedding_matrix,category_dict):
    
    MAX_LENGTH = 30    
    
    news_encoder = get_doc_encoder(title_word_embedding_matrix,entity_word_embedding_matrix,category_dict)
    inter_model = get_inter_model()
    
    clicked_title_input = Input(shape=(MAX_CLICK,MAX_TITLE+MAX_ENTITY+1,), dtype='float32')    
    title_inputs = Input(shape=(1+npratio,MAX_TITLE+MAX_ENTITY+1,),dtype='float32') 
    
    clicked_news_vecs = TimeDistributed(news_encoder)(clicked_title_input)
    title_vecs = TimeDistributed(news_encoder)(title_inputs)

    scores = []
    for i in range(1+npratio):
        news_vec = keras.layers.Lambda(lambda x:x[:,i,:])(title_vecs)
        score = inter_model([news_vec,clicked_news_vecs,])
        scores.append(score)
    scores = keras.layers.Concatenate(axis=-1)(scores)
    
    logits = keras.layers.Activation(keras.activations.softmax,name = 'recommend')(scores)     

    model = Model([title_inputs, clicked_title_input,],logits) # max prob_click_positive
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=Adam(lr=0.00005), 
                  metrics=['acc'])

    return model,news_encoder,inter_model
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, TimeDistributed, LSTM, Dropout, Activation, Lambda, concatenate, add

def deepBach(num_features_lr,         # size of left or right feature vectors
             num_features_c,          # size of central feature vectors
             num_pitches,             # size of output
             num_features_meta, 
             num_units_lstm = [200, 200],
             num_dense = 200, 
             timesteps = 16):    

    '''
    Builds DeepBach NN Model with all the layers
    '''
    ###### INPUT ######
    left_features = Input(shape = (timesteps, num_features_lr),
                          name = 'left_features')
    
    right_features = Input(shape = (timesteps, num_features_lr),
                           name = 'right_features')
    
    center_features = Input(shape = (num_features_c,),
                             name = 'center_feautres')
    
    
    ##### INPUT METADATA #####
    left_metas = Input(shape = (timesteps, num_features_meta),
                       name = 'left_metas')
    
    right_metas = Input(shape = (timesteps, num_features_meta),
                        name = 'right_metas')
    
    center_metas = Input(shape = (num_features_meta,),
                          name = 'center_metas')
    
    
    ##### EMBEDDING LAYERS #####
    embedding_left = Dense(input_dim = num_features_lr + num_features_meta,
                           units = num_dense,
                           name = 'embedding_left')
    
    embedding_right = Dense(input_dim = num_features_lr + num_features_meta,
                            units = num_dense,
                            name = 'embedding_right')
    
    
    ##### Left & Right LSTM Layers #####
    lstm_left = TimeDistributed(embedding_left)(concatenate([left_features, left_metas]))
    lstm_right = TimeDistributed(embedding_right)(concatenate([right_features, right_metas]))

    return_sequences = True
    
    for stack_index in range(len(num_units_lstm)):
        # For the last stack, only return the last output of the sequence:
        if stack_index == len(num_units_lstm) - 1:
            return_sequences = False
            
        lstm_left = LSTM(num_units_lstm[stack_index],
                         return_sequences = return_sequences,
                         name = 'lstm_left_' + str(stack_index))(lstm_left)
    
        lstm_right = LSTM(num_units_lstm[stack_index],
                          return_sequences = return_sequences,
                          name = 'lstm_right_' + str(stack_index))(lstm_right)


    ##### Center FC Layer #####
    fc_center = concatenate([center_features, center_metas])
    fc_center = Dense(num_dense, activation = 'relu')(fc_center)
    fc_center = Dense(num_dense, activation = 'relu')(fc_center)

    
    ##### Merging FC Layer #####    
    input2merge = concatenate([lstm_left, fc_center, lstm_right])
    fc_merge = Dense(num_dense, activation = 'relu')(input2merge)
    
    
    ##### Final Softmax FC Layer
    fc_final = Dense(num_pitches, activation = 'softmax', name = 'pitch_prediction')(fc_merge)
    
    
    ##### Final Model #####
    model = Model( input = [left_features, center_features, right_features,
                            left_metas, center_metas, right_metas],
                   
                   output = fc_final )
    
    
    ##### Optimizer & Loss #####
    model.compile(loss = {'pitch_prediction':'categorical_crossentropy'}, 
                  optimizer = 'adam',
                  metrics = ['accruacy'])
    
    print(model.summary())
    
    return model
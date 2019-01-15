import torch
import random
import numpy as np

from torch import nn, optim

from Data.preprocess_data import PreprocessData

'''
    DeepBach LSTM model to be run for **EACH VOICE**
'''

class DeepBach_NN(nn.Module):
    def __init__(self,
                 preprocess_data_class_instance : PreprocessData,
                 voice_index : int,
                 n_embed_notes : int,
                 n_embed_metas : int,
                 n_layers_lstm : int,
                 n_hidden_lstm : int,
                 dropout_lstm : float,
                 n_hidden_fc = 200):
        
        super(DeepBach_NN, self).__init__()
        
        self.preprocess_data_class_instance = preprocess_data_class_instance
        self.voice_index = voice_index
        self.n_voices = self.preprocess_data_class_instance.num_voices
        ### range of all possible notes per voice: ###
        self.n_notes_per_voice = [len(d) for d in preprocess_data_class_instance.note2index_dicts]
        self.n_metas = len(self.preprocess_data_class_instance.metadatas) + 1
        self.metadatas = preprocess_data_class_instance.metadatas
        self.n_values_per_meta = [meta.num_values for meta in self.metadatas] + [self.n_voices]
        
        self.n_embed_notes = n_embed_notes
        self.n_embed_metas = n_embed_metas
        self.n_layers_lstm = n_layers_lstm
        self.n_hidden_lstm = n_hidden_lstm
        self.n_hidden_fc = n_hidden_fc 
        
        ######################### < < FIRST LAYER > > #########################
        ### (1.1) [n_voices] note embedding layers ###
        self.note_embedding = nn.ModuleList(
                [nn.Embedding(n_notes, n_embed_notes) for n_notes in self.n_notes_per_voice] )
        
        ### (1.2) [n_meta] metadata embedding layers ###
        self.meta_embedding = nn.ModuleList(
                [nn.Embedding(n_values, n_embed_metas) for n_values in self.n_values_per_meta] )


        ####################### < < SECOND LAYER > > ##########################
        ### (2.1) LEFT & RIGHT LSTM LAYER ###
        self.lstm = nn.LSTM(input_size = n_embed_notes*self.n_voices + n_embed_metas*self.n_metas, 
                            hidden_size = n_hidden_fc, 
                            num_layers = n_layers_lstm, 
                            batch_first = True, 
                            dropout = dropout_lstm)
        
        ### (2.2) CENTER FC LAYER ###
        self.fc_center = nn.Sequential(
                            nn.Linear(n_embed_notes*self.n_voices + n_embed_metas*self.n_metas, 
                                      n_hidden_fc),
                            nn.ReLU(),
                            nn.Linear(n_hidden_fc, n_hidden_lstm))
                            
        
        ######################## < < THIRD LAYER > > ##########################
        ### (3.1) Merging Layer ###
        self.fc_merge = nn.Linear(n_hidden_lstm*3, n_hidden_fc)
        
        ### (3.2) Final Output Layer: # of output = # of notes in the voice ###
        self.fc_final = nn.Sequential(
                            nn.Linear(n_hidden_lstm*3, n_hidden_fc),
                            nn.ReLU(),
                            nn.Linear(n_hidden_fc, self.n_notes_per_voice[voice_index]))
    
    

    def forward(self, input_tensor):
        
        ######################### 1. CONFIGURE INPUTS #########################
        notes, metas = input_tensor
        left_notes, center_notes, right_notes = notes
        left_metas, center_metas, right_metas = metas
        
        ### SHAPES ###
        ### l,r_notes : (batch_size, ~seq_length/2, num_voices)
        ### c_notes   : (batch_size, num_voices-1)
        
        ### l,r_metas : (batch_size, ~seq_length/2, num_metas+1)
        ### c_metas   : (batch_size, num_metas+1)
        
        batch_size, n_voices, seq_length_left = left_notes.shape
        _, _, seq_length_right = right_notes.shape[2]       
        
        
        ###################### 2. NOTE EMBEDDING LAYERS #######################
        l_embed_notes = []
        r_embed_notes = []
        c_embed_notes = []
        c_voice_i = 0
        
        for i in range(self.n_voices):
            # (batch_size, ~seq_length/2, n_embed_notes)
            l_embed_per_voice = self.note_embedding[i](left_notes[:, :, i])
            l_embed_notes.append(l_embed_per_voice)
            
            r_embed_per_voice = self.note_embedding[i](right_notes[:, :, i])
            r_embed_notes.append(r_embed_per_voice)
            
            if i == self.voice_index:
                pass
            else:
                c_embed_per_voice = self.note_embedding[i](center_notes[:, c_voice_i])
                # (batch_size, n_embed_notes)
                c_embed_notes.append(c_embed_per_voice)
                c_voice_i += 1
        
        # (batch_size, ~seq_length/2, n_embed_notes*n_voices)
        l_embedded_notes = torch.cat(l_embed_notes, 2) 
        r_embedded_notes = torch.cat(r_embed_notes, 2)
        # (batch_size, n_embed_notes*(n_voices-1))
        c_embedded_notes = torch.cat(c_embed_notes, 1)
        
        
        ###################### 3. META EMBEDDING LAYERS #######################
        l_embed_metas = []
        r_embed_metas = []
        c_embed_metas = []
        
        ### c_metas   : (batch_size, num_metas+1)
        for i in range(self.n_metas):
            # (batch_size, ~seq_length/2, n_embed_metas)
            l_embed_per_meta = self.meta_embedding[i](left_metas[:, :, i])
            l_embed_metas.append(l_embed_per_meta)
            
            r_embed_per_meta = self.note_embedding[i](right_metas[:, :, i])
            r_embed_metas.append(r_embed_per_meta)
            
            c_embed_per_meta = self.note_embedding[i](center_metas[:, i])
            c_embed_metas.append(c_embed_per_meta)
        
        # (batch_size, ~seq_length/2, n_embed_metas*n_metas)
        l_embedded_metas = torch.cat(l_embed_metas, 2)
        r_embedded_metas = torch.cat(r_embed_metas, 2)
        # (batch_size, n_embed_metas*n_metas)
        c_embedded_metas = torch.cat(c_embed_metas, 1)
        
        
        ################ 4. CONCATENATE NOTE & META EMBEDDINGS ################
        left_embedded = torch.cat([l_embedded_notes, l_embedded_metas], 2)
        right_embedded = torch.cat([r_embedded_notes, r_embedded_metas], 2)
        center_embedded = torch.cat([c_embedded_notes, c_embedded_metas], 1)

        
        ##################### 5. LEFT & RIGHT LSTM LAYERS #####################
        hidden = self.init_hidden(batch_size)
        #self.lstm = nn.LSTM(input_size = n_embed_notes*self.n_voices + n_embed_metas*self.n_metas, 
        # (batch_size, seq_length_left, n_hidden_lstm)
        left_lstm_out, left_hidden = self.lstm(left_embedded, hidden)
        # (batch_size, n_hidden_lstm)
        left_lstm_out = left_lstm_out[:, -1, :] # ONLY extract the last output
        
        right_lstm_out, right_hidden = self.lstm(right_embedded, hidden)
        right_lstm_out = right_lstm_out[:, -1, :]
        
        
        ################### 6. CENTER FULLY-CONNECTED LAYER ###################
        # (batch_size, n_hidden_lstm)
        center_fc_out = self.fc_center(center_embedded)
        
        
        ################ 7. CONCATENATE ALL OUTPUTS SO FAR ################
        # (batch_size, n_hidden_lstm*3)
        input_to_final = torch.cat([left_lstm_out, center_fc_out, right_lstm_out], 1)
        
        
        ################### 8. FINAL FULLY-CONNECTED LAYER ####################
        # (batch_size, n_notes for self.voice_index)
        fc_out_final = self.fc_final(input_to_final)
  
    
    
    def init_hidden(self, batch_size):
        ''' Initialize hidden nodes w/ normal distribution '''
        device = "cuda" if torch.cuda.is_available else "cpu"
        hidden = (torch.randn(self.n_layers_lstm, batch_size, self.n_hidden_lstm).to(device),
                  torch.randn(self.n_layers_lstm, batch_size, self.n_hidden_lstm).to(device))
        
        return hidden
        
    
    
    def train_model(self, 
                    dataloader, 
                    optimizer,
                    batch_size = 16,
                    n_epochs = 10):
        
        dataloaders = self.preprocess_data_class_instance.data_loaders( 
                                batch_size = batch_size, split=(0.85, 0.10))     
        (train_loader, valid_loader, test_loader) = dataloaders
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr = 0.001)
        
        step = 0
        print_every = 500
        min_loss_v = np.inf
        self.train()
        
        for e in range(n_epochs):
            
            for chorale_tensor, meta_tensor in train_loader:
                step += 1
                
                if torch.cuda.is_available:
                    chorale_tensor = chorale_tensor.cuda().long()
                    meta_tensor = meta_tensor.cuda().long()
            
                batch_size, n_voices, seq_length = chorale_tensor.shape
                
                ########### 1. SPLIT UP LEFT, RIGHT, & CENTER NOTES ###########
                # small random shift in ticks: 
                #    number of more ticks to input in lstm_left (past) than lsmt_right (future)
                random_shift = random.randint(0, self.preprocess_data_class_instance.subdivision)
                split_index = seq_length // 2 + random_shift
                
                left_notes = chorale_tensor[:, :, :split_index].transpose(1,2)
                right_notes_index = torch.LongTensor(np.arange(seq_length, split_index, -1))
                right_notes = chorale_tensor.index_select(2, right_notes_index).transpose(1,2)
            
                if self.n_voices == 1:
                    center_notes = None  
                else:
                    center_voice_ids = [i for i in range(self.n_voices) if not i == self.voice_index]
                    center_voice_ids = torch.LongTensor(center_voice_ids)
                    center_notes = chorale_tensor[:, :, split_index].index_select(1, center_voice_ids)
            
                notes = (left_notes, center_notes, right_notes)
            
                ##################### 2. DEFINE NOTE LABEL ####################
                label = chorale_tensor[:, self.voice_index, split_index]
                 
                ########### 3. SPLIT UP LEFT, RIGHT, & CENTER METAS ###########
                meta_tensor_single_voice = meta_tensor[:, self.voice_index, :, :]
                
                left_metas = meta_tensor_single_voice[:, :, :split_index, :]
                right_metas = meta_tensor_single_voice.index_select(2, right_notes_index)
                center_metas = meta_tensor_single_voice[:, :, split_index, :]
                
                metas = (left_metas, center_metas, right_metas)

                ############## 4. RUN INPUTS THROUGH THE NETWORK ##############
                optimizer.zero_grad()
                note_probs = self.forward((notes, metas))
                
                loss = criterion(note_probs, label)
                loss.backward()
                
                nn.utils.clip_grad_norm(self.parameters(), clip = 5)
                optimizer.step()

                ################# 5. VALIDATE & PRINT PROGRESS ################                
                if step % print_every == 0:
                    valid_losses = []
                    self.eval()
                    
                    for chorale_tensor_v, meta_tensor_v in valid_loader: 
                        ########### 5.1 SPLIT UP LEFT, RIGHT, & CENTER NOTES ###########
                        left_notes_v = chorale_tensor_v[:, :, :split_index].transpose(1,2)
                        right_notes_v = chorale_tensor_v.index_select(2, right_notes_index).transpose(1,2)
                    
                        if self.n_voices == 1:
                            center_notes_v = None  
                        else:
                            center_notes_v = chorale_tensor_v[:, :, split_index].index_select(1, center_voice_ids)
                    
                        notes_v = (left_notes_v, center_notes_v, right_notes_v)
                    
                        ##################### 5.2 DEFINE NOTE LABEL ####################
                        label_v = chorale_tensor_v[:, self.voice_index, split_index]
                        
                        ########### 5.3 SPLIT UP LEFT, RIGHT, & CENTER METAS ###########
                        meta_tensor_single_voice_v = meta_tensor_v[:, self.voice_index, :, :]
                        
                        left_metas_v = meta_tensor_single_voice_v[:, :, :split_index, :]
                        right_metas_v = meta_tensor_single_voice_v.index_select(2, right_notes_index)
                        center_metas_v = meta_tensor_single_voice_v[:, :, split_index, :]
                        
                        metas_v = (left_metas_v, center_metas_v, right_metas_v)

                        ########### 5.4 RUN INPUTS THROUGH THE FROZEN NETWORK ###########
                        note_probs_v = self.forward((notes_v, metas_v))
                        loss_v = criterion(note_probs_v, label_v)
                        valid_losses.append(loss_v)
                        
                    avg_loss_v = np.mean(valid_losses)
                    if avg_loss_v < min_loss_v:
                        min_loss_v = avg_loss_v
                        torch.save(self.state_dict(), 'models/' + "model_epoch" + e)
                        
                    print("Epoch: {}/{}.".format(e, n_epochs), 
                          "Training_Loss: ", loss.item(),
                          "Valid Loss: ", avg_loss_v)
                    
                    self.train()
                
        
        
    def __repr__(self):
        return f'VoiceModel(' \
               f'{self.dataset.__repr__()},' \
               f'{self.main_voice_index},' \
               f'{self.note_embedding_dim},' \
               f'{self.meta_embedding_dim},' \
               f'{self.num_layers},' \
               f'{self.lstm_hidden_size},' \
               f'{self.dropout_lstm},' \
               f'{self.hidden_size_linear}' \
               f')'
    
    
    
    
    def load(self):
        state_dict = torch.load('models/' + self.__repr__())
        print(f'Loading {self.__repr__()}')
        self.load_state_dict(state_dict)
        
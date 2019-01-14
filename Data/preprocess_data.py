import numpy as np
from tqdm import tqdm

import music21
from music21 import interval #, stream

import torch
from torch.utils.data import TensorDataset

from Data.helper_data import standard_name, standard_note, SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, OUT_OF_RANGE, REST_SYMBOL                       
#from Data.metadata import FermataMetadata


class PreprocessData():
    def __init__(self, 
                 corpus_iterator,
                 dataset_name,
                 voice_ids,
                 cache_dir,
                 metadatas = None,
                 seq_size = 8,
                 subdivision = 4):
        """
        :param corpus_iterator: calling this function returns an iterator over chorales 
                              (as music21 scores)
        :param dataset_name: name of the dataset
        :param voice_ids: list of voice_indexes to be used
        :param metadatas: list[Metadata], the list of used metadatas
        :param seq_size: in beats
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        self.corpus_iterator = corpus_iterator
        self.dataset_name = dataset_name
        self.num_voices = len(voice_ids)
        self.metadatas = metadatas
        self.seq_size = seq_size
        self.subdivision = subdivision
        self.cache_dir = cache_dir
        
        self.index2note_dicts = None
        self.note2index_dicts = None
        self.voice_ranges = None  # in midi pitch
    
    def is_valid(self, chorale):
        ''' A chorale must be 4-part to be valid '''
        if not len(chorale.parts) == 4:
            return False
        
        return True
    
        
    def get_iterator(self):
        iterator = (chorale for chorale in self.corpus_iterator() if self.is_valid(chorale))
        
        return iterator
    
    
    def compute_index_dicts(self):
        
        print("Computing index dicts...")
        
        self.note2index_dicts = [{} for _ in range(self.num_voices)]
        self.index2note_dicts = [{} for _ in range(self.num_voices)]
        
        # n sets of unique notes, for n voices in the chorale:
        note_sets = [set() for _ in range(self.num_voices)]
        # Add additional notes to each of the n sets
        for note_set in note_sets:
            note_set.add(SLUR_SYMBOL)
            note_set.add(START_SYMBOL)
            note_set.add(END_SYMBOL)
            note_set.add(REST_SYMBOL)
        
        # Iterate over the ENTIRE corpus of chorales:
        #    so we get ALL unique notes used in each voice across the entire corpus
        for chorale in tqdm(self.get_iterator()):
            for voice_i, part in enumerate(chorale.parts[:self.num_voices]):
                for element in part.flat.notesAndRests:
                    note_sets[voice_i].add(standard_name(element))    
                    
        for voice_i in range(self.num_voices):
            self.note2index_dicts[voice_i] = {note:note_i for note_i, note 
                                                in enumerate(note_sets[voice_i])}
            
            self.index2note_dicts[voice_i] = {note_i:note for note_i, note 
                                                in enumerate(note_sets[voice_i])}
        
        
        
    def compute_voice_ranges(self):
        
        assert self.index2note_dicts is not None
        assert self.note2index_dicts is not None
        
        # list of n lists, w/ each list containing all unique notes in voice i:
        note_lists = [[standard_note(note) for note in note_dict] 
                            for note_dict in self.note2index_dicts]
        
        pitch_lists = [[note.pitch.midi for note in note_list if note.isNote] 
                            for note_list in note_lists]
        
        self.voice_ranges = [(min(pitch_list), max(pitch_list)) for pitch_list in pitch_lists]
      
        
        
    def voice_range_in_part(self, part, offsetStart, offsetEnd):
        
        notes = part.flat.getElementsByOffset(offsetStart,
                                              offsetEnd,
                                              includeEndBoundary = False,
                                              mustBeginInSpan = True,
                                              mustFinishInSpan = False,
                                              classList = [music21.note.Note,
                                                           music21.note.Rest])
            
        pitches = [note.pitch.midi for note in notes if note.isNote]
        
        if len(pitches) > 0:
            return (min(pitches), max(pitches))
        else:
            return None
    
    
    def voice_range_in_subsequence(self, chorale, offsetStart, offsetEnd):
        ''' Returns None if no note present in any one of the voices --> no transposition '''
        
        voice_ranges = []
        
        for part in chorale.parts[:self.num_voices]:
            voice_range_part = self.voice_range_in_part(part, offsetStart, offsetEnd)
            
            if voice_range_part is None:
                return None
            else:
                voice_ranges.append(voice_range_part)
        
        return voice_ranges
            


    ''' Used by make_tensor_dataset '''
    def min_max_transposition(self, seq_ranges_current):
        ''' Returns: (minimum transposition, maximum transposition), in pitches
        '''
        if seq_ranges_current is None:
            transposition = (0,0)
        else:
            # self.voice_ranges: a list of n sets (min_pitch, max_pitch) for n voices
            # transpositions: a list of 4 sets -
            #      [(min_v1, max_v1), ... , (min_v4, max_v4)]
            transpositions = [(min_pitch_corpus - min_pitch_current,
                               max_pitch_corpus - max_pitch_current)
            
                              for ((min_pitch_corpus, max_pitch_corpus),
                                   (min_pitch_current, max_pitch_current))
                              
                              in zip(self.voice_ranges, seq_ranges_current)]
            
            # Convert transpositions into a list of 2 sets -
            #      [(min_v1, minv2, minv3, minv4), (max_v1, max_v2, max_v3, max_v4)]
            transpositions = [trans for trans in zip(*transpositions)]
            # max of min, min of max
            transposition = [max(transpositions[0]), min(transpositions[1])]
            
        return transposition
        
    
    
    ''' used by score_to_tensor '''
    def part_to_tensor(self, part, part_i, offsetStart, offsetEnd):
        
        ### 1. Obtain list of notes in a part of a transposed chorale:
        notes = list(part.flat.getElementsByOffset(offsetStart,
                                                   offsetEnd,
                                                   classList = [music21.note.Note,
                                                                music21.note.Rest]))
        
        note_names_and_pitches = [(note.nameWithOctave, note.pitch.midi)
                                          for note in notes if note.isNote]
        
        # length of the part in quarter subdivisions
        length = int((offsetEnd - offsetStart) * self.subdivision) 
    
        ### 2. Add entries to dictionaries if not present (since chorale is transposed):
        ### (should only be called by make_tensor_dataset when transposing)
        
        note2index = self.note2index_dicts[part_i]
        index2note = self.index2note_dicts[part_i]
        voice_range = self.voice_ranges[part_i]
        min_pitch, max_pitch = voice_range
        
        for note_name, pitch in note_names_and_pitches:
            # if a transposed note is out of pitch range:
            if (pitch < min_pitch) or (pitch > max_pitch):
                note_name = OUT_OF_RANGE
            
            # if a transposed note is is within range,
            #    but was never played within the untransposed chorale:
            if note_name not in note2index:
                new_index = len(note2index)
                index2note.update({new_index : note_name})
                note2index.update({note_name : new_index})
                print('Warning: Entry ' + str({new_index : note_name}) +
                              ' added to dictionaries')
            
        ### 3. Construct note sequence:
        note_seq = np.zeros((length,2)) # 1 column for note index, 1 column for is_articulated
        
        is_articulated = True
        notes_and_rests = part.flat.notesAndRests
        num_notes = len(notes_and_rests)

        subdiv_i, note_i = 0, 0
    
        while subdiv_i < length: # length = length of chorale in subdivisions
            if note_i < num_notes - 1:
                # if the next note starts after the current subdiv_i
                if notes_and_rests[note_i + 1].offset * self.subdivision > subdiv_i:
                    note_seq[subdiv_i, :] = [note2index[standard_name(notes_and_rests[note_i])], 
                                              is_articulated]
                    subdiv_i += 1
                    is_articulated = False
                # if the next note starts at the current subdiv_i
                else:
                    note_i += 1
                    is_articulated = True
                    
            else:  # Last note in the chorale voice
                note_seq[subdiv_i, :] = [note2index[standard_name(notes_and_rests[note_i])],
                                         is_articulated]
                subdiv_i += 1
                is_articulated = False
        
        seq = [note[0] if note[1] else note2index[SLUR_SYMBOL] for note in note_seq]
        seq = np.array(seq)
        
        ### 4. Convert the sequence into a tensor & give it a voice_id dimension
        tensor = torch.from_numpy(seq).long().unsqueeze_(0)

        return tensor                
                
                
    '''
    chorale_tensor = self.score_to_tensor(
                            chorale_transposed, 
                            offsetStart = 0.,
                            offsetEnd = chorale_transposed.flat.highestTIme)

    '''
    ''' Used by transposed_score_and_metadata_tensors '''
    def chorale_to_tensor(self, chorale_transposed, offsetStart, offsetEnd):
        
        chorale_tensor = []
        
        for part_i, part in enumerate(chorale_transposed.parts[:self.num_voices]):
    
            part_tensor = self.part_to_tensor(part,
                                              part_i,
                                              offsetStart = offsetStart,
                                              offsetEnd = offsetEnd)
            chorale_tensor.append(part_tensor)
        
        # Combine part_tensors in voice_id dimension
        chorale_tensor = torch.cat(chorale_tensor, 0)
        
        return chorale_tensor # shape: (num_voices, chorale_length)
    
    
    
    ''' Used by transposed_score_and_metadata_tensors '''
    def metadata_to_tensor(self, transposed_score):
        
        metadatas = []
        ### 1. Obtain a sequence of metadata values 
        if self.metadatas:
            for metadata in self.metadatas:
                # shape : (length,)
                metadata_seq = torch.from_numpy(
                        metadata.evaluate(transposed_score, self.subdivision)
                                                ).long().clone()
                
                # duplicate metadata for each of the n voices in the score
                # (key, fermta, & beat metadatas are identical between voices)
                # shape: (num_voices, length)
                metadata_all_voices = metadata_seq.repeat(self.num_voices, 1)
                
                # shape: (num_voices, length, 1)
                metadatas.append(metadata_all_voices.unsqueeze_(-1))

            # at the end of for loop:
            #   metadatas: list of num_metadatas tensors of shape (num_voices, length, 1)
        
        length = int(transposed_score.duration.quarterLength * self.subdivision)
        
        ### 2. Add voice indexes:
        voice_ids = torch.from_numpy(np.arange(self.num_voices)).long().clone()
        voice_ids = voice_ids.repeat(length, 1)       # (length, num_voices)
        voice_ids = torch.transpose(voice_ids, 0, 1)  # (num_voices, length)
        voice_ids.unsqueeze_(-1)                      # (num_voices, length, 1)
        
        metadatas.append(voice_ids)
        # Now, metadatas: list of (num_metadatas+1) tensors of shape (num_voices, length, 1)
        
        all_metadatas_tensor = torch.cat(metadatas, 2)
    
        return all_metadatas_tensor # shape:(num_voices, length, num_metadatas+1)
        
    

    ''' Used by make_tensor_dataset '''
    def transposed_score_and_metadata_tensors(self, chorale, semitone):
        '''
        Convert chorale to a tuple: (chorale_tensor, metadata_tensor)
        --> the original chorale is transposed semitone number of semi-tones
        '''
        # 1. Transpose:
        #    Compute the most "natural" interval given # of semi-tones
        #    - interval_type: "M"-major, "m"-minor, "A"-augmented, "P"-perfect, "d"-diminished
        #    - interval_step: 1st, 2nd, 3rd, 4th, ...
        #    ex) ~(4): C-C#, C#-D, D-D#, D#-E --> C-E --> ('M',3) : majord 3rd
        interval_type, interval_step = interval.convertSemitoneToSpecifierGeneric(semitone)
        interval_name = interval_type + str(interval_step)
        trans_interval = interval.Interval(interval_name)
        
        chorale_transposed = chorale.transpose(trans_interval)
        # .highestTime: end time of the note w/ highestOffset (= last note)
        # usually = .quarterLength
        chorale_tensor = self.chorale_to_tensor(chorale_transposed, 
                                              offsetStart = 0.,
                                              offsetEnd = chorale_transposed.flat.highestTime)
        
        metadatas_tensor = self.metadata_to_tensor(chorale_transposed)
        
        return chorale_tensor, metadatas_tensor
    
    
    
    ''' Used by make_tensor_dataset '''
    def chorale_tensor_with_padding(self, 
                                    chorale_tensor, #(num_voices, length in ticks)
                                    start_tick,
                                    end_tick):
        
        assert start_tick < end_tick
        assert end_tick > 0
        
        length = chorale_tensor.size()[1]
        
        padded_chorale = []
        
        ''' Left Padding with START_SYMBOL'''
        if start_tick < 0:
            start_symbol_indexes = np.array([note2index[START_SYMBOL]
                                             for note2index in self.note2index_dicts])
            
            start_symbol_indexes = torch.from_numpy(start_symbol_indexes).long().clone()
            
            # shape: (-start_tick, num_voices)
            start_symbol_indexes = start_symbol_indexes.repeat(-start_tick, 1)
            
            # shape: (num_voices, -start_tick)
            start_symbol_indexes = start_symbol_indexes.transpose(0,1)
            padded_chorale.append(start_symbol_indexes)
            
        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length
        _slice = chorale_tensor[:, slice_start:slice_end] #(num_voices, <= seq_size)
        
        padded_chorale.append(_slice)
        
        ''' Right Padding with END_SYMBOL'''
        if end_tick > length:
            end_symbol_indexes = np.array([note2index[END_SYMBOL]
                                             for note2index in self.note2index_dicts])
            
            end_symbol_indexes = torch.from_numpy(end_symbol_indexes).long().clone()
            
            # shape: (end_tick-length, num_voices)
            end_symbol_indexes = end_symbol_indexes.repeat(end_tick - length, 1)
            
            # shape: (num_voices, end_tick-length)
            end_symbol_indexes = end_symbol_indexes.transpose(0,1)
            padded_chorale.append(end_symbol_indexes)
            
        padded_chorale = torch.cat(padded_chorale, 1)
        
        return padded_chorale  # shape: (num_voices, seq_size) 
    


    ''' Used by make_tensor_dataset '''
    def metadata_tensor_with_padding(self, 
                                    metadata_tensor, #(n_voices, length, n_metadata)
                                    start_tick,
                                    end_tick): 
        
        assert start_tick < end_tick
        assert end_tick > 0
        
        num_voices, length, num_metadatas = metadata_tensor.size()
        
        padded_metadata = []
        
        ''' Left Padding with 0'''
        if start_tick < 0:
            start_symbols = np.zeros((self.num_voices, -start_tick, num_metadatas))
            start_symbols = torch.from_numpy(start_symbols).long().clone()
            padded_metadata.append(start_symbols)
            
        slice_start = start_tick if start_tick > 0 else 0
        slice_end = end_tick if end_tick < length else length
        _slice = metadata_tensor[:, slice_start:slice_end, :] #(num_voices, <=seq_size, n_metadata)
        
        padded_metadata.append(_slice)
        
        ''' Right Padding with 0'''
        if end_tick > length:
            end_symbols = np.zeros((self.num_voices, end_tick - length, num_metadatas))
            end_symbols = torch.from_numpy(end_symbols).long().clone()
            padded_metadata.append(end_symbols)
            
        padded_metadata = torch.cat(padded_metadata, 1)
        
        return padded_metadata  # shape: (num_voices, seq_size, n_metadata)    
        
        
    
    def make_tensor_dataset(self):
        
        print("Making Tensor Dataset...")
        
        self.compute_index_dicts()
        self.compute_voice_ranges()
        
        one_tick = 1/self.subdivision
        
        chorale_tensor_dataset = []
        metadata_tensor_dataset = []
        
        chorale_iterator = self.get_iterator()
        
        for chorale_i, chorale in enumerate(chorale_iterator): 
            '''
            Quoting DeepBach Paper:
                "We choose to augment dataset by adding all chorale transpositions 
                which fit the vocal ranges defined by the intial corpus"
            '''
            chorale_transpositions = {}
            metadata_transpositions = {}
            
            lowest_offset = chorale.flat.lowestOffset # beat at which first note starts 
            highest_offset = chorale.flat.highestOffset # beat at which last note starts
            
            lowest_offset = lowest_offset - (self.seq_size - one_tick)
            
            offset_i = 1
            semitone_i = 1
            
            for offsetStart in np.arange(lowest_offset, highest_offset, one_tick):
                #print("offset: ", offset_i)
                # if start_tick < 0, music21 automatically converts it to 0
                offsetEnd = offsetStart + self.seq_size
                seq_ranges_current = self.voice_range_in_subsequence(chorale,
                                                                     offsetStart,
                                                                     offsetEnd)
                transposition = self.min_max_transposition(seq_ranges_current)
                min_trans_seq, max_trans_seq = transposition
                
                for semitone in range(min_trans_seq, max_trans_seq + 1):
                    #print("semitone: ", semitone_i)
                    start_tick = int(offsetStart * self.subdivision) # could be negative
                    end_tick = int(offsetEnd * self.subdivision)
                    
                    try:
                        if semitone not in chorale_transpositions:
                            # chorale_tensor: (num_voices, length)
                            # metadata_tensor: (num_voices, length, num_metadatas+1)
                            
                            (chorale_tensor, 
                             metadata_tensor) = self.transposed_score_and_metadata_tensors(
                                                         chorale, 
                                                         semitone = semitone)

                            chorale_transpositions.update({semitone : chorale_tensor})
                            metadata_transpositions.update({semitone : metadata_tensor})
                            
                        
                        else:
                            chorale_tensor = chorale_transpositions[semitone]
                            metadata_tensor = metadata_transpositions[semitone]
                            
                        chorale_tensor_temp = self.chorale_tensor_with_padding(chorale_tensor,
                                                                              start_tick, 
                                                                              end_tick)
                        
                        metadata_tensor_temp = self.metadata_tensor_with_padding(metadata_tensor,
                                                                                start_tick, 
                                                                                end_tick)
                            
                        # Add batch dimension to tensor & Append to dataset list:
                        # shape: (1, num_voices, seq_size)    
                        chorale_tensor_temp.unsqueeze_(0).int()
                        chorale_tensor_dataset.append(chorale_tensor_temp)
                        
                        metadata_tensor_temp.unsqueeze_(0).int()
                        metadata_tensor_dataset.append(metadata_tensor_temp)
                        
                    except KeyError:
                        # (quoting orig. author from github)
                        # "some problems may occur with the key analyzer"
                        print(f"KeyError with chorale {chorale_i}")
                        
                    semitone_i += 1
                offset_i += 1
        
        # Combine across batch dimension               
        chorale_tensor_dataset = torch.cat(chorale_tensor_dataset, 0)
        metadata_tensor_dataset = torch.cat(metadata_tensor_dataset, 0)

        dataset = TensorDataset(chorale_tensor_dataset,
                                metadata_tensor_dataset)

        print(f'Sizes: {chorale_tensor_dataset.size()}, {metadata_tensor_dataset.size()}')
        
        return dataset
        
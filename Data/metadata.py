import numpy as np
from music21 import analysis, stream


class TickMetadata():
    def __init__(self, subdivision):
        self.is_global = False
        self.num_values = subdivision
        self.name = 'tick'
        
    def evaluate(self, chorale, subdivision):
        ''' Return: array of tick values in range 0-3 '''
        assert subdivision == self.num_values
        
        length = int(chorale.duration.quarterLength * subdivision)
        ticks = [beat % self.num_values for beat in range(length)] 
        
        return np.array(ticks)
        
    def generate(self, length):
        ticks = [beat % self.num_values for beat in range(length)] 
        
        return np.array(ticks)
    
    
    
class FermataMetadata():
    def __init__(self):
        self.is_global = False
        self.num_vales = 2
        self.name = 'fermata'
        
    def evaluate(self, chorale, subdivision):
        part = chorale.parts[0] # soprano
        length = int(chorale.duration.quarterLength * subdivision)
        notes = part.flat.notes
        num_notes = len(notes)
        subdiv_i = 0
        note_i = 0
        fermatas = np.zeros(length)
        fermata = False
        
        while subdiv_i < length:
            if note_i < num_notes - 1:
                # if the next note starts after the current subdiv_i
                if notes[note_i + 1].offset * subdivision > subdiv_i:
                    # if there is fermata sign above the note, 
                    #   note.expressions = [<music21.expressions.Fermata>]
                    if len(notes[note_i].expressions) == 1:
                        fermata = True
                    else:
                        fermata = False
                        
                    fermatas[subdiv_i] = fermata
                    subdiv_i += 1
                # if the next note starts at the current subdiv_i
                else:
                    note_i += 1
            
            else: # at the last note
                if len(notes[note_i].expressions) == 1:
                    fermata = True
                else:
                    fermata = False
                    
                fermatas[subdiv_i] = fermata
                subdiv_i += 1
    
        return np.array(fermatas, dtype = np.int32)
    
    def generate(self, length):
        ''' Return: Fermata every 2 bars '''
        # fermata = True if subdiv = 29, 30, 31, 61, 62, 63, ...
        # Usually time signiture : 4 / 4 -> 4*4 = 16 subdivs in 1 bar
        return np.array( [1 if i % 32 > 28 else 0 for i in range(length)] )
        
    
    
class KeyMetadata():
    def __init__(self, window_size = 4):
        self.is_global = False
        self.window_size = window_size
        self.num_max_sharps = 7
        self.num_values = 16
        self.name = 'key'
        
    def get_index(self, value):
        '''
        value : # of sharps (between -7 & 7)
        Return : index in the representation (between 1 & 15)
        ** Representation : range 1 (7 flats) to 8 (no sharps or flats) to 15 (7 sharps)
        '''
        return value + self.num_max_sharps + 1
    
    def get_value(self, index):
        '''
        value : index in the representation (between 1 & 15)
        Return : # of sharps (between -7 & 7)
        ** Representation : range 1 (7 flats) to 8 (no sharps or flats) to 15 (7 sharps)
        '''
        return index - self.num_max_shapes - 1
      
    def evaluate(self, chorale, subdivision):
        chorale_with_measures = stream.Score()
        for part in chorale.parts:
            chorale_with_measures.append(part.makeMeasures())
        
        keyAnalysis = analysis.floatingKey.KeyAnalyzer(chorale_with_measures)
        keyAnalysis.windowSize = self.window_size
        res = keyAnalysis.run()
        
        measure_offset_map = chorale_with_measures.parts.measureOffsetMap()
        
        length = int(chorale.duration.quarterLength * subdivision)
        
        key_signatures = np.zeros(length)
        
        measure_i = -1
        
        for subdiv_i in range(length):
            beat_i = subdiv_i / subdivision
            
            if beat_i in measure_offset_map:
                measure_i += 1
                
                if measure_i == len(res):
                    measure_i -= 1
                    
            key_signatures[subdiv_i] = self.get_index(res[measure_i].sharps)
            
        return np.array(key_signatures, dtype = np.int32)
    
    def generate(self, length):
        ''' return C major '''
        return np.full(length, self.get_index(0))
    
    


class IsPlayingMetadata():
    def __init__(self, voice_index, min_num_ticks):
        '''
        Determines if a SINGLE Voice i IS PLAYING OR NOT
        Voice i is considered to be **MUTED** if:
            there are more than 'window_size' consective subdivisions that contains a REST
        min_num_ticks : minimum length in ticks(quarter subdivisions) for a 
                        REST to be taken into account in Metadata
        '''
        self.voice_index = voice_index
        self.min_num_ticks = min_num_ticks
        self.is_global = False
        self.num_values = 2
        self.name = 'isplaying'
        
    def convert_to_int(self, value):  
        ''' Convert True -> 1 & False -> 0 '''
        return int(value)
    
    def convert_to_bool(self, index):
        ''' Convert 1 -> True & 0 -> False '''
        return bool(index)
    
    def evaluate(self, chorale, subdivision):
        '''  Input: a music21 chorale (w/ all 4 voices)- shape: (4, length) '''
        length = int(chorale.duration.quarterLength * subdivision)
        metadatas = np.ones(length)  # 1 metadata / 1 subdivision
        voice = chorale.parts[self.voice_index]  # one of 4 voice parts- (1, length)
        
        for note_or_rest in voice.notesAndRests:  # for each Chorale Note
            is_playing = True
            
            if note_or_rest.isRest:
                if (note_or_rest.quarterLength * subdivision) >= self.min_num_ticks:
                    is_playing = False
            
            start_tick = note_or_rest.offset * subdivision
            end_tick = start_tick + note_or_rest.quarterLength * subdivision
            metadatas[start_tick:end_tick] = self.convert_to_int(is_playing)
            
        return metadatas
    
    def generate(self, length):
        ''' return is_playing = True for ALL notes '''
        return np.ones(length)
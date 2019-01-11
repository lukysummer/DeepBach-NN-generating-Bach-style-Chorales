from muisc21 import note, expressions, harmony, corpus
from itertools import islice

### CONSTANTS ###
SLUR_SYMBOL = "__"
START_SYMBOL = "START"
END_SYMBOL = "END"
REST_SYMBOL = "rest"
OUT_OF_RANGE = "OOR"
PAD_SYMBOL = "XX"


def standard_name(element, voice_range = None):
    ''' 
    Converts music21 objects to str (name)
    :param element     : note or rest
    :param voice_range : (min_pitch, max_pitch) -- optional
    :return : given element's name in string
    '''
    if isinstance(element, note.Note):
        if voice_range is not None:
            min_pitch, max_pitch = voice_range
            pitch = element.pitch.midi
            
            if pitch < min_pitch or pitch > max_pitch:
                return OUT_OF_RANGE
            
        return element.nameWithOctave
    
    if isinstance(element, note.Rest):
        return element.name # = "rest"
    
    if isinstance(element, str):
        return element
    
    if isinstance(element, harmony.ChordSymbol):
        return element.figure
    
    if isinstance(element, expressions.TextExpression):
        return element.content


def standard_note(string):
    ''' 
    Converts str (name) representing a music21 object to the corresponding object 
    :param string : str (name) representing a music21 object 
    :return : corresponding music21 object
    '''
    if string == "rest":
        return note.Rest()
    
    ### Treat additional symbols as RESTS ###
    if (string == START_SYMBOL) or (string == END_SYMBOL) or (string == PAD_SYMBOL):
        return note.Rest()
    
    if string == SLUR_SYMBOL:
        return note.Rest()
        
    if string == OUT_OF_RANGE:
        return note.Rest()
        
    else:
        return note.Note(string)
    
    
class ShortChoraleIteratorGen:
    '''
    Class used for DEBUGGING:
        when called, returns an Iterator over 3 Bach Chorales
    '''
    def __init__(self):
        pass
    
    def __call__(self):
        iterator = (chorale for chorale in islice(corpus.chorales.Iterator(), 3))
        
        return iterator.__iter__()
    
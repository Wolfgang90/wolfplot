import numpy as np
import pandas as pd
import os
#import itertools

class EventLog:
    """
        Initialize class
    """    
    def __init__(self, path_ev):
        # Eventlog path
        self.path_ev = path_ev
        
    """
        Loads eventlog data
    """            
    def load_eventlog(self, sep = ",", encoding="utf-8", convert_format='%d.%m.%y %H:%M:%S',convert = None):
        filename, file_extension = os.path.splitext(self.path_ev)
        if file_extension == '.csv':
            self._load_csv(sep,encoding, convert_format, convert)
        else:
            print('The file type ', file_extension, ' is currently not supported')
            
    def _load_csv(self, sep,encoding,convert_format, convert):
        # Create inline function for dateparse to pandas dataframe
        to_datetime = lambda x: pd.datetime.strptime(x, convert_format)
        
        # Load csv
        self.eventlog = pd.read_csv(self.path_ev, sep, encoding = encoding, converters={convert: to_datetime})

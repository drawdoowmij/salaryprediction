class LoadData:
    ''' This class will load the csv files dynamically. 
        It accepts a variable number of arguments.'''
    
    def __init__(self, *args):
        self.args = args
                
    def load(self):
        list_of_df=[]
        for x in self.args:
            fname = pd.read_csv(x)
            list_of_df.append(fname)
            
        return list_of_df[0], list_of_df[1], list_of_df[2]
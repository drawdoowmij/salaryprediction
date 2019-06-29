class OutOfSample:
    """ Uses the input dataframe and contructs features to be used in the 
        selected models.  This class can end up being quite extensive depending
        on the number of features that need to be created.

    Parameters
    ----------
    df: feature DataFrame 
    model: best performing estimator
    jtdict: dictionary to convert job type to a number
    degdict: dictionary to convert degree to a number
    le:  label encoder
            
    Returns
    -------
    Nothing
        
    """
    def __init__(self, df, model, jtdict, degdict, le):
        self.df         = df
        self.model      = model
        self.jtdict     = jtdict
        self.degdict    = degdict
        self.le         = le
        self.res        = pd.DataFrame()
        self.predict    = None
        
    def basic_feat(self):
        self.res = pd.read_csv('Salary_Feature.csv')
        ## encode major and industry
        self.df['major'] = self.le.fit_transform(self.df['major'])
        self.df['industry'] = self.le.fit_transform(self.df['industry'])
        ## convert jobType to numeric value
        self.df['jobType'] = self.df['jobType'].map(self.jtdict)
        self.df['degree']  = self.df['degree'].map(self.degdict)
        ## dont need jobId and companyId
        self.df.drop(['jobId', 'companyId'], axis=1, inplace=True)
        self.df = pd.concat([self.df, self.res['num_1'], self.res['cat_1']], axis=1)
        self.df.dropna(inplace=True)
                
    def run_model(self):
        print('Running model on test data ....')
        self.inst = self.model
        self.predict = self.inst.predict(self.df)
        print('Model finished -- the first 10 predicted values are {}'.format(self.predict[:10]))
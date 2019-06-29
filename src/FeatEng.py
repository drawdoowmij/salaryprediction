class FeatEng:
    """ Uses the input dataframe and contructs features to be used in the 
        selected models.  This class can end up being quite extensive depending
        on the number of features that need to be created.

    Parameters
    ----------
    df: feature DataFrame 
    jtdict: dictionary to convert job type to a number
    degdict: dictionary to convert degree to a number
    le:  label encoder
        
    Returns
    -------
    Nothing
        
    """
    def __init__(self, df, jtdict, degdict, le):
        self.df         = df
        self.jtdict     = jtdict
        self.degdict    = degdict
        self.le         = le
        self.df_cat     = pd.DataFrame()
        self.df_num     = pd.DataFrame()
        self.res        = pd.DataFrame()
                
                
    def feat_engineering(self):
        ## encode major and industry
        self.df['major'] = self.le.fit_transform(self.df['major'])
        self.df['industry'] = self.le.fit_transform(self.df['industry'])
        ## convert jobType to numeric value
        self.df['jobType'] = self.df['jobType'].map(self.jtdict)
        self.df['degree']  = self.df['degree'].map(self.degdict)
        g_obj_cat = self.df.groupby(['companyId', 'degree', 'industry', 'jobType', 'major'])
        g_obj_num = self.df.groupby(['milesFromMetropolis', 'yearsExperience'])
        self.df_cat['cat_1'] = g_obj_cat['salary'].mean()
        self.df_num['num_1'] = g_obj_num['salary'].mean()
        self.df_cat = self.df_cat.reset_index() 
        self.df_num = self.df_num.reset_index()
        ## create new dataframe with added features
        self.res = pd.merge(self.df, self.df_cat, 
                            on=['companyId', 'degree', 'industry', 'jobType', 'major'], how='left')
        self.res = pd.merge(self.res, self.df_num, 
                            on=['milesFromMetropolis', 'yearsExperience'], how='left')
        ## put columns in proper sequence
        cols = self.res.columns.to_list()
        format_cols = cols[:8] + cols[-2:-1] + cols[-1:] + cols[-3:-2]
        self.res = self.res[format_cols]
        ## dont need jobId and companyId
        self.res.drop(['jobId', 'companyId'], axis=1, inplace=True)
        self.res.to_csv('Salary_Feature.csv')
        
                
    def check_corr(self):
        ## check the correlation between features
        plt.figure(figsize=(12,8))
        sns.set(font_scale=1.4)
        sns.heatmap(round(self.res.corr(), 2), annot=True, 
                    cmap='coolwarm', annot_kws={'size':15}, vmax=0.6)
        plt.savefig('{}/{}'.format('images', 'feat_corr.png'))
        plt.show();
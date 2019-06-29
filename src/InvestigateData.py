class InvestigateData:
    """Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    train_feat : DataFrame
        Training dataframe containing all the fields to build our features with
    test_feat : DataFrame
        Testing dataframe to be used when we have a finished model
    train_sal : DataFrame
        Training dataframe containing the salaries, to be used as targets
    

    Returns
    -------
    DataFrames -- self.feat_sal_return, self.basecase_merge
        
    """    
        
    def __init__(self, train_feat, test_feat, train_sal):
        self.train_feat = train_feat
        self.test_feat  = test_feat
        self.train_sal  = train_sal
        self.feat_sal_return = pd.DataFrame()
                
    @staticmethod   
    def get_data_info(**kwargs):
        for i, j in kwargs.items():
            print(10*'=' + ' ' + color.RED + i + color.END + color.BOLD + 
                  ' first 5 rows ' + color.END + 10*'=' + '\n')
            display(print(j.head()))
            print(10*'=' + ' ' + color.RED + i + color.END + color.BOLD + 
                  ' Information ' + color.END + 10*'=' + '\n')
            print(j.info())
            print(10*'=' + ' ' + color.RED + i + color.END + color.BOLD + 
                  ' Description ' + color.END  + 10*'=' + '\n')
            print(j.describe())
            print()
            
    def data_investigate(self):
        ## merge salary in with the training data
        self.basecase_merge = self.train_feat.merge(self.train_sal, how='left', on='jobId')
        self.feat_sal_merge = self.train_feat.merge(self.train_sal, how='left', on='jobId')
        self.out_feat_sal_merge = self.test_feat.merge(self.train_sal, how='left', on='jobId')
        
        self.working_df = self.feat_sal_merge
                    
        #look for missing data or not a number
        if self.working_df.isna().sum()[0] == 0:
            print(color.BOLD + 'No records with NaN found.' + color.END)
            print('\n')
        else:
            print(color.RED + 'Attention: Need to deal with NaN values.' + color.END)
            print('\n')
        #look for duplicate data
        if self.working_df.duplicated().sum() == 0:
            print(color.BOLD + 'No duplicate records found.' + color.END)
            print('\n')
        else:
            print(color.RED + 'Attention: Need to deal with duplicates.' + color.END)
            print('\n')
            
        ## check value counts
        print(color.BOLD + 'Value Counts by Degree' + color.END)
        print(self.working_df['degree'].value_counts())
        print('\n')
        print(color.BOLD + 'Value Counts by Major' + color.END)
        print(self.working_df['major'].value_counts())
        print('\n')
        print(color.BOLD + 'Value Counts by Industry' + color.END)
        print(self.working_df['industry'].value_counts())
            
        if len(self.working_df[self.working_df['salary'] == 0]) > 0:
            ## calculate the lower fence of a box plot, 
            ## we'll use this a the lower threshod for salary
            self.lower_fence = (self.working_df.describe()['salary'][4] - 
               (1.5 * (self.working_df.describe()['salary'][6] - 
                       self.working_df.describe()['salary'][4])))
            ## remove any entry where the salary is not greater than the lower fence
            self.mask_sal = self.working_df['salary'] > self.lower_fence
            self.working_df = self.working_df[self.mask_sal]
            ##print(self.feat_sal_merge[self.mask1])
            print('\n')
            print(color.BOLD + 'Salaries of 0 dollars are suspicious '  
                             + 'and we will drop all salaries that are ' 
                             + 'below the lower fence of the IQR.' + color.END)
            print('\n')
            
        self.feat_sal_return = self.working_df
            
        return (self.feat_sal_return, self.basecase_merge)
            
    def save_clean_file(self):
        self.feat_sal_merge.to_csv('Feat_Sal_Merge_Clean.csv')
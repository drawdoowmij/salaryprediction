class CompareEst:
    """ Uses the a list of estimators to ?????.

    Parameters
    ----------
    df: feature DataFrame 
    regest: list of estimators
            
    Returns
    -------
    Key for the minimum value for mean squared error.
        
    """
    def __init__(self, regest):
        self.regest = regest
        self.df = pd.read_csv('Salary_Feature.csv', header=0)
        ## create dataframes for the actual and the predicted salary values
        self.actual_predict_lr    = pd.DataFrame()
        self.actual_predict_ridge = pd.DataFrame()
        self.actual_predict_rfr   = pd.DataFrame()
        self.key_min = None
        self.models = {}
    
    @staticmethod
    def plot_feat_importance(df, x, y, style, title):
        plt.style.use(style)
        df.plot(x=x, y=y, kind='bar', figsize=(12,8))
        plt.title(title)
        plt.savefig('{}/{}.png'.format('images', title))
        plt.show();
        
    @staticmethod
    def plot_act_pred(lr_df, ridge_df, rfr_df):
        plt.figure(figsize=(12,8))
        sns.distplot(lr_df['actual'], hist = False, 
                     color = 'r', label = 'Actual')
        sns.distplot(lr_df['predicted'], hist = False, 
                     color = 'b', label = 'LR Predicted')
        sns.distplot(ridge_df['predicted'], hist = False, 
                     color = 'g', label = 'Ridge Predicted')
        sns.distplot(rfr_df['predicted'], hist = False, 
                     color = 'y', label = 'Random Forest Predicted')
        plt.title('Actual vs Predicted')
        plt.savefig('{}/{}'.format('images', 'act_pred.png'))
        plt.show();
    
    def setup_data_split(self):
        ## features
        self.X = self.df.loc[:,'jobType':'num_1']
        ## target 
        self.y = self.df['salary']
        ## split traning and testing portions of the dataframe
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                test_size=0.3, random_state=101)
                
    def fit_predict_score(self, k, v):
        ## instantiate, fit and predict using selected estimator
        self.inst = v
        self.inst.fit(self.X_train, self.y_train)
        self.predict = self.inst.predict(self.X_test)
        self.inst_scores = cross_val_score(self.inst, self.X_train, self.y_train, 
                                           scoring='neg_mean_squared_error', cv=5)
        print('Cross Validation MSE scores for {} are \n {}\n'.format(self.inst.__class__, 
                                                                      self.inst_scores))
        print('Avg MSE for {} \n is {}\n'.format(self.inst.__class__,
                                                 -np.mean(self.inst_scores)))
        self.models[k] = -np.mean(self.inst_scores)
        print('models dict = {}'.format(self.models))
        
    def save_est(self, key_min):
        ## save the model to disk
        filename = 'best_model.sav'
        pickle.dump(self.regest[key_min], open(filename, 'wb'))
        print('Model written to disk!')
        return filename
        
    def load_est(self, est_name):   
        ## load model from disk
        loaded_model = pickle.load(open(est_name, 'rb'))
        print('Loaded model from disk!')
        return loaded_model
        
        
    def compare_estimators(self):
        for key, value in self.regest.items():
            if key == 'lr':
                CompareEst.fit_predict_score(self, key, value)
                self.coef = pd.concat([pd.DataFrame(self.X_train.columns),
                                       pd.DataFrame(np.transpose(self.inst.coef_))], axis = 1)
                self.coef.columns = ['jobType', 'coef']
                self.coef = self.coef.sort_values('coef', ascending=False)
                ##  plot function for feature importance
                CompareEst.plot_feat_importance(self.coef, 'jobType', 'coef', 'fivethirtyeight', 
                                                           'Linear Regression Feature Importance')
                ##  save actual and predicted in a dataframe
                self.actual_predict_lr['actual'] = self.y_test
                self.actual_predict_lr['predicted'] = self.predict
            elif key == 'ridge':
                CompareEst.fit_predict_score(self, key, value)
                self.coef = pd.concat([pd.DataFrame(self.X_train.columns),
                                       pd.DataFrame(np.transpose(self.inst.coef_))], axis = 1)
                self.coef.columns = ['jobType', 'coef']
                self.coef = self.coef.sort_values('coef', ascending=False)
                ##  plot function for feature importance
                CompareEst.plot_feat_importance(self.coef, 'jobType', 'coef', 'seaborn-darkgrid', 
                                                           'Ridge Regression Feature Importance')
                ##  save actual and predicted in a dataframe
                self.actual_predict_ridge['actual'] = self.y_test
                self.actual_predict_ridge['predicted'] = self.predict
            elif key == 'rfr':
                CompareEst.fit_predict_score(self, key, value)
                self.coef = pd.concat([pd.DataFrame(self.X_train.columns),
                                       pd.DataFrame(np.transpose(self.inst.feature_importances_))], 
                                       axis = 1)
                self.coef.columns = ['jobType', 'coef']
                self.coef = self.coef.sort_values('coef', ascending=False)
                ##  plot function for feature importance
                CompareEst.plot_feat_importance(self.coef, 'jobType', 'coef', 'seaborn-dark-palette', 
                                                           'Random Forest Feature Importance')
                ##  save actual and predicted in a dataframe
                self.actual_predict_rfr['actual'] = self.y_test
                self.actual_predict_rfr['predicted'] = self.predict
            print(color.BOLD + 100*'=' + color.END)
            
            
        CompareEst.plot_act_pred(self.actual_predict_lr, 
                                 self.actual_predict_ridge, 
                                 self.actual_predict_rfr)
        
        ##  find best performing estimator -- has the lowest MSE
        self.key_min = min(self.models.keys(), key=(lambda x: self.models[x]))
        print('min value in models = {}, estimator = {}'.format(self.models[self.key_min], 
                                                                self.regest[self.key_min]))
        return self.key_min
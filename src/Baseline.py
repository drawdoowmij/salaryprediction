class Baseline:
    """ Runs a basline linear regression model with limited features 
        just to get a basline measure of the mean squared errorr.
        Future models will be compared to this one for validity.

    Parameters
    ----------
    df: feature DataFrame 
    est: estimator to use for baseline measure
    yrsExp:  feature to use
    mileMetro:  feature to use
    sal: target
           
    Returns
    -------
    Nothing
        
    """
    def __init__(self, df, est, yrsExp, milesMetro, sal):
        self.df         = df
        self.sal        = sal
        self.est        = est
        self.yrsExp     = yrsExp
        self.milesMetro = milesMetro
        
                
    def baseline_model(self):
        ## baseline features
        X = self.df.loc[:, self.yrsExp:self.milesMetro]
        ## target
        y = self.df[self.sal]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        ## instantiate linear regression   
        lr = self.est
        lr.fit(X_train, y_train)
        ## Predict our model
        predict = lr.predict(X_test)
        lr_scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=5)
        print('\n')
        print(color.BOLD + 'Baseline Model Information' + color.END)
        print('Linear Regression score is {}'.format(lr.score(X_test, y_test)))
        print('The mean squared errors are {}'.format(lr_scores))
        print('The average mean squared error is {}'.format(-np.mean(lr_scores)))
        print('\n')
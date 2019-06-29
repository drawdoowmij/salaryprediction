class EDA:
    """ Does the exploratory data analysis

    Parameters
    ----------
    df: DataFrame to analyze
    jobT: column to use in analysis
    sal: column to use in analysis
    deg: column to use in analysis
    ind: column to use in analysis
    maj: column to use in analysis
    job_type: list of all the available job types
       
    Returns
    -------
    Nothing
        
    """
    def __init__(self, df, jobT, sal, deg, ind,  maj, job_type):
        self.df         = df
        self.sal        = sal
        self.jobT       = jobT
        self.ind        = ind
        self.deg        = deg
        self.maj        = maj
        self.job_type   = job_type
                
    def plot_sal(self):
        plt.figure(figsize=(15,8))
        sns.distplot(self.df[self.sal])
        plt.axvline(x=self.df[self.sal].mean(), linewidth=4, color='r')
        plt.text(self.df[self.sal].mean()+.1,0,'Mean')                 
        plt.xlabel('Salary', fontsize=16)
        plt.savefig('{}/{}'.format('images', 'salary_dist.png'))
        plt.show();
        print('The average overall salary is {}'.format(np.round(self.df[self.sal].mean()),decimals = 0))
        print('The salary standard deviation is {}'.format(np.round(self.df[self.sal].std()), decimals = 0))
        
    def plot_violin(self):
        violin_type = [self.deg, self.ind, self.maj]
            
        for i in violin_type:
            plt.figure(figsize=(15,8))
            sns.set(font_scale=1.5, palette='coolwarm')
            sns.violinplot(data=self.df, x=i, y=self.sal, hue=self.jobT, 
                           dodge=True, hue_order=self.job_type)
            plt.title('Salary Distribution By ' + str(i).title())
            plt.ylabel('Salary', fontsize=16)
            if i == 'industry':
                plt.xlabel('Industry', fontsize=16)
                fname = 'Industry_violin.png'
            elif i == 'degree':
                plt.xlabel('Level of Education', fontsize=16)
                fname = 'Degree_violin.png'
            elif i == 'major':
                plt.xlabel('Field of Study', fontsize=16)
                fname = 'Major_violin.png'
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), 
                       ncol=4, fancybox=True, shadow=True)
            plt.savefig('{}/{}'.format('images', fname))
            plt.show();
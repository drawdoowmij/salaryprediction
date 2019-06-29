# salaryprediction
Salary Prediction Project (Python)

<p>The goal of this project is to be able to predict a salary based off certain job related characteristics. The original feature input file consists of the following fields:<\p>

•	Jobid                    
•	Companyid             
•	JobType – values are CEO, CFO, CTO, Vice President, Manager, Senior, Junior, Janitor                
•	Degree – values are Doctoral, Masters, Bachelors, High School, or None             
•	Major – values are Biology, Business, Chemistry, Computer Science, Engineering, Literature, Math, Physics, or None                 
•	Industry– values are Auto, Education, Finance, Health, Oil, Service, or Web
•	YearsExperience – number of years in profession 
•	MilesFromMetropolis – number of miles from major metropolis 

The target file consists of the salaries for all the various entries in the feature file.

We will investigate the data and look for any missing values, outliers and anomalies.  We will also do some exploratory data analysis to look for relationships and trends in the data.  Once the suspect data has been identified and cleaned with the approval of the projects’ sponsors, we will run a baseline model to predict salaries.  The we will proceed with an iterative process of creating features, tuning the model parameters, running the model and assessing the accuracy of the model.

This project is split into 6 classes:

1.	Load the data
2.	Investigate the data
3.	EDA – Exploratory Data Analysis
4.	Baseline Model
5.	Feature Engineering 
6.	Compare Estimators

In this jupyter notebook they are listed sequentially but in a production environment they would be split into their respective libraries for ease of maintenance and modification.

Future revision would include adding docstrings to all the methods in every class, adding additional features, experimenting with other estimators, tighter error checking by using try-except blocks. Incorporating test driven development (TDD) just to mention a few.  


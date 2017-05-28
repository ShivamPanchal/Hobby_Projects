
# coding: utf-8

# # Black Friday- Sales Prediction
# 

# **Problem Statement**
# 
# A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.
# 
# 
# The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# 
# 
# Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# In[1]:

import os 
os.chdir("F://")
os.getcwd()


# ## Importing the Required Libraries

# In[2]:

import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().magic('matplotlib inline')


# ## Loading the dataset

# In[3]:

df = pd.read_csv("blackfriday.csv")


# ## Stage-1
# # Data Cleaning and Preprocessing

# In[4]:

df.shape


# In[5]:

df.head()


# In[6]:

df.describe()


# In[7]:

# Checking whether the dataset contains null values
df.isnull().values.any()


# In[8]:

df.dtypes         ## Find data types


# In[9]:

df.info()


# <a id='Missing Value Treatment'></a>
# ## Missing Value Treatment ##
#  
# ***Why missing values treatment is required?***
# 
# Missing data in the training data set can reduce the power / fit of a model or can lead to a biased model because we have not analysed the behavior and relationship with other variables correctly. It can lead to wrong prediction or classification.
# 
# ***Why my data has missing values?***
# 
# We looked at the importance of treatment of missing values in a dataset. Now, let’s identify the reasons for occurrence of these missing values. They may occur at two stages:
# 
# **Data Extraction**: It is possible that there are problems with extraction process. In such cases, we should double-check for correct data with data guardians. Some hashing procedures can also be used to make sure data extraction is correct. Errors at data extraction stage are typically easy to find and can be corrected easily as well.
# 
# **Data collection**: These errors occur at time of data collection and are harder to correct. They can be categorized in four types:
# 
# 1)**Missing completely at random**: This is a case when the probability of missing variable is same for all observations. For example: respondents of data collection process decide that they will declare their earning after tossing a fair coin. If an head occurs, respondent declares his / her earnings & vice versa. Here each observation has equal chance of missing value.
# 
# 2)**Missing at random**: This is a case when variable is missing at random and missing ratio varies for different values / level of other input variables. For example: We are collecting data for age and female has higher missing value compare to male.
# 
# 3)**Missing that depends on unobserved predictors**: This is a case when the missing values are not random and are related to the unobserved input variable. For example: In a medical study, if a particular diagnostic causes discomfort, then there is higher chance of drop out from the study. This missing value is not at random unless we have included “discomfort” as an input variable for all patients.
# 
# 4)**Missing that depends on the missing value itself**: This is a case when the probability of missing value is directly correlated with missing value itself. For example: People with higher or lower income are likely to provide non-response to their earning.
#  
# ***Which are the methods to treat missing values ?**
# 
# **Deletion**:  It is of two types: List Wise Deletion and Pair Wise Deletion.
# In list wise deletion, we delete observations where any of the variable is missing. Simplicity is one of the major advantage of this method, but this method reduces the power of model because it reduces the sample size.
# In pair wise deletion, we perform analysis with all cases in which the variables of interest are present. Advantage of this method is, it keeps as many cases available for analysis. One of the disadvantage of this method, it uses different sample size for different variables.
# 
# Deletion methods are used when the nature of missing data is “Missing completely at random” else non random missing values can bias the model output.
# 
# **Mean/ Mode/ Median Imputation**: Imputation is a method to fill in the missing values with estimated ones. The objective is to employ known relationships that can be identified in the valid values of the data set to assist in estimating the missing values. Mean / Mode / Median imputation is one of the most frequently used methods. It consists of replacing the missing data for a given attribute by the mean or median (quantitative attribute) or mode (qualitative attribute) of all known values of that variable. It can be of two types:- ***Generalized Imputation***: In this case, we calculate the mean or median for all non missing values of that variable then replace missing value with mean or median. Like in above table, variable “Manpower” is missing so we take average of all non missing values of “Manpower”  (28.33) and then replace missing value with it.
# Similar case Imputation: In this case, we calculate average for gender “Male” (29.75) and “Female” (25) individually of non missing values then replace the missing value based on gender. For “Male“, we will replace missing values of manpower with 29.75 and for “Female” with 25.
# 
# **Prediction Model**:  Prediction model is one of the sophisticated method for handling missing data. Here, we create a predictive model to estimate values that will substitute the missing data. In this case, we divide our data set into two sets: One set with no missing values for the variable and another one with missing values. First data set become training data set of the model while second data set with missing values is test data set and variable with missing values is treated as target variable. Next, we create a model to predict target variable based on other attributes of the training data set and populate missing values of test data set.We can use regression, ANOVA, Logistic regression and various modeling technique to perform this. There are 2 drawbacks for this approach:
# 1)The model estimated values are usually more well-behaved than the true values
# 2)If there are no relationships with attributes in the data set and the attribute with missing values, then the model will not be precise for estimating missing values.
# 
# **KNN Imputation**: In this method of imputation, the missing values of an attribute are imputed using the given number of attributes that are most similar to the attribute whose values are missing. The similarity of two attributes is determined using a distance function. It is also known to have certain advantage & disadvantages.
# 
# *Advantages*:
# k-nearest neighbour can predict both qualitative & quantitative attributes Creation of predictive model for each attribute with missing data is not required Attributes with multiple missing values can be easily treated Correlation structure of the data is taken into consideration
# 
# *Disadvantage*:
# KNN algorithm is very time-consuming in analyzing large database. It searches through all the dataset looking for the most similar instances.Choice of k-value is very critical. Higher value of k would include attributes which are significantly different from what we need whereas lower value of k implies missing out of significant attributes.
# 
# After dealing with missing values, the next task is to deal with outliers. Often, we tend to neglect outliers while building models. This is a discouraging practice. Outliers tend to make your data skewed and reduces accuracy. Let’s learn more about outlier treatment.
# 
#  
# 

# In[10]:

df.isnull().sum()   ## Find number of NA in each colunm


# In[11]:

df.isnull().sum().sum()  # Total NA in data


# In[12]:

df['Product_Category_2'].median()                    ## Find median of Product_Category_2 to fill NA


# In[13]:

df['Product_Category_2'].fillna(value=9,inplace=True)            ## Filling NA for Product_Category_2


# In[14]:

df['Product_Category_3'].median()                 ## Find median of Product_Category_2 to fill NA


# In[15]:

df['Product_Category_3'].fillna(value=14,inplace=True)    ## Filling NA for Product_Category_2


# In[16]:

df.isnull().sum().sum()    ## There are no NA in the train data


# ## Step 2
# # Exploratory Data Visualizaation

# In[17]:

## Find Unique value in User ID

df.User_ID.nunique()


# In[18]:

## Number of unique in Product ID

df.Product_ID.nunique()      ## There are 3631 unique product ID outof 550068 observations


# In[19]:

## Number of Categorical variable

(df.dtypes=='object').sum()  ## There are 5 categorical variables


# <a id='Univariate Analysis'></a>
# ## Univariate Analysis ##
# 
# At this stage, we explore variables one by one. Method to perform uni-variate analysis will depend on whether the variable type is categorical or continuous. Let’s look at these methods and statistical measures for categorical and continuous variables individually:
# 
# **Continuous Variables**:- In case of continuous variables, we need to understand the central tendency and spread of the variable. These are measured using various statistical metrics visualization methods like ** Histogram & Boxplot **.
# We will use Matplotlib & Seaborn package.

# In[20]:

## which continuous variables in data

df.dtypes  ## data types of variables


# In[21]:

(df.dtypes=='int64').sum()                 # there are 5 variables which have integer as dtype


# In[22]:

(df.dtypes=='float64').sum()               # there are 2 variables which have float as dtype


# In[23]:

# Check the distribution of dependent variable i.e Purchase
df['Purchase'].hist(bins=20).set_title('Purchase Pattern')


# Looking at Purchase Pattern we can see that number of count is more at 6000 & 7000 and above 15000 have very less count.

# In[24]:

# Check the distribution of dependent variable i.e Product Category 1

df['Product_Category_1'].hist(bins=20).set_title('Product_Category_1 Pattern')


# In[25]:

# Check the distribution of Product_Category_2 

df['Product_Category_2'].hist(bins=20).set_title('Product_Category_2 Pattern')


# In[26]:

# Check the distribution of Product_Category_3 

df['Product_Category_3'].hist(bins=20).set_title('Product_Category_3 Pattern')


# In[27]:

# Check the distribution of Occupation

df['Occupation'].hist(bins=20).set_title('Occupation Pattern')


# **Categorical Variables**:- For categorical variables, we’ll use frequency table to understand distribution of each category. We can also read as percentage of values under each category. It can be be measured using two metrics, Count and Count% against each category. Bar chart can be used as visualization.

# In[28]:

df.dtypes


# In[29]:

# Number of Gender count

dim = (10,10)
sns.countplot(x="Gender", data=df, palette="Greens_d").set_title('Data: Gender-wise count')


# In[30]:

## Number of count of Age

dim = (10,10)
sns.countplot(x='Age',data=df,palette="Blues").set_title('Data: Agewise count')


# In[31]:

## Number of count of City

dim = (10,10)

sns.countplot(x='City_Category',data=df,palette="pastel").set_title('Data:  Citywise count')


# In[32]:

## Number of count of stay

dim = (10,10)
sns.countplot(x='Stay_In_Current_City_Years',data=df,palette="deep").set_title('Data:  Stay-wise count')


# In[33]:

## Number of count of Marital_Status

sns.countplot(x='Marital_Status',data=df,palette="muted").set_title('Data:  Marital_Status-wise count')


# In[34]:

## Number of count of Product_ID 

sns.countplot(x='Product_ID',data=df,palette="pastel").set_title('Data:  Product_ID-wise count')


# <a id='Bi-variate Analysis'></a>
# ## Bi-variate Analysis ##
# 
# Bi-variate Analysis finds out the relationship between two variables. Here, we look for association and disassociation between variables at a pre-defined significance level. We can perform bi-variate analysis for any combination of categorical and continuous variables. The combination can be: Categorical & Categorical, Categorical & Continuous and Continuous & Continuous. Different methods are used to tackle these combinations during analysis process.
# 
# Let’s understand the possible combinations in detail:
# 
# **Continuous & Continuous**: While doing bi-variate analysis between two continuous variables, we should look at **scatter plot**. It is a nifty way to find out the relationship between two variables. The pattern of scatter plot indicates the relationship between variables. The relationship can be linear or non-linear.

# In[35]:

df.dtypes


# **Categorical & Continuous**: While exploring relation between categorical and continuous variables, we can draw box plots for each level of categorical variables. If levels are small in number, it will not show the statistical significance. To look at the statistical significance we can perform Z-test, T-test or ANOVA.
# 
# 1) *Z-Test/ T-Test*:- Either test assess whether mean of two groups are statistically different from each other or not.
# 
# ***If the probability of Z is small then the difference of two averages is more significant**. The T-test is very similar to Z-test but it is used when number of observation for both categories is less than 30.
# 
# 2) *ANOVA*:- It assesses whether the average of more than two groups is statistically different.
# 
# Example: Suppose, we want to test the effect of five different exercises. For this, we recruit 20 men and assign one type of exercise to 4 men (5 groups). Their weights are recorded after a few weeks. We need to find out whether the effect of these exercises on them is significantly different or not. This can be done by comparing the weights of the 5 groups of 4 men each.
# 

# In[36]:

from scipy import stats


# In[37]:

gender_grp = df.groupby(['Gender'])
gender_grp['Purchase'].describe()


# In[38]:

gender_grp.boxplot(column=['Purchase'],return_type='axes')


# **Boxplot shows Female & Male are not significantly different but we will check by Hypothesis**

# In[39]:

# make purchase data for hypotheis of Female & Male to find out either F & M are significantly different

M_purchase = df[df['Gender']=='F']['Purchase']
F_purchase = df[df['Gender']=='M']['Purchase']


# In[40]:

stats.ttest_ind(M_purchase, F_purchase,equal_var=False)  ## equal_var means if True means they have same sd 


# **By doing Hypothesis we can say that they have mean purchase difference i.e they are significant**

# In[41]:

## Age wise hypothesis

age_grp = df.groupby(['Age'])
age_grp.boxplot(column=['Purchase'],return_type='axes',figsize=(10,10))


# In[42]:

age_grp['Purchase'].describe()


# In[43]:

## subset the data as per levels from Age

p_0to17 = df[df['Age']=='0-17']['Purchase']
p_18to25 = df[df['Age']=='18-25']['Purchase']
p_26to35 = df[df['Age']=='26-35']['Purchase']
p_36to45 = df[df['Age']=='36-45']['Purchase']
p_46to50 = df[df['Age']=='46-50']['Purchase']
p_51to55 = df[df['Age']=='51-55']['Purchase']
p_55 = df[df['Age']=='55+']['Purchase']


# In[44]:

## Test the anova

stats.f_oneway(p_0to17,p_18to25,p_26to35,p_36to45,p_46to50,p_51to55,p_55)


# *Age groups have different purchase mean,so we will consider this Age variable for model building.
# 

# In[45]:

grp_marital = df.groupby(['Marital_Status'])
grp_marital['Purchase'].describe()


# In[46]:

grp_marital.boxplot(column=['Purchase'],return_type='axes',figsize=(10,5))


# In[47]:

## subset Marital status

m0 = df[df['Marital_Status']== 0]['Purchase']
m1 = df[df['Marital_Status']== 1]['Purchase']

## Test Anova 
stats.f_oneway(m0,m1)


# In[48]:

city_grp = df.groupby(['City_Category'])
city_grp['Purchase'].describe()


# In[49]:

city_grp.boxplot(column=['Purchase'],return_type='axes',figsize=(10,10))


# In[50]:

# Subset City_Category
city_A = df[df['City_Category']=='A']['Purchase']
city_B = df[df['City_Category']=='B']['Purchase']
city_C = df[df['City_Category']=='C']['Purchase']

# Test ANOVA
stats.f_oneway(city_A,city_B,city_C)


# *City_Category variables groups have signifiacntly different purchase mean, we can take this variable for analysis.*

# In[51]:

# group Stay_In_Current_City_Years
grp_yrs = df.groupby(['Stay_In_Current_City_Years'])
grp_yrs['Purchase'].describe()


# In[52]:

grp_yrs.boxplot(column=['Purchase'],return_type='axes',figsize=(10,10))


# In[53]:

# subset Product ID

p1 = df[df['Product_ID']=='P00000142']['Purchase']
p2 = df[df['Product_ID']=='P00000242']['Purchase']
p3 = df[df['Product_ID']=='P00000342']['Purchase']
p4 = df[df['Product_ID']=='P00000442']['Purchase']
p5 = df[df['Product_ID']=='P0099642']['Purchase']
p6 = df[df['Product_ID']=='P0099742']['Purchase']
p7 = df[df['Product_ID']=='P0099842']['Purchase']
p8 = df[df['Product_ID']=='P0099942']['Purchase']

# test anova
stats.f_oneway(p1,p2,p3,p4,p5,p6,p7,p8)


# *Product ID groups have significant difference in purchase mean*

# In[54]:

# group Product_Category_1
grp_cat1 = df.groupby(['Product_Category_1'])
grp_cat1['Purchase'].describe()


# In[55]:

# group Product_Category_2
grp_cat2 = df.groupby(['Product_Category_2'])
grp_cat2['Purchase'].describe()


# In[56]:

# group Product_Category_3
grp_cat3 = df.groupby(['Product_Category_3'])
grp_cat3['Purchase'].describe()


# In[57]:

# group Occupation
grp_occ = df.groupby(['Occupation'])
grp_occ['Purchase'].describe()


# In[58]:

df.dtypes


# In[59]:

### Pivot Table 

impute_grps = df.pivot_table(values=["Purchase"], index=["Gender","City_Category","Stay_In_Current_City_Years","Occupation"], aggfunc=np.mean)
impute_grps


# **Categorical & Categorical**: To find the relationship between two categorical variables, we can use following methods:
# 
# *Two-way table*: We can start analyzing the relationship by creating a two-way table of count and count%. The rows represents the category of one variable and the columns represent the categories of the other variable. We show count or count% of observations available in each combination of row and column categories.
# 
# *Stacked Column Chart*: This method is more of a visual form of Two-way table.
# 
# *Chi-Square Test*: This test is used to derive the statistical significance of relationship between the variables. Also, it tests whether the evidence in the sample is strong enough to generalize that the relationship for a larger population as well. Chi-square is based on the difference between the expected and observed frequencies in one or more categories in the two-way table. It returns probability for the computed chi-square distribution with the degree of freedom.
# 
# Probability of 0: It indicates that both categorical variable are dependent
# Probability of 1: It shows that both variables are independent.
# Probability less than 0.05: It indicates that the relationship between the variables is significant at 95% confidence.
# 

# In[60]:

## Stacked chart for visualization

var = df.groupby(['Age','Gender']).Gender.count()
var.unstack().plot(kind='bar',stacked=True, grid=False,figsize=(10,5))


# In[61]:

## Chi-sq test

chi_AG = pd.crosstab(df['Gender'],df['Age'])
chi_AG


# In[62]:

stats.chi2_contingency(chi_AG)


#                                      Gender & Age are dependent on each other

# In[63]:

# stack 
var1 = df.groupby(['City_Category','Gender']).Gender.count()
var1.unstack().plot(kind='bar',stacked=True, grid=False,figsize=(10,5),color=['pink','lightgreen'])


# In[64]:

## cross tab

chi_GC = pd.crosstab(df['Gender'],df['City_Category'])
chi_GC


# In[65]:

stats.chi2_contingency(chi_GC)


#                                               Gender & City_Category dependent

# In[66]:

# stack 
var2 = df.groupby(['Marital_Status','Gender']).Gender.count()
var2.unstack().plot(kind='bar',stacked=True, grid=False,figsize=(10,5),color=['grey','lightblue'])


# In[67]:

## cross tab

chi_Gm = pd.crosstab(df['Gender'],df['Marital_Status'])
chi_Gm


# In[68]:

stats.chi2_contingency(chi_Gm)


# <a id='Outlier'></a>
# ## Outlier ##
# Outlier is a commonly used terminology by analysts and data scientists as it needs close attention else it can result in wildly wrong estimations. Simply speaking, Outlier is an observation that appears far away and diverges from an overall pattern in a sample.
# 
# **What are the types of Outliers?**
# 
# Outlier can be of two types: Univariate and Multivariate. Above, we have discussed the example of univariate outlier. These outliers can be found when we look at distribution of a single variable. Multi-variate outliers are outliers in an n-dimensional space. In order to find them, you have to look at distributions in multi-dimensions.

# In[69]:

plt.figure(figsize=(15, 4))
sns.boxplot(df['Purchase'],color='c')


# ## Variable Transformation##
# What are the common methods of Variable Transformation?
# 
# There are various methods used to transform variables. As discussed, some of them include square root, cube root, logarithmic, binning, reciprocal and many others. Let’s look at these methods in detail by highlighting the pros and cons of these transformation methods.
# 
# **Logarithm**: Log of a variable is a common transformation method used to change the shape of distribution of the variable on a distribution plot. It is generally used for reducing right skewness of variables. Though, It can’t be applied to zero or negative values as well.
# 
# **Square / Cube root**: The square and cube root of a variable has a sound effect on variable distribution. However, it is not as significant as logarithmic transformation. Cube root has its own advantage. It can be applied to negative values including zero. Square root can be applied to positive values including zero.
# 
# **Binning**: It is used to categorize variables. It is performed on original values, percentile or frequency. Decision of categorization technique is based on business understanding. For example, we can categorize income in three categories, namely: High, Average and Low. We can also perform co-variate binning which depends on the value of more than one variables.
#  

# # Data Preparation #
# 
# <a id='Label encoding'></a>
# ### Label Encoding ###
# 

# In[70]:

#Label Encoding

df = df.ix[:,[0,1,2,3,4,5,6,7,8,9,10,11]] #for subseting by index

#Importing LabelEncoder and initializing it
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for col in df.columns.values:
    # Encoding only categorical variables
    if df[col].dtypes=='object':
        # Using whole data to form an exhaustive list of levels
        data=df[col]
        le.fit(data.values)
        df[col]=le.transform(df[col])


# <a id='Dummy Variables'></a>
# ### Dummy Variables ###

# In[71]:

## Create dummy variables 

df = pd.get_dummies(df, columns=['City_Category','Age','Stay_In_Current_City_Years'])
print (df.shape)


# In[72]:

df.to_csv('blackfriday_modified.csv')


# In[73]:

df.head()


# <a id='Data Spliting'></a>
# ## Data Spliting ##

# In[74]:

# Moving the Purchase column to the last
Purchase = df["Purchase"]
df.drop(labels = ['Purchase'], axis = 1, inplace= True)
df.insert(23,"Purchase", Purchase)


# In[75]:

df.head()


# In[76]:

### Convert User ID & Product_ID in 
df['User_ID'] = df['User_ID'].astype('category')
df['Product_ID'] = df['Product_ID'].astype('category')


# In[91]:

import numpy as np
array = df.values
X = np.array(array[:,0:23])
y = np.array(array[:,23])
# '''X = df[:,0:23]
#y = df[:,23]
#X  = np.array[:,0:23]
#y = np.array[:,23]'''


# In[92]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


# In[93]:

print (X_train.shape)
print (X_test.shape)


# In[94]:

print(y_train.shape)
print(y_test.shape)


# # Model Building
# 

# # Linear Regressor

# In[103]:

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[104]:

# Predict on test
from sklearn.metrics import mean_squared_error
from math import sqrt

predictions_1 = lm.predict(X_test)
lm_rmse = sqrt(mean_squared_error(y_test, predictions_1))
print(lm_rmse)


# # Decision Tree Regressor

# In[107]:

from sklearn import tree

## Model fitting
dt = tree.DecisionTreeRegressor(max_depth = 7)
dt.fit(X_train,y_train)

## Predict
predictions_2 = dt.predict(X_test)

dt_rmse = sqrt(mean_squared_error(y_test, predictions_3))


# In[108]:

print(dt_rmse)


# # Random Forest Regressor

# In[96]:

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=7,n_estimators=100,verbose = 1)

rf.fit(X_train,y_train)
predictions_2  = rf.predict(X_test)


# In[97]:

from sklearn.metrics import mean_squared_error
from math import sqrt

rf_rmse = sqrt(mean_squared_error(y_test, predictions_2))


# In[98]:

print(rf_rmse)


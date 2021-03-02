import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
import seaborn as sns

df=pd.read_csv(r'C:\Users\manis\Downloads\life-expectancy-who\Life Expectancy Data.csv')

df=df.rename(columns={"Life expectancy ":"Life_Expectancy", "Measles ":"Measles", " BMI ":"BMI", "under-five deaths ":"under-five deaths",
                      "Diphtheria ":"Diphtheria", ' HIV/AIDS':'HIV/AIDS', ' thinness  1-19 years':'thinness_1to19_years', ' thinness 5-9 years':'thinness_5to9_years'
                    })

#checking null values
print("Null Percentage of Columns")
print((df.isnull().sum()*100)/df.isnull().count())

#filling null values using interpolate
country_list = df.Country.unique()
fill_list = ['Life_Expectancy','Adult Mortality','Alcohol','Hepatitis B','BMI',
             'Polio','Total expenditure','Diphtheria','GDP','Population','thinness_1to19_years',
             'thinness_5to9_years','Income composition of resources','Schooling']
for country in country_list:
    df.loc[df['Country'] == country,fill_list] = df.loc[df['Country'] == country,fill_list].interpolate()
df=df.dropna()
print("After using Interpolate")
print((df.isnull().sum()*100)/df.isnull().count())

#detecting outliers
col_dict = {'Life_Expectancy':1,'Adult Mortality':2,'infant deaths':3,'Alcohol':4,'percentage expenditure':5,
            'Hepatitis B':6,'Measles':7,'BMI':8,'under-five deaths':9,'Polio':10,
            'Total expenditure':11,'Diphtheria':12,'HIV/AIDS':13,'GDP':14,'Population':15,'thinness_1to19_years':16,
            'thinness_5to9_years':17,'Income composition of resources':18,'Schooling':19}

# Detect outliers in each variable using box plots.
plt.figure(figsize=(20,30))

for variable,i in col_dict.items():
                     plt.subplot(5,4,i)
                     plt.boxplot(df[variable],whis=1.5)
                     plt.title(variable)

plt.show()

#removing outliers
plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Life_Expectancy = df['Life_Expectancy']
plt.boxplot(original_Life_Expectancy)
plt.title("original_Life_Expectancy")

plt.subplot(1,2,2)
winsorized_Life_Expectancy = winsorize(df['Life_Expectancy'],(0.01,0))
plt.boxplot(winsorized_Life_Expectancy)
plt.title("winsorized_Life_Expectancy")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_Adult_Mortality = df['Adult Mortality']
plt.boxplot(original_Adult_Mortality)
plt.title("original_Adult_Mortality")

plt.subplot(1,2,2)
winsorized_Adult_Mortality = winsorize(df['Adult Mortality'],(0,0.03))
plt.boxplot(winsorized_Adult_Mortality)
plt.title("winsorized_Adult_Mortality")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_Infant_Deaths = df['infant deaths']
plt.boxplot(original_Infant_Deaths)
plt.title("original_Infant_Deaths")

plt.subplot(1,2,2)
winsorized_Infant_Deaths = winsorize(df['infant deaths'],(0,0.10))
plt.boxplot(winsorized_Infant_Deaths)
plt.title("winsorized_Infant_Deaths")
#plt.show()

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Alcohol = df['Alcohol']
plt.boxplot(original_Alcohol)
plt.title("original_Alcohol")

plt.subplot(1,2,2)
winsorized_Alcohol = winsorize(df['Alcohol'],(0,0.01))
plt.boxplot(winsorized_Alcohol)
plt.title("winsorized_Alcohol")
#plt.show()

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Percentage_Exp = df['percentage expenditure']
plt.boxplot(original_Percentage_Exp)
plt.title("original_Percentage_Exp")

plt.subplot(1,2,2)
winsorized_Percentage_Exp = winsorize(df['percentage expenditure'],(0,0.12))
plt.boxplot(winsorized_Percentage_Exp)
plt.title("winsorized_Percentage_Exp")
#plt.show()

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_HepatitisB = df['Hepatitis B']
plt.boxplot(original_HepatitisB)
plt.title("original_HepatitisB")

plt.subplot(1,2,2)
winsorized_HepatitisB = winsorize(df['Hepatitis B'],(0.11,0))
plt.boxplot(winsorized_HepatitisB)
plt.title("winsorized_HepatitisB")
#plt.show()

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Measles = df['Measles']
plt.boxplot(original_Measles)
plt.title("original_Measles")

plt.subplot(1,2,2)
winsorized_Measles = winsorize(df['Measles'],(0,0.19))
plt.boxplot(winsorized_Measles)
plt.title("winsorized_Measles")
#plt.show()
df = df.drop('Measles',axis=1)

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_Under_Five_Deaths = df['under-five deaths']
plt.boxplot(original_Under_Five_Deaths)
plt.title("original_Under_Five_Deaths")

plt.subplot(1,2,2)
winsorized_Under_Five_Deaths = winsorize(df['under-five deaths'],(0,0.12))
plt.boxplot(winsorized_Under_Five_Deaths)
plt.title("winsorized_Under_Five_Deaths")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_Polio = df['Polio']
plt.boxplot(original_Polio)
plt.title("original_Polio")

plt.subplot(1,2,2)
winsorized_Polio = winsorize(df['Polio'],(0.09,0))
plt.boxplot(winsorized_Polio)
plt.title("winsorized_Polio")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_Tot_Exp = df['Total expenditure']
plt.boxplot(original_Tot_Exp)
plt.title("original_Tot_Exp")

plt.subplot(1,2,2)
winsorized_Tot_Exp = winsorize(df['Total expenditure'],(0,0.01))
plt.boxplot(winsorized_Tot_Exp)
plt.title("winsorized_Tot_Exp")
#plt.show()

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Diphtheria = df['Diphtheria']
plt.boxplot(original_Diphtheria)
plt.title("original_Diphtheria")
plt.subplot(1,2,2)
winsorized_Diphtheria = winsorize(df['Diphtheria'],(0.10,0))
plt.boxplot(winsorized_Diphtheria)
plt.title("winsorized_Diphtheria")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_HIV = df['HIV/AIDS']
plt.boxplot(original_HIV)
plt.title("original_HIV")

plt.subplot(1,2,2)
winsorized_HIV = winsorize(df['HIV/AIDS'],(0,0.16))
plt.boxplot(winsorized_HIV)
plt.title("winsorized_HIV")
#plt.show()

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_GDP = df['GDP']
plt.boxplot(original_GDP)
plt.title("original_GDP")

plt.subplot(1,2,2)
winsorized_GDP = winsorize(df['GDP'],(0,0.13))
plt.boxplot(winsorized_GDP)
plt.title("winsorized_GDP")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_Population = df['Population']
plt.boxplot(original_Population)
plt.title("original_Population")

plt.subplot(1,2,2)
winsorized_Population = winsorize(df['Population'],(0,0.14))
plt.boxplot(winsorized_Population)
plt.title("winsorized_Population")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_thinness_1to19_years = df['thinness_1to19_years']
plt.boxplot(original_thinness_1to19_years)
plt.title("original_thinness_1to19_years")

plt.subplot(1,2,2)
winsorized_thinness_1to19_years = winsorize(df['thinness_1to19_years'],(0,0.04))
plt.boxplot(winsorized_thinness_1to19_years)
plt.title("winsorized_thinness_1to19_years")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_thinness_5to9_years = df['thinness_5to9_years']
plt.boxplot(original_thinness_5to9_years)
plt.title("original_thinness_5to9_years")

plt.subplot(1,2,2)
winsorized_thinness_5to9_years = winsorize(df['thinness_5to9_years'],(0,0.04))
plt.boxplot(winsorized_thinness_5to9_years)
plt.title("winsorized_thinness_5to9_years")
#plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
original_Income_Comp_Of_Resources = df['Income composition of resources']
plt.boxplot(original_Income_Comp_Of_Resources)
plt.title("original_Income_Comp_Of_Resources")

plt.subplot(1,2,2)
winsorized_Income_Comp_Of_Resources = winsorize(df['Income composition of resources'],(0.05,0))
plt.boxplot(winsorized_Income_Comp_Of_Resources)
plt.title("winsorized_Income_Comp_Of_Resources")
#plt.show()

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
original_Schooling = df['Schooling']
plt.boxplot(original_Schooling)
plt.title("original_Schooling")
plt.subplot(1,2,2)
winsorized_Schooling = winsorize(df['Schooling'],(0.02,0.01))
plt.boxplot(winsorized_Schooling)
plt.title("winsorized_Schooling")
#plt.show()
#adding the winsorized columns
df['winsorized_Life_Expectancy'] = winsorized_Life_Expectancy
df['winsorized_Adult_Mortality'] = winsorized_Adult_Mortality
df['winsorized_Infant_Deaths'] = winsorized_Infant_Deaths
df['winsorized_Alcohol'] = winsorized_Alcohol
df['winsorized_Percentage_Exp'] = winsorized_Percentage_Exp
df['winsorized_HepatitisB'] = winsorized_HepatitisB
df['winsorized_Under_Five_Deaths'] = winsorized_Under_Five_Deaths
df['winsorized_Polio'] = winsorized_Polio
df['winsorized_Tot_Exp'] = winsorized_Tot_Exp
df['winsorized_Diphtheria'] = winsorized_Diphtheria
df['winsorized_HIV'] = winsorized_HIV
df['winsorized_GDP'] = winsorized_GDP
df['winsorized_Population'] = winsorized_Population
df['winsorized_thinness_1to19_years'] = winsorized_thinness_1to19_years
df['winsorized_thinness_5to9_years'] = winsorized_thinness_5to9_years
df['winsorized_Income_Comp_Of_Resources'] = winsorized_Income_Comp_Of_Resources
df['winsorized_Schooling'] = winsorized_Schooling

print(df.describe())
print(df.columns)
all_col = ['Life_Expectancy','winsorized_Life_Expectancy','Adult Mortality','winsorized_Adult_Mortality','infant deaths',
         'winsorized_Infant_Deaths','Alcohol','winsorized_Alcohol','percentage expenditure','winsorized_Percentage_Exp','Hepatitis B',
         'winsorized_HepatitisB','under-five deaths','winsorized_Under_Five_Deaths','Polio','winsorized_Polio','Total expenditure',
         'winsorized_Tot_Exp','Diphtheria','winsorized_Diphtheria','HIV/AIDS','winsorized_HIV','GDP','winsorized_GDP',
         'Population','winsorized_Population','thinness_1to19_years','winsorized_thinness_1to19_years','thinness_5to9_years',
         'winsorized_thinness_5to9_years','Income composition of resources','winsorized_Income_Comp_Of_Resources',
         'Schooling','winsorized_Schooling']

#histograms
plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Life_Expectancy'])
plt.title("original_Life_expectancy")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Life_Expectancy'])
plt.title("winsorized_life_expectancy")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Adult Mortality'])
plt.title("original Adult Mortality")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Adult_Mortality'])
plt.title("winsorized adult mortality")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['infant deaths'])
plt.title("original Infant Deaths")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Infant_Deaths'])
plt.title("winsorizeed infant deaths")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Alcohol'])
plt.title("original Alcohol")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Alcohol'])
plt.title("winsorized Alcohol")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['percentage expenditure'])
plt.title("original Percentage expenditure")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Percentage_Exp'])
plt.title("winsorized Percentage Expenditure")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Hepatitis B'])
plt.title("original Hepatitis B")
plt.subplot(1,2,2)
plt.hist(df['winsorized_HepatitisB'])
plt.title("winsorized HepatitisB")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['under-five deaths'])
plt.title("Original Under five deaths")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Under_Five_Deaths'])
plt.title("winsorized under-five deaths")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Polio'])
plt.title("original Polio")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Polio'])
plt.title("winsorized Polio")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Total expenditure'])
plt.title("original Total expenditure")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Tot_Exp'])
plt.title("winsorized Total expenditure")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Diphtheria'])
plt.title("original Diphtheria")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Diphtheria'])
plt.title("winsorized Diphtheria")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['HIV/AIDS'])
plt.title("original HIV/AIDS")
plt.subplot(1,2,2)
plt.hist(df['winsorized_HIV'])
plt.title("winsorized HIV/AIDS")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['GDP'])
plt.title("original GDP")
plt.subplot(1,2,2)
plt.hist(df['winsorized_GDP'])
plt.title("winsorized GDP")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Population'])
plt.title("original Population")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Population'])
plt.title("winsorized Population")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['thinness_1to19_years'])
plt.title("original thinness 1 to 19 years")
plt.subplot(1,2,2)
plt.hist(df['winsorized_thinness_1to19_years'])
plt.title("winsorized thinness 1 to 19 years")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['thinness_5to9_years'])
plt.title("original thinness 5 to 9 years")
plt.subplot(1,2,2)
plt.hist(df['winsorized_thinness_5to9_years'])
plt.title("winsorized thinness 5 to 9 years")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Income composition of resources'])
plt.title("original Income composition of resources")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Income_Comp_Of_Resources'])
plt.title("winsorized Income composition of resources")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.hist(df['Schooling'])
plt.title("original Schooling")
plt.subplot(1,2,2)
plt.hist(df['winsorized_Schooling'])
plt.title("winsorized Schooling")
plt.show()

#barchart
plt.figure(figsize=(6,6))
plt.bar(df.groupby('Status')['Status'].count().index,df.groupby('Status')['winsorized_Life_Expectancy'].mean())
plt.xlabel("Status",fontsize=12)
plt.ylabel("Avg Life_Expectancy",fontsize=12)
plt.title("Life_Expectancy w.r.t Status")
plt.show()

#scatter plot
plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Adult_Mortality"])
plt.title("LifeExpectancy vs AdultMortality")
plt.subplot(1,2,2)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Infant_Deaths"])
plt.title("LifeExpectancy vs Infant_Deaths")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Alcohol"])
plt.title("LifeExpectancy vs Alcohol")
plt.subplot(1,2,2)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Percentage_Exp"])
plt.title("LifeExpectancy vs Percentage_Exp")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_HepatitisB"])
plt.title("LifeExpectancy vs HepatitisB")
plt.subplot(1,2,2)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Under_Five_Deaths"])
plt.title("LifeExpectancy vs Under_Five_Deaths")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Polio"])
plt.title("LifeExpectancy vs Polio")
plt.subplot(1,2,2)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Tot_Exp"])
plt.title("LifeExpectancy vs Tot_Exp")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Diphtheria"])
plt.title("LifeExpectancy vs Diphtheria")
plt.subplot(1,2,2)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_HIV"])
plt.title("LifeExpectancy vs HIV")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_GDP"])
plt.title("LifeExpectancy vs GDP")
plt.subplot(1,2,2)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Population"])
plt.title("LifeExpectancy vs Population")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_thinness_1to19_years"])
plt.title("LifeExpectancy vs thinness_1to19_years")
plt.subplot(1,2,2)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_thinness_5to9_years"])
plt.title("LifeExpectancy vs thinness_5to9_years")
plt.show()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Income_Comp_Of_Resources"])
plt.title("LifeExpectancy vs Income_Comp_Of_Resources")
plt.subplot(1,2,2)
plt.scatter(df["winsorized_Life_Expectancy"], df["winsorized_Schooling"])
plt.title("LifeExpectancy vs Schooling")
plt.show()
#heatmap
df_win = df.iloc[:,21:]
df_win['Country'] = df['Country']
df_win['Year'] = df['Year']
df_win['Status'] = df['Status']
df_win_num = df_win.iloc[:,:-3]
cormat = df_win_num.corr()

plt.figure(figsize=(15,15))
sns.heatmap(cormat, square=True, annot=True, linewidths=.5)
plt.title("Correlation matrix among winsorized variables")
plt.show()


x = pd.DataFrame(df).to_csv('5502finalproject.csv', header=True, index=None)


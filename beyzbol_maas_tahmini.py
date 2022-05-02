#########################################
# Beyzbol Maaş Tahmini
#########################################
# Görev :
# • Veri ön işleme,
# • Özellik mühendisliği
# işlemleri gerçekleştirerek maaş tahmin modeli geliştiriniz.
#########################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("Odev_calismalari/7.hafta/hitters.csv")
df.head()


##################################
# KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# ADIM 1: GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


##################################
# ADIM 2: NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


##################################
# ADIM 3: KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

for col in cat_cols:
    n_cat = df[col].value_counts()
    ratio = df[col].value_counts() / len(df[col]) * 100
    print(pd.concat([n_cat, ratio], axis=1, keys=[(col+" count"), "ratio"]), end="\n\n")


##################################
# ADIM 4: KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


##################################
# ADIM 5: AYKIRI GÖZLEM ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col,": ", check_outlier(df, col))


##################################
# ADIM 6: EKSİK GÖZLEM ANALİZİ
##################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)


##################################
# ADIM 7: KORELASYON
##################################

corr = df[num_cols].corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


##################################
# FEATURE ENGINEERING
##################################

##################################
# ADIM 1: EKSİK DEĞERLER İÇİN İŞLEMLER
##################################

df.iloc[df[(df.isnull().sum(axis=1) >= 1)].index]

df["Salary"].fillna(df.groupby("Division")["Salary"].transform("mean"), inplace=True)

#.loc ile doldurma
#df.loc[(df["Division"] == "E") & (df["Salary"].isnull()>0), "Salary"] = df.loc[df["Division"] == "E", "Salary"].mean()
#df.loc[(df["Division"] == "W") & (df["Salary"].isnull()>0), "Salary"] = df.loc[df["Division"] == "W", "Salary"].mean()

#Başarısız denemeler
#df["Salary"] = df.loc[(df["Division"] == "E") & (df["Salary"].isnull()>0)].fillna(df.groupby("Division")["Salary"].mean()[0])
#df["Salary"] = df.loc[df["Division"] == "W", "Salary"].fillna(df.groupby("Division")["Salary"].mean()[0])


df.head(20)

##################################
# ADIM 1: AYKIRI DEĞERLER İÇİN İŞLEMLER
##################################

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)


##################################
# ADIM 2: YENİ DEĞİŞKENLER OLUŞTURMA
##################################

df.head()
df.loc[(df["League"] == df["NewLeague"]), "NEW_LeagueChange"] = "No"
df["NEW_LeagueChange"].fillna("Yes", inplace=True)

df.loc[(df["Years"]>=0) & (df["Years"]<=3),"NEW_YearCat"] = "0-3 Year"
df.loc[(df["Years"]>=4) & (df["Years"]<=7),"NEW_YearCat"] = "4-7 Year"
df.loc[(df["Years"]>=8) & (df["Years"]<=11),"NEW_YearCat"] = "8-11 Year"
df.loc[(df["Years"]>=12),"NEW_YearCat"] = "12+ Year"


df["NEW_AtBatOther"] = df["CAtBat"] - df["AtBat"]
df["NEW_HitsOther"] = df["CHits"] - df["Hits"]
df["NEW_HmRunOther"] = df["CHmRun"] - df["HmRun"]
df["NEW_RunsOther"] = df["CRuns"] - df["Runs"]
df["NEW_RBIOther"] = df["CRBI"] - df["RBI"]
df["NEW_WalksOther"] = df["CWalks"] - df["Walks"]



##################################
# ADIM 3: ENCODING İŞLEMLERİ
##################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()

#One Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols, drop_first=True)

df.head()



##################################
# ADIM 4: NUMERİK DEĞİŞKENLER İÇİN STANDARTLAŞTIRMA İŞLEMLERİ
##################################

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


####################
# MODELLEME
####################

X = df.drop('Salary', axis=1)

y = df[["Salary"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

reg_model = LinearRegression().fit(X_train, y_train)

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Test RKARE
reg_model.score(X_test, y_test)

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))


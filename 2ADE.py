#pertemuan Pertama

# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np

#Matriks A
print("\nMatriks A")
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
Matriks1=np.array([1,2,3,4])
print(Matriks1)

#Matriks B
print("\nMatriks B")
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
Matriks2=np.array([[1,2,3,4],[1,2,3,4]])
print(Matriks2)

#matriks satuan
print("\nMatriks Satuan")
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
Matriks_ones=np.ones((4,5))
print(Matriks_ones)

#matriks nol
print("\nMatriks nol")
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
Matriks_nol=np.zeros((4,5))
print(Matriks_nol)

#matriks random
print("\nMatriks Random")
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
Matriks_random=np.random.random((4,5))
print(Matriks_random)

#array penuh
print("\nArray Penuh")
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
Array_penuh=np.full((1,5),5)
print(Array_penuh)

#array kosong
print("\nArray penuh")
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
Array_kosong=np.empty((2,5))
print(Array_kosong)


#Pertemuan Ke 2

# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np

# Membuat DataFrame (tabel data) dengan Pandas
dFrame = pd.DataFrame({'f': np.linspace(1, 10, 10)})
# Membuat DataFrame (tabel data) dengan Pandas
dFrame = pd.concat([dFrame, pd.DataFrame(np.random.randn(10, 5), columns=list('EDCBA'))], axis=1)
dFrame
# Definisikan nilai dengan warna
def Warna(value):
    if value < 0:
        color = 'red'
    elif value > 0:
        color = 'black'
    else:
        color = 'green'
    return 'color: %s' % color
s = dFrame.style.applymap(Warna, subset=['A', 'B', 'C', 'D', 'E'])
s
# Highlight Bilangan Maximum dengan warna orange, minimum dengan warna hijau
def highlight(x):
    Maksimum = x == x.max()
    return ['background-color: orange' if v else '' for v in Maksimum]

def highlight2(x):
    Minimum = x == x.min()
    return ['background-color: green' if v else '' for v in Minimum]
styled_df = dFrame.style.apply(highlight, axis=0).apply(highlight2, axis=0)
#styled_df = styled_df.applymap(lambda x: 'color: red' if pd.isnull(x) else '') # Highlight NaNs with red color

styled_df
# Import Seaborn untuk visualisasi statistik berbasis Matplotlib
import seaborn as sns
# Visualisasi menggunakan Seaborn
colorMap=sns.light_palette("blue", as_cmap=True)
Gradasi=dFrame.style.background_gradient(cmap=colorMap)
Gradasi
***Distribusi***
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
from IPython.display import Math, Latex
from IPython.core.display import Image
# Import Seaborn untuk visualisasi statistik berbasis Matplotlib
import seaborn as sns
# Visualisasi menggunakan Seaborn
sns.set(color_codes=True)
# Visualisasi menggunakan Seaborn
sns.set(rc={'figure.figsize':(10,6)})
from scipy.stats import uniform
number = 10000
start = 20
width = 25
uniform_data = uniform.rvs(size=number, loc=start, scale=width)
# Visualisasi menggunakan Seaborn
axis = sns.distplot(uniform_data, bins=100, kde=True, color='skyblue', hist_kws={'linewidth': 15})
axis.set(xlabel='Distribusi Uniform', ylabel='Frekuensi')

# DISTRIBUSI NORMAL
from scipy.stats import norm
normal_data = norm.rvs(size=90000, loc=20, scale=30)
# Visualisasi menggunakan Seaborn
axis = sns.distplot(normal_data, bins=100, kde=True, color='skyblue', hist_kws={'linewidth': 15, 'alpha': 0.568})
axis.set(xlabel='Distribusi Normal', ylabel='Frekuensi')
# DISTRIBUSI EKSPONENSIAL
from scipy.stats import expon

expon_data = expon.rvs(scale=1, loc=0, size=1000)
# Visualisasi menggunakan Seaborn
axis = sns.distplot(expon_data, kde=True, bins=100, color='skyblue', hist_kws={'linewidth': 15})
axis.set(xlabel='Distribusi Eksponensial', ylabel='Frekuensi')
# DISTRIBUSI BINOMIAL
from scipy.stats import binom

binomial_data = binom.rvs(n=10, p=0.8, size=10000)
# Visualisasi menggunakan Seaborn
axis = sns.distplot(binomial_data, kde=False, color='red', hist_kws={'linewidth': 15})
axis.set(xlabel='Distribusi Binomial', ylabel='Frekuensi')

#pertemuan 3
# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# Membaca file CSV dari URL atau file lokal
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",delimiter=";")
df.head()
# Menampilkan ringkasan statistik data
df.describe()
# Menghitung nilai rata-rata (mean)
df.mean()
# Menghitung nilai tengah (median)
df.median()
# Menghitung nilai modus (frekuensi terbanyak)
df.mode()
Kualitas=df['quality']
# Menghitung nilai rata-rata (mean)
Rata2=Kualitas.mean()
# Menghitung nilai tengah (median)
Median=Kualitas.median()
# Menghitung nilai modus (frekuensi terbanyak)
Modus=Kualitas.mode()
print("Nilai Rata-rata: ",Rata2)
print("Nilai Median: ",Median)
print("Nilai Modus: ",Modus)
#Standar defiasi
# Menghitung standar deviasi (penyebaran data)
Standar_Deviasi=df.std()
Standar_Deviasi
# Menghitung standar deviasi (penyebaran data)
std_quality=df['quality'].std()
print("Standar Deviasi Dari Quality=",std_quality)
#varian
# Menghitung nilai variansi
Varian1=df.var()
Varian1
# Menghitung nilai variansi
Varian2=df['quality'].var()
print("Nialai Variansi Dari Quality=",Varian2)
# Menghitung skewness (kemiringan distribusi data)
df.skew()
# Menghitung kurtosis (keruncingan distribusi data)
Kurtosis=df["quality"].kurt()
print("Nilai Kurtosis Dari Quality=",Kurtosis)
#persentil dari quality
quality=df['quality']
# Menghitung persentil dari data
prs=np.percentile(quality,50)
print("Q2=",prs)
# Menghitung persentil dari data
prs2=np.percentile(quality,25)
print("Q1=",prs2)
# Menghitung persentil dari data
prs3=np.percentile(quality,75)
print("Q3=",prs3)
IQR=prs3-prs2
print("IQR=",IQR)
#boxplot
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
Nilai_Fisika=[34,35,35,35,35,35,36,36,37,37,37,37,37,38,38,38,39,39,40,40,40,40,40,41,42
,42,42,42,42,42,42,42,43,43,43,43,44,44,44,44,44,44,45,45,45,45,45,46,46,46
,46,46,46,47,47,47,47,47,47,48,48,48,48,48,49,49,49,49,49,49,49,49,52,52,52
,53,53,53,53,53,53,53,53,54,54,54,54,54,54,54,55,55,55,55,55,56,56,56,56,56
,56,57,57,57,58,58,59,59,59,59,59,59,59,60,60,60,60,60,60,60,61,61,61,61,61
,62,62,63,63,63,63,63,64,64,64,64,64,64,64,65,65,65,66,66,67,67,68,68,68,68
,68,68,68,69,70,71,71,71,72,72,72,72,73,73,74,75,76,76,76,76,77,77,78,79,79
,80,80,81,84,84,85,85,87,87,88]

Nilai_Bindo=[49,49,50,51,51,52,52,52,52,53,54,54,55,55,55,55,56,56,56,56,56,57,57,57,58
,58,58,59,59,59,60,60,60,60,60,60,60,61,61,61,62,62,62,62,63,63,67,67,68,68
,68,68,68,68,69,69,69,69,69,69,70,71,71,71,71,72,72,72,72,73,73,73,73,74,74
,74,74,74,75,75,75,76,76,76,77,77,78,78,78,79,79,79,80,80,82,83,85,88]

Nilai_Komputer=[56,57,58,58,58,60,60,61,61,61,61,61,61,62,62,62,62,63,63,63,63,63,64,64,64
,64,65,65,66,66,67,67,67,67,67,67,67,68,68,68,69,69,70,70,70,71,71,71,73,73
,74,75,75,76,76,77,77,77,78,78,81,82,84,89,90]

# Membuat boxplot untuk distribusi data
plt.boxplot(Nilai_Fisika,showmeans=True,whis=99)

# Membuat boxplot untuk distribusi data
box=plt.boxplot(Nilai_Bindo,showmeans=True,whis=99)
plt.setp(box['boxes'][0],color='blue')
plt.setp(box['whiskers'][0],color='red')
plt.setp(box['whiskers'][1],color='yellow')
plt.setp(box['caps'][0],color='green')
plt.setp(box['caps'][1],color='black')
Data=[Nilai_Fisika,Nilai_Bindo,Nilai_Komputer]
# Membuat boxplot untuk distribusi data
plt.boxplot(Data,showmeans=True,whis=100)
#hitunglah nilai kolom 'pH'
#1. mean
#2. median
#3. modus
#4. standart defiasi
#5. Varian
#6. Kuartil (Q1, Q2, Q3, dan IQR)
#7. koefisin Skewness
#8. koefisien Kurtosis
#9. Boxplot
#1 mean
# Menghitung nilai rata-rata (mean)
ari_mean=df['pH'].mean()
print("Nilai Rata-rata Dari pH=",ari_mean)
#2 Median
# Menghitung nilai tengah (median)
ari_median=df['pH'].median()
print("Nilai Median Dari pH=",ari_median)
#3 modus
# Menghitung nilai modus (frekuensi terbanyak)
ari_modus=df['pH'].mode()
print("Nilai Modus Dari pH=",ari_modus)
#4 standart defiasi
# Menghitung standar deviasi (penyebaran data)
ari_ph=df['pH'].std()
print("Standar Deviasi Dari pH=",ari_ph)
#5 Varian
# Menghitung nilai variansi
ari_varian=df['pH'].var()
print("Nialai Variansi Dari pH=",ari_varian)
#6. Kuartil (Q1, Q2, Q3, dan IQR)
Kurt=df['pH']
# Menghitung persentil dari data
Kuartil=np.percentile(Kurt,50)
print("Nilai Kuartil Dari pH=",Kuartil)
# Menghitung persentil dari data
Kuartil1=np.percentile(Kurt,25)
print("Nilai Kuartil 1 Dari pH=",Kuartil1)
# Menghitung persentil dari data
Kuartil2=np.percentile(Kurt,75)
print("Nilai Kuartil 2 Dari pH=",Kuartil2)
IQR=Kuartil2-Kuartil1
print("Nilai IQR Dari pH=",IQR)
#7. koefisin Skewness
# Menghitung skewness (kemiringan distribusi data)
SK=df['pH'].skew()
print("Nilai Skewness Dari pH=",SK)


#8. koefisien Kurtosis
# Menghitung kurtosis (keruncingan distribusi data)
Kurtosis=df['pH'].kurt()
print("Nilai Kurtosis Dari pH=",Kurtosis)
#9. Boxplot
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
# Membuat boxplot untuk distribusi data
box=plt.boxplot(df['pH'],showmeans=True,whis=99)

#pertemuan ke 4
#LINE_CHART
!pip install faker
!pip install radar

import datetime
import math
# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd
import random
import radar
from faker import Faker
fake = Faker()

def generateData(n):
    listdata = []
    start = datetime.datetime(2019, 8, 1)
    end = datetime.datetime(2019, 8, 30)
    delta = end - start
    for _ in range(n):
        date = radar.random_datetime(start='2019-08-1', stop='2019-08-30').strftime("%Y-%m-%d")
        price = round(random.uniform(900, 1000), 4)
        listdata.append([date, price])
# Membuat DataFrame (tabel data) dengan Pandas
    df = pd.DataFrame(listdata, columns = ['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
# Menghitung nilai rata-rata (mean)
    df = df.groupby(by='Date').mean()
    return df
df = generateData(50)
df.head(10)
df.to_csv(r'stock.csv')
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
# Membuat grafik garis dengan Matplotlib
plt.plot(df)
plt.show()
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (14, 10)
# Membuat grafik garis dengan Matplotlib
plt.plot(df)

#BAR_CHART_PART1
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
import random
import calendar
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
months = list(range(1, 13))

sold_quantity = [round(random.uniform(100, 200)) for x in range(1,13)]
figure, axis = plt.subplots()
plt.xticks(months, calendar.month_name[1:13], rotation=20)
plot = axis.bar(months, sold_quantity)
for rectangle in plot:
    height = rectangle.get_height()
    axis.text(rectangle.get_x() + rectangle.get_width() /2., 1.002 *height, '%d' % int(height), ha='center')
plt.show()

#BAR_CHART_PART2
months = list(range(1, 13))
sold_quantity = [round(random.uniform(100, 200)) for x in range(1, 13)]
figure, axis = plt.subplots()
plt.yticks(months, calendar.month_name[1:13], rotation=20)
plot = axis.barh(months, sold_quantity)
for rectangle in plot:
    width = rectangle.get_width()
    axis.text(width + 2.5, rectangle.get_y() + 0.38, '%d' % int(width),
              ha='center', va = 'bottom')
plt.show()

age = list(range(0, 65))
sleep = []

classBless = ['newborns(0-3)', 'infants(4-11)', 'toddlers(12-24)', 'preschoolers(36-60)', 'school-age(6-13)', 'teenagers(14-17)', 'young adults(18-25)', 'adults(26-64)']
headers_cols = ['age','min_recommended', 'max_recommended', 'may_be_appropriate_min', 'may_be_appropriate_max', 'min_not_recommended', 'max_not_recommended']

# Newborn (0-3)
for i in range(0, 4):
    min_recommended = 14
    max_recommended = 17
    may_be_appropriate_min = 11
    may_be_appropriate_max = 13
    min_not_recommended = 11
    max_not_recommended = 19
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# infants(4-11)
for i in range(4, 12):
    min_recommended = 12
    max_recommended = 15
    may_be_appropriate_min = 10
    may_be_appropriate_max = 11
    min_not_recommended = 10
    max_not_recommended = 18
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# toddlers(12-24)
for i in range(12, 25):
    min_recommended = 11
    max_recommended = 14
    may_be_appropriate_min = 9
    may_be_appropriate_max = 10
    min_not_recommended = 9
    max_not_recommended = 16
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# preschoolers(36-60)
for i in range(36, 61):
    min_recommended = 10
    max_recommended = 13
    may_be_appropriate_min = 8
    may_be_appropriate_max = 9
    min_not_recommended = 8
    max_not_recommended = 14
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# school-aged-children(72-156)
for i in range(72, 157):
    min_recommended = 9
    max_recommended = 11
    may_be_appropriate_min = 7
    may_be_appropriate_max = 8
    min_not_recommended = 7
    max_not_recommended = 12
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# teenagers(168-204)
for i in range(168, 204):
    min_recommended = 8
    max_recommended = 10
    may_be_appropriate_min = 7
    may_be_appropriate_max = 11
    min_not_recommended = 7
    max_not_recommended = 11
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# young-adults(216-300)
for i in range(216, 301):
    min_recommended = 7
    max_recommended = 9
    may_be_appropriate_min = 6
    may_be_appropriate_max = 11
    min_not_recommended = 6
    max_not_recommended = 11
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# adults(312-768)
for i in range(312, 769):
    min_recommended = 7
    max_recommended = 9
    may_be_appropriate_min = 6
    may_be_appropriate_max = 10
    min_not_recommended = 6
    max_not_recommended = 10
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])
# older-adults(>=780)
for i in range(769, 780):
    min_recommended = 7
    max_recommended = 8
    may_be_appropriate_min = 5
    may_be_appropriate_max = 6
    min_not_recommended = 5
    max_not_recommended = 9
    sleep.append([i, min_recommended, max_recommended, may_be_appropriate_min, may_be_appropriate_max, min_not_recommended, max_not_recommended])

# Convert to DataFrame
# Membuat DataFrame (tabel data) dengan Pandas
sleepDF = pd.DataFrame(sleep, columns=headers_cols)
sleepDF.head(10)
sleepDF.to_csv(r'sleep_vs_age.csv')

# Plotting
# Import Seaborn untuk visualisasi statistik berbasis Matplotlib
import seaborn as sns
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
# Visualisasi menggunakan Seaborn
sns.set()

# Scatter plot
# Membuat scatter plot (plot sebaran)
plt.scatter(x=sleepDF["age"]/12., y=sleepDF["min_recommended"])
# Membuat scatter plot (plot sebaran)
plt.scatter(x=sleepDF["age"]/12., y=sleepDF["max_recommended"])
plt.xlabel('Age of person in Years')
plt.ylabel('Total hours of sleep required')
plt.show()

# Line plot
# Membuat grafik garis dengan Matplotlib
plt.plot(sleepDF['age']/12., sleepDF['min_recommended'], 'g--')
# Membuat grafik garis dengan Matplotlib
plt.plot(sleepDF['age']/12., sleepDF['max_recommended'], 'r--')
plt.xlabel('Age of person in Years')
plt.ylabel('Total hours of sleep required')
plt.show()


# Import Seaborn untuk visualisasi statistik berbasis Matplotlib
import seaborn as sns
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt

# Set some default parameters of matplotlib
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150

# Use style from seaborn. Try to comment the next line and see the difference in graph
# Visualisasi menggunakan Seaborn
sns.set()

# Load the Iris dataset
# Visualisasi menggunakan Seaborn
df = sns.load_dataset('iris')

df['species'] = df['species'].map({'setosa': 0, "versicolor": 1, "virginica": 2})

# A regular scatter plot
# Membuat scatter plot (plot sebaran)
plt.scatter(x=df["sepal_length"], y=df["sepal_width"], c = df.species)

# Create labels for axises
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Display the plot on the screen
plt.show()

# Load the Iris dataset
# Visualisasi menggunakan Seaborn
df = sns.load_dataset('iris')
df['species'] = df['species'].map({'setosa': 0, 'versicolor': 1,'virginica': 2})

# Create bubble plot
# Membuat scatter plot (plot sebaran)
plt.scatter(df.petal_length, df.petal_width, s=50*df.petal_length*df.petal_width, c=df.species, alpha=0.3)

# Create labels for axises
plt.xlabel('Septal Length')
plt.ylabel('Petal length')
plt.show()
# Visualisasi menggunakan Seaborn
df = sns.load_dataset('iris')
df['species'] = df['species'].map({'setosa': 0, "versicolor": 1,"virginica": 2})
# Visualisasi menggunakan Seaborn
sns.scatterplot(x=df["sepal_length"], y=df["sepal_width"], hue=df.species, data=df)
# House loan Mortgage cost per month for a year
houseLoanMortgage = [9000, 9000, 8000, 9000,8000, 9000, 9000, 9000,9000, 8000, 9000, 9000]
# Utilities Bills for a year
utilitiesBills = [4218, 4218, 4218, 4218,4218, 4218, 4219, 2218,3218, 4233, 3000, 3000]
# Transportation bill for a year
transportation = [782, 900, 732, 892,334, 222, 300, 800,900, 582, 596, 222]
# Car mortgage cost for one year
carMortgage = [700, 701, 702, 703,704, 705, 706, 707,708, 709, 710, 711]
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
# Import Seaborn untuk visualisasi statistik berbasis Matplotlib
import seaborn as sns
# Visualisasi menggunakan Seaborn
sns.set()
months= [x for x in range(1,13)]
# Create placeholders for plot and add required color
# Membuat grafik garis dengan Matplotlib
plt.plot([],[], color='sandybrown', label='houseLoanMortgage' )
# Membuat grafik garis dengan Matplotlib
plt.plot([],[], color='tan', label='utilitiesBills')
# Membuat grafik garis dengan Matplotlib
plt.plot([],[], color='bisque', label='transportation' )
# Membuat grafik garis dengan Matplotlib
plt.plot([],[], color='darkcyan', label='carMortgage')
# Add stacks to the plot
plt.stackplot(months, houseLoanMortgage, utilitiesBills, transportation,
carMortgage, colors=['sandybrown', 'tan', 'bisque', 'darkcyan'])
plt.legend()
# Add Labels
plt.title('Household Expenses')
plt.xlabel('Months of the year')
plt.ylabel('Cost')
# Display on the screen
plt.show()
# Create URL to JSON file (alternatively this can be a filepath)
url ='https://raw.githubusercontent.com/hmcuesta/PDA_Book/master/Chapter3/pokemonByType.csv'
# Load the first sheet of the JSON file into a data frame
# Membaca file CSV dari URL atau file lokal
pokemon = pd.read_csv(url, index_col='type')
pokemon
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
plt.pie(pokemon['amount'], labels=pokemon.index, shadow=False,
startangle=90, autopct='%1.1f%%',)
plt.axis('equal')
plt.show()
pokemon.plot.pie(y="amount", figsize=(20, 10))
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# Years under consideration
years = ["2010", "2011", "2012", "2913", "2014"]
# Available watt
columns = ['4.5W', '6.0W', '7.0W','S8.5W','9.5W','13.5W', '15W']
unitsSold = [
[65, 141, 88, 111, 104, 71, 99],
[85, 142, 89, 112, 103, 73, 98],
[75, 143, 98, 113, 89, 75, 93],
[65, 144, 91, 114, 98, 77, 92],
[55, 145, 92, 115, 88, 79, 93],
]
# Define the range and scale for the y axis
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
values =np.arange(0, 600, 100)
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
colors =plt.cm.OrRd(np.linspace(0, 0.7, len(years)))
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
index = np.arange(len(columns)) + 0.3
bar_width =0.7
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
y_offset =np.zeros(len(columns))
fig, ax =plt.subplots()
cell_text =[]
n_rows =len(unitsSold)
for row in range(n_rows):
# Membuat grafik batang vertikal
    plot =plt.bar(index, unitsSold[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset =y_offset +unitsSold[row]
    cell_text.append(['%1.1f' % (x) for x in y_offset])
    i=0
# Each iteration of this for loop, labels each bar with corresponding value for the given year
for rect in plot:
    height =rect.get_height()
    ax.text(rect.get_x() +rect.get_width()/2, y_offset[i],'%d' % int(y_offset[i]), ha='center', va='bottom')
    i =i+1
# Add a table to the bottom of the axes
the_table =plt.table(cellText=cell_text, rowLabels=years, rowColours=colors, colLabels=columns, loc='bottom')
plt.ylabel("Units Sold")
plt.xticks([])
plt.title('Number of LED Bulb Sold/Year')
plt.show()
subjects = ["C programming", "Numerical methods", "Operating system", "DBMS", "Computer Network"]
plannedGrade = [90, 95, 92, 68, 68, 90]
actualGrade = [75, 89, 89, 80, 80, 75]
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
theta = np.linspace(0, 2 * np.pi, len(plannedGrade) )
plt.figure(figsize = (10,6))
plt. subplot (polar=True)
(lines,labels) = plt.thetagrids(range(0, 360, int(360/len(subjects))), (subjects) )
# Membuat grafik garis dengan Matplotlib
plt.plot(theta, plannedGrade)
plt.fill(theta, plannedGrade, 'b', alpha=0.2)
# Membuat grafik garis dengan Matplotlib
plt.plot(theta, actualGrade)
plt.legend(labels=('Planned Grades', 'Actual Grades'),loc=1)
plt.title("Plan vs Actual grades by Subject")
plt.show()
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
#Create data set
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
yearsOfExperience = np.array([10, 16, 14, 5, 10, 11, 16, 14, 3, 14, 13, 19,2, 5, 7, 3, 20,11,
17, 16, 13, 18, 5, 7, 18, 15, 20, 2, 7, 0, 4, 14, 1, 14, 18,
8, 11, 12, 2, 9, 7, 11, 2, 6, 15, 2, 14, 13, 4, 6, 15, 3,
6, 10, 2, 11, 0, 18, 0, 13, 16, 18, 5, 14, 7, 14, 18])
nbins = 20
# Membuat histogram dari data
n, bins, patches = plt.hist(yearsOfExperience, bins=nbins)
plt.xlabel("Years of experience with Python Programming" )
plt.ylabel("Frequency")
plt.title("Distribution of Python programming experience in the vocational training session")
# Menghitung nilai rata-rata (mean)
plt.axvline(x=yearsOfExperience.mean(), linewidth=3, color = 'g')
plt.show()

plt.figure(figsize = (10,6))
nbins = 20
# Membuat histogram dari data
n, bins, patches = plt.hist(yearsOfExperience, bins=nbins,density=1)
plt.xlabel("Years of experience with Python Programming" )
plt.ylabel("Frequency")
plt.title("Distribution of Python programming experience in the vocational training session")
# Menghitung nilai rata-rata (mean)
plt.axvline(x=yearsOfExperience.mean(), linewidth=3, color = 'g')
# Menghitung nilai rata-rata (mean)
mu = yearsOfExperience.mean()
# Menghitung standar deviasi (penyebaran data)
sigma = yearsOfExperience.std()
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1/sigma* (bins - mu))**2))
# Membuat grafik garis dengan Matplotlib
plt.plot(bins, y, '--')
plt.show()
#Read the dataset
url = 'https://raw.githubusercontent.com/PacktPublishing/Hands-on-Exploratory-Data-Analysis-with-Python/refs/heads/master/Chapter%202/cardata.csv'
# Membaca file CSV dari URL atau file lokal
carDF =pd.read_csv(url)
carDF
#Group by manufacturer and take average mileage
# Menghitung nilai rata-rata (mean)
processedDF =carDF[['cty', 'manufacturer']].groupby('manufacturer').apply(lambda x: x.mean())
#Sort the values by cty and reset index
processedDF.sort_values('cty', inplace=True)
processedDF.reset_index(inplace=True)
#Plot the graph
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
ax.vlines(x=processedDF.index, ymin=0, ymax=processedDF.cty, color='firebrick', alpha=0.7, linewidth=2)
ax.scatter(x=processedDF.index, y=processedDF.cty, s=75,color='firebrick', alpha=0.7)

#Annotate Title
ax.set_title('Lollipop Chart for Highway Mileage using car dataset', fontdict={'size':22})
ax.set_ylabel('Miles Per Gallon')
ax. set_xticks(processedDF .index)
ax.set_xticklabels(processedDF.manufacturer.str.upper(),
rotation=65, fontdict={'horizontalalignment': 'right', 'size':12})
ax.set_ylim(0, 30)
#Write the values in the plot
for row in processedDF.itertuples():
    ax.text(row.Index, row.cty+.5, s=round(row.cty, 2),ha= 'center', va='bottom')
#Display the plot on the screen
plt.show()

#pertemuan ke 5
# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd

# Membuat DataFrame (tabel data) dengan Pandas
df1SE = pd.DataFrame({'StudentID': [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
                      'ScoreSE': [22, 66, 31, 51, 71, 91, 56, 32, 52, 73, 92]})

# Membuat DataFrame (tabel data) dengan Pandas
df2SE = pd.DataFrame({'StudentID': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                      'ScoreSE': [98, 93, 44, 77, 69, 56, 31, 53, 78, 93, 56, 77, 33, 56, 27]})

# Membuat DataFrame (tabel data) dengan Pandas
df1ML = pd.DataFrame({'StudentID': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
                      'ScoreML': [39, 49, 55, 77, 52, 86, 41, 77, 73, 51, 86, 82, 92, 23, 49]})

# Membuat DataFrame (tabel data) dengan Pandas
df2ML = pd.DataFrame({'StudentID': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      'ScoreML': [93, 44, 78, 97, 87, 89, 39, 43, 88, 78]})

# Option 1
# Menggabungkan beberapa DataFrame secara vertikal atau horizontal
dfSE = pd.concat([df1SE, df2SE], ignore_index=True)
# Menggabungkan beberapa DataFrame secara vertikal atau horizontal
dfML = pd.concat([df1ML, df2ML], ignore_index=True)
# Menggabungkan beberapa DataFrame secara vertikal atau horizontal
df1 = pd.concat([dfML, dfSE], axis=1)

df1

# Option 2
# Menggabungkan dua DataFrame berdasarkan kolom kunci
df2 = dfSE.merge(dfML, how='inner')
df2

# Menggabungkan dua DataFrame berdasarkan kolom kunci
df3 = dfSE.merge(dfML, how='left')
df3

# Menggabungkan dua DataFrame berdasarkan kolom kunci
df4 = dfSE.merge(dfML, how='right')
df4

# Menggabungkan dua DataFrame berdasarkan kolom kunci
df5 = dfSE.merge(dfML, how='outer')
df5

# Buat dataframe
# Membuat DataFrame (tabel data) dengan Pandas
left1 = pd.DataFrame({'key': ['apple', 'ball', 'apple', 'apple', 'ball', 'cat'],
                      'value': range(6)})
# Membuat DataFrame (tabel data) dengan Pandas
right1 = pd.DataFrame({'group_val': [33.4, 5.0]}, index=['apple', 'ball'])
left1

# Menggabungkan dua DataFrame berdasarkan kolom kunci
df1 = pd.merge(left1, right1, left_on='key', right_index=True)
df1

# Menggabungkan dua DataFrame berdasarkan kolom kunci
df2 = pd.merge(left1, right1, left_on='key', right_index=True, how='outer')
df2

# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
data = np.arange(15).reshape((3, 5))
indexers = ['Rainfall', 'Humidity', 'Wind']
# Membuat DataFrame (tabel data) dengan Pandas
dframe1 = pd.DataFrame(data, index=indexers, columns=['Bergen', 'Oslo', 'Trondheim', 'Stavanger', 'Kristiansand'])
dframe1

stacked = dframe1.stack()
stacked

stacked.unstack()

series1 = pd.Series([000, 111, 222, 333], index=['zeros', 'ones', 'twos', 'threes'])
series2 = pd.Series([444, 555, 666], index=['fours', 'fives', 'sixes'])
# Menggabungkan beberapa DataFrame secara vertikal atau horizontal
frame2 = pd.concat([series1, series2], keys=['Number1', 'Number2'])
frame2

frame2.unstack()

# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd
# Membuat DataFrame (tabel data) dengan Pandas
frame3 = pd.DataFrame({'column 1': ['Looping'] * 3 + ['Functions'] * 4,
                       'column 2': [10, 10, 22, 23, 23, 24, 24]})
frame3

frame3.duplicated()

frame4 = frame3.drop_duplicates()
frame4

frame3['column 3'] = range(7)
frame3

frame3['column 3'] = range(7)
frame5 = frame3.drop_duplicates(['column 2'])
frame5

# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# Membuat DataFrame (tabel data) dengan Pandas
replaceFrame = pd.DataFrame({'column 1': [200., 3000., -786., 3000., 234., 444., -786., 332., 3332.],
                             'column 2': range(9)})
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
replaceFrame.replace(to_replace=-786., value=np.nan)

# Membuat DataFrame (tabel data) dengan Pandas
replaceFrame = pd.DataFrame({'column 1': [200., 3000., -786., 3000., 234., 444., -786., 332., 3332.],
                             'column 2': range(9)})
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
replaceFrame.replace(to_replace=[-786., 0], value=[np.nan, 2])

# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
data = np.arange(15, 30).reshape(5, 3)
# Membuat DataFrame (tabel data) dengan Pandas
dfx = pd.DataFrame(data, index=['apple', 'banana', 'kiwi', 'grapes', 'mango'], columns=['store1', 'store2', 'store3'])
dfx

# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
dfx['store4'] = np.nan
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
dfx.loc['watermelon'] = np.arange(15, 19)
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
dfx.loc['oranges'] = np.nan
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
dfx['store5'] = np.nan
dfx['store4']['apple'] = 20
dfx
dfx.isnull()

dfx.notnull()

dfx.isnull().sum()

dfx.isnull().sum().sum()

dfx.count()

# Mengambil data dari store4 yang tidak null
dfx.store4[dfx.store4.notnull()]

# Menghapus nilai NaN dari kolom store4
# Menghapus baris/kolom yang mengandung nilai NaN
dfx.store4.dropna()

# Menghapus baris yang memiliki NaN dari seluruh DataFrame
# Menghapus baris/kolom yang mengandung nilai NaN
dfx.dropna()

# DROPPING BY ROW
# Menghapus baris/kolom yang mengandung nilai NaN
dfx.dropna(how='all')

# DROPPING BY COLUMNS
# Menghapus baris/kolom yang mengandung nilai NaN
dfx.dropna(how='all', axis=1)

# DROPPING dengan threshold
# Menghapus baris/kolom yang mengandung nilai NaN
dfx.dropna(thresh=5, axis=1)

#pertemuan ke 6
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
ar1 = np.array([100, 200, np.nan, 300])
ser1 = pd.Series(ar1)
# Menghitung nilai rata-rata (mean)
ar1.mean(), ser1.mean()
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
data = np.arange(15, 30).reshape(5, 3)
# Membuat DataFrame (tabel data) dengan Pandas
dfx = pd.DataFrame(data, index = ['apple', 'banana', 'kiwi', 'grapes', 'mango'],
columns = ['store1', 'store2', 'store3'])
dfx
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
dfx['store4'] = np.nan
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
dfx.loc['watermelon'] = np.arange(15, 19)
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
dfx.loc['oranges'] = np.nan
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
dfx['store5'] = np.nan
dfx['store4']['apple'] = 20.
dfx
#Let's compute the total quantity of fruits sold by store4:
ser2 = dfx.store4
ser2.sum()
# Menghitung nilai rata-rata (mean)
ser2.mean()
#Note that NaNs are treated as @s. It is the same for cumulative summing:
ser2.cumsum()
# Mengganti nilai NaN dengan nilai lain (seperti 0 atau metode forward/backward fill)
filleDf=dfx.fillna(0)
filleDf
# Note that in the preceding dataframe, all the NaN values are replaced by 0. Replacing the
# values with 0 will affect several statistics including mean, sum, and median.
# Menghitung nilai rata-rata (mean)
dfx.mean()
#Now, let's compute the mean from the filled dataframe with the following command:
# Menghitung nilai rata-rata (mean)
filleDf.mean()
#Note that there are slightly different values. Hence, filling with 0 might not be the optimal solution.
# Mengganti nilai NaN dengan nilai lain (seperti 0 atau metode forward/backward fill)
dfx. store4. fillna(method='ffill' )
# The direction of the fill can be changed by changing method="bfill'
# Mengganti nilai NaN dengan nilai lain (seperti 0 atau metode forward/backward fill)
dfx.store4.fillna(method="bfill")
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
ser3=pd.Series([100,np.nan,np.nan,np.nan,292])
ser3.interpolate()
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
data=np.arange(15).reshape((3,5))
indexers=['Rainfall', 'Humidity', 'Wind']
# Membuat DataFrame (tabel data) dengan Pandas
dframe1=pd.DataFrame(data, index=indexers,columns= [ 'Bergen', 'Oslo', 'Trondheim', 'Stavanger', 'Kristiansand'])
dframe1
dframe1.index=dframe1.index.map(str.upper)
dframe1
dframe1.rename(index=str.title, columns=str.upper)
dframe1.rename(index=str.lower, columns=str.upper)
#1. Let's say we have data on the heights of a group of students as follows:
height = [120, 122, 125, 127, 121, 123, 137, 131, 161, 145, 141,132]
height
bins = [118, 125, 135, 160, 200]
category = pd.cut(height, bins)
category
#3. We can set a right=False argument to change the form of interval:
category = pd.cut(height, [118, 126, 136, 161, 266], right=False)
category
#4. We can check the number of values in each bin by using the
pd.value_counts(category)
#5. We can also indicate the bin names by passing a list of labels:
bin_names = ['Short Height', 'Average height', 'Good Height', 'Taller']
pd.cut(height, bins, labels=bin_names)
# Membaca file CSV dari URL atau file lokal
df = pd.read_csv('https://raw.githubusercontent.com/PacktPublishing/hands-on-exploratory-data-analysis-with-python/master/Chapter%204/sales.csv')
df.head(10)
#2. Now, suppose we want to calculate the total price based on the quantity sold and
# the unit price. We can simply add a new column, as shown here:
df['TotalPrice' ]=df['UnitPrice']*df['Quantity']
df
TotalTransaction = df['TotalPrice']
TotalTransaction[np.abs(TotalTransaction) > 3000000]
df[np.abs(TotalTransaction) > 654]
---
# **Latihan 6**
---
# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd
# Import library NumPy untuk manipulasi array dan operasi matematika
import numpy as np
# 1. Load Data CSV
# Membaca file CSV dari URL atau file lokal
df = pd.read_csv('https://cdn.uisi.ac.id/covid_2020-12-01.csv?authuser=1')
# 2. Tampilkan 50 data awal
df.head(50)
# 3. Tampilkan jumlah baris dan kolom
print(f"Jumlah Baris: {df.shape[0]}")
print(f"Jumlah Kolom: {df.shape[1]}\n")
df.info()
# 4. Tambahkan kolom baru dengan singkatan nama anda serta isi dengan NAN
# Operasi menggunakan NumPy untuk membuat array atau melakukan kalkulasi
df['ari'] = np.nan
df['ari'].head()
# 5. Ubahlah nilai NAN dengan string 'NOL'
# Mengganti nilai NaN dengan nilai lain (seperti 0 atau metode forward/backward fill)
df['ari'] = df['ari'].fillna('NOL')
df['ari'].head()
# 6. Tampilkan total jumlah kasus untuk usia 60 tahun ke atas untuk provinsi dengan kasus lebih dari 500
df[['jumlah_kasus', 'usia_60_keatas']].loc[df['jumlah_kasus'] > 500]

#Soal1

# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd


data = {
    'Bilangan_Bulat': [1, 2, 3, 4, 5],
    'Nama': ['Ari', 'Natasya', 'wawati', 'Gundul', 'Rendra'],
    'Nilai': [87.5, 92.0, 78.5, 85.0, 90.5]
}


# Membuat DataFrame (tabel data) dengan Pandas
df = pd.DataFrame(data)


print(df)

#Soal2
# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd


file_url = 'https://bit.ly/ADE_quiz'


# Membaca file CSV dari URL atau file lokal
df = pd.read_csv(file_url)

df.head()


df.info()


# Menampilkan ringkasan statistik data
df.describe()


#Soal3
# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd


url = 'https://bit.ly/ADE_quiz'
# Membaca file CSV dari URL atau file lokal
df = pd.read_csv(url)


df.head()

#Soal4
# Menampilkan jumlah baris dan kolom
baris, kolom = df.shape
print(f"Jumlah baris: {baris}")
print(f"Jumlah kolom: {kolom}")

#Soal5

# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd


url = 'https://bit.ly/ADE_quiz'
# Membaca file CSV dari URL atau file lokal
df = pd.read_csv(url)


print(df['wind'])

#Soal6
# Import library Pandas untuk analisis dan manipulasi data
import pandas as pd


url = 'https://bit.ly/ADE_quiz'
# Membaca file CSV dari URL atau file lokal
df = pd.read_csv(url)


print(df.iloc[20])

#Soal7
#menghitung nilai rata rata pada frame dat 1

# Menghitung nilai rata-rata (mean)
rata_rata = df['Bilangan_Bulat'].mean()
print(f"Rata-rata Bilangan_Bulat: {rata_rata}")

#Soal8
#mencarri nilai tengah pada data frame 1

# Menghitung nilai tengah (median)
median = df['Bilangan_Bulat'].median()
print(f"Median (nilai tengah) Bilangan_Bulat: {median}")

#Soal9
#menghitung standart deviasi

# Menghitung standar deviasi (penyebaran data)
std_dev = df['Bilangan_Bulat'].std()
print(f"Standar deviasi Bilangan_Bulat: {std_dev}")
#Soal10 tampilkan quartil 3
# Menghitung Quartil 3

q3 = df['Bilangan_Bulat'].quantile(0.75)
print(f"Quartil 3 (Q3) Bilangan_Bulat: {q3}")


#Soal11
#Buatlah Box Plotnya

# Import modul pyplot dari Matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
# Import Seaborn untuk visualisasi statistik berbasis Matplotlib
import seaborn as sns


# Visualisasi menggunakan Seaborn
sns.boxplot(x=df['Bilangan_Bulat'])


plt.title('Box Plot untuk Bilangan_Bulat')
plt.show()

#Soal2
#Buatlah visualisasi Dari salah Satu Data Frame Yang Anda Buat Pilih jenis grafik yang akan anda buat

#nilai grafik berdasarkan nama pada data frame 1

plt.figure(figsize=(8, 6))
# Membuat grafik batang vertikal
plt.bar(df['Nama'], df['Nilai'], color='skyblue')


plt.title('Nilai Siswa berdasarkan Nama', fontsize=14)
plt.xlabel('Nama', fontsize=12)
plt.ylabel('Nilai', fontsize=12)


plt.show()

#Soal13


from cProfile import label
from operator import index
import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.figure_factory as pl
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
from PIL import Image
from wordcloud import WordCloud

import plotly.express as px

#Scaling & metrics libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

## Classification algos libraries
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

## Regression algos libraries
from sklearn.linear_model import LinearRegression #Linear Regression is a Machine Learning classification algorithm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split #Splitting of Dataset
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

st.set_option('deprecation.showPyplotGlobalUse', False)
pd.set_option('display.max_columns', None)
alt.data_transformers.disable_max_rows()

df_zomato =pd.read_csv("df_zomato_v1.csv") 

#Top cities vs restaurants
df_zomato_subset = df_zomato[df_zomato.city.isin(["Bangalore","Mumbai","Pune","Chennai","New Delhi","Jaipur","Kolkata","Ahmedabad","Goa","Lucknow"])]
df_zomato_cities = df_zomato_subset.copy()

#Top cities vs Metro
df_zomato_metro = df_zomato[df_zomato.city.isin(["Mumbai","Chennai","New Delhi","Kolkata","Hyderabad","Bangalore"])]

#Top cities vs Non-Metro
df_zomato_non_metro = df_zomato[~df_zomato.city.isin(["Mumbai","Chennai","New Delhi","Kolkata","Hyderabad","Bangalore"])]

# Establishments types data clean-up to have proper strings..
df_zomato_establishments = df_zomato.groupby("establishment").count()["res_id"].sort_values(ascending=False).head(7).rename_axis("establishment").reset_index(name="restaurants_count")

df_zomato["establishment"] = df_zomato["establishment"].apply(lambda x:x[2:-2])

# Establishments subset..
df_zomato_establishment_subset = df_zomato[df_zomato.establishment.isin(["Quick Bites","Casual Dining","Café","Bakery","Dessert Parlour","Sweet Shop","Beverage Shop"])]
df_zomato_establishment_subset = df_zomato_establishment_subset[(df_zomato_establishment_subset['average_cost_for_two'] < 2000)]

#Create 'metro' categorical value
metro_list=df_zomato.city.isin(["Mumbai","New Delhi","Chennai","Kolkata","Hyderabad","Bangalore"])

#Define options for analysis
options=["General overview","City","Establishment","Data Analysis Summary","ML_Classification_Modeling","ML_Regression_Modeling","ML_User_Predictions","ML Modeling & Prediction Summary"]

selected_type = st.sidebar.radio('Data Analysis, Modeling & Predictions',options)

st.write(""" 
    # Zomato Restaurants in India Analysis
""")

st.write("")
st.write("")
st.write("")
st.write("")

#options = ["Classification", "Regression"]
#selected_type_ML = st.sidebar.radio('ML Models',options_ML)

if (selected_type == "General overview"):

    img = Image.open("zomato_logo.jpg")
    st.image(img)

    df_zomato =pd.read_csv("df_zomato_v1.csv")

    if st.checkbox("Project Motive"):
         st.write(
          """
            Prior to Covid pandemic, Online ordering for food or groceries wasn't considered majorly and people mostly visited the stores/restaurants to buy the requisite items. 
            During pandemic, the Government or self imposed restrictions and safety measures ensured that we chose online ordering as one of the primary options for purchasing groceries or food. 
            Now, online ordering has become a major phenomenon and is not just restricted to ordering from home or office in India. People do order food or items from trains or shops 
            and we have so many apps (Zomato, Dunzo, Swiggy, Travelkhana, Zepto etc) catering to the online delivery needs. Various factors (like Location, Establishment types, Cost, Rating, Votes, Coupons) 
            effect the way we plan our ordering on the apps. The Mini-project goal is to study some of the key factors influencing online purchase.
    """
    )

    if st.checkbox("Dataset description"):
         st.write(
          """
            Zomato is one of the popular food ordering mobile apps in India. The 'Zomato Indian Restaurants' dataset used for analysis is picked up from Kaggle and primarily provides 
            details of various factors (Cities, Chains, Cuisines, Ratings, Cost, Locality etc) involved in Online ordering through Zomato. 
      """)  

    if st.checkbox("Issues with Dataset"):
        st.write(
          """
             The 'Zomato Indian Restaurants' dataset used for analysis is picked up from Kaggle. This dataset primarily provides details of various attributes (City names, Establishment type, Cuisines, Ratings, Costetc) 
            involved in Online ordering through Zomato app. Highlighted below are few issues with the dataset.     
                1. Big file size (~ 114 mb). Issue was primarily with duplicate values & irrelevant fields. ~75% of data has duplicate values for 'res_id' (restaurant ids) and removing the 
                duplicate values helped reduce the file size to a major extent.   
                2. Multiple irrelevant & redundant fields/attributes (like Latitude, Longitude, Address, Country-id).  The dataset has around 28 columns of which only around 12 columns are relevant for EDA. 
                Removing the unnecessary columns further reduced the size of the dataset meant for analysis.  
                3. Missingness with respect to 1 of the variables (zipcode) is of type Missingness Completely at Random (MCAR). This field was omitted as City name can be used instead of zipcode to confirm the location.   
                4. Only 1 categorical column (price_range). This issue was addressed by creating 2 more categorical columns ('metro' & 'rating').  
        """)

    if st.checkbox("Shape (rows X columns)"):
        st.write(df_zomato.shape)
        st.write(
          """
            The dataset after pre-processing & clean-up has 14 columns/attributes and 55568 rows/records.
          """)

    if st.checkbox("Attributes"):
        st.write(list(df_zomato.columns))

    if st.checkbox("Data Overview"):
        st.write(df_zomato.head())

    if st.checkbox("Ratings distribution"):
        explode = (0, 0.03) 
        labels = df_zomato['rating'].unique()
        rating_counts = df_zomato['rating'].value_counts()

        fig, ax = plt.subplots()
        ax.pie(rating_counts, labels=labels, autopct='%.f%%')
        ax.set_title("Ratings distribution pie chart")
        ax.legend()
        st.pyplot(fig,size=(1,1))  

        st.write(
          """
              Ratings distribution piechart shows that majority (~80%) of the ratings are in 4-5 range. Ratings are 1 of the key factors influencing online ordering and a healthy rating (3.5-5) will always motivate more people to use online ordering.
        """)

    if st.checkbox("Average Cost distribution"):
        df_zomato_cost_subset = df_zomato[(df_zomato['average_cost_for_two'] < 5000)]
        plt.xlabel("Average cost in rupees (Indian currency)")
        fig1 = plt.figure()
        ax = fig1.gca()
        df_zomato_cost_subset["average_cost_for_two"].plot(kind="kde", xlim=(0,3000), xlabel="Average Cost")
        ax.set_title("Average Cost per order (in Indian Rupees)")
        st.pyplot(fig1)
        st.write(
          """
              Average cost distribution shows that majority of the online orders are in the range of 0-1000/- Rupees (~ 12-13$ USD ). As can be observed, availability of low cost options is 1 of the key factors influencing the online ordering.
        """)

    if st.checkbox("Rating vs Average Cost"):
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.xlabel("Rating")
        plt.ylabel("Average Cost")

        df_zomato_scaled_cost = df_zomato[(df_zomato['average_cost_for_two'] <2500)]
        sns.violinplot(x=df_zomato_scaled_cost.rating,y=df_zomato.average_cost_for_two)
        #sns.barplot(x="rating",y="average_cost_for_two",data=df_zomato_cities,order=["Bangalore","Mumbai","Pune","Chennai","New Delhi","Jaipur","Kolkata"],errorbar=None)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title("Rating vs Average Cost")
        plt.legend()

        st.pyplot(fig,size=(1,1))

        st.write(
        """
              As can be observed, average cost for almost all the ratings (0-4) have the median of around ~Rs. 500/-. For the highest rating (5), the median cost is around Rs. 1000/-
              and is almost double that of the lower ratings. This significantly implies that ratings are always associated with the higher price range.
        """)

        #sns.violinplot(x=df_zomato_scaled_cost.rating,y=df_zomato.average_cost_for_two)

if (selected_type == "City"):

    st.write(""" 
        ### Analysis with reference to cities
        """)

    city_names = df_zomato["city"].unique()

    hl_str = ""
    for i in city_names:
        hl_str += str(i) + " "
    wordcloud = WordCloud(width = 800, height = 400, 
                    background_color ='white', 
                    min_font_size = 10, max_words=30).generate(hl_str) 

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show()
    st.pyplot()

    if st.checkbox("Top Cities with reference to Restaurants Count"):
        city_res_counts = df_zomato.groupby("city").count()["res_id"].sort_values(ascending=False).head(15).rename_axis("city").reset_index(name="restaurant_counts")

        labels = city_res_counts.city
        values = city_res_counts.restaurant_counts

        fig = plt.figure(figsize=(8,7))
        plt.xlabel("City Name")
        plt.ylabel("Restaurants Count")

        ax = fig.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        #Used the VIBGYOR pattern for colors..
        colors_list = ["Violet","Indigo","Blue","Green","Yellow","Orange","Red"]

        graph = plt.bar(labels,values,color=colors_list)
        plt.xticks(rotation=70)
        plt.title("City vs Restaurants count")

        i=0
        for p in graph:
            wid = p.get_width()
            ht = p.get_height()
            x,y = p.get_xy()
            plt.text(x+wid/2,y+(ht*1.01),  str(values[i]), ha='center',weight='bold')
            i+=1
        
        st.pyplot(fig,size=(1,1))

        st.write(
        """
              The graph shows that the online ordering is available not just in Metros (Mega-cities), but even in Tier-2 & Tier-3 cities. 
              While Metro cities (Delhi, Mumbai, Chennai, Kolkatta, Bangalore, Hyderabad) have more online ordering options, even the lower rung cities have comparable number of options to order online.
        """)

    if st.checkbox("City vs Ratings for Top 7 Cities"):
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.xlabel("City Name")
        plt.ylabel("Ratings")

        sns.barplot(x="city",y="rating",data=df_zomato_cities,order=["Bangalore","Mumbai","Pune","Chennai","New Delhi","Jaipur","Kolkata"],errorbar=None)
        ax.bar_label(ax.containers[0],fmt='%.1f%%')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title("Top Cities vs Ratings")
        plt.legend()

        st.pyplot(fig,size=(1,1))

        st.write(
        """
              As can be seen, average online order ratings for almost all the top cities is ~3.7 (out of 5). A healthy rating is very important factor influencing the ordering and all the cities have 
              decent ratings for the online orders..
        """)


    if st.checkbox("Metro vs Non-Metro ratio for Restaurant counts"):
        fig, ax = plt.subplots()

        labels = ["Non-Metro","Metro"]
        metro_counts = df_zomato['metro'].value_counts()

        explode = (0, 0.03) 

        ax.pie(metro_counts, labels=labels, autopct='%1.1f%%',explode=explode)
        ax.set_title("Metro vs Non-Metro restaurants")
        ax.legend()

        st.pyplot(fig,size = (1,1)) 

        st.write(
        """
            The pie-chart shows that the online ordering is not just limited to Metros/mega-cities. Infact, the non-Metro (Tier-2 & Tier-3) cities contribute to ~80% of the restaurants count.
        """)

    if st.checkbox("Average Cost"):
        st.write(""" 
            Metro or Non-Metro.. Please select from slider..
        """)

        #Top cities vs Metro
        df_zomato_metro = df_zomato[df_zomato.city.isin(["Mumbai","Chennai","New Delhi","Kolkata","Hyderabad","Bangalore"])]

        df_zomato_metro = df_zomato_metro[(df_zomato_metro['average_cost_for_two'] < 3000)]
        df_zomato_metro = df_zomato_metro[(df_zomato_metro['average_cost_for_two'] > 0)]

        #Top cities vs Non-Metro
        df_zomato_non_metro = df_zomato[~df_zomato.city.isin(["Mumbai","Chennai","New Delhi","Kolkata","Hyderabad","Bangalore"])]

        df_zomato_non_metro = df_zomato_non_metro[(df_zomato_non_metro['average_cost_for_two'] < 3000)]
        df_zomato_non_metro = df_zomato_non_metro[(df_zomato_non_metro['average_cost_for_two'] > 0)]

        opt=st.sidebar.select_slider('Metro or Non-Metro?', options=['No', 'Yes'])

        if(opt=='Yes'):
            fig1 = plt.figure()
            sns.distplot(df_zomato_metro.average_cost_for_two,rug=True,kde_kws={"lw":3},rug_kws={"height":0.05, 'color':'xkcd:red'})
            plt.title("Metros vs Average Cost")

            ax = fig1.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.title("Metros vs Average Cost")
            st.pyplot(fig1,size=(1,1))

        elif(opt=='No'):
            fig1 = plt.figure()
            sns.distplot(df_zomato_non_metro.average_cost_for_two,rug=True,kde_kws={"lw":3},rug_kws={"height":0.05, 'color':'xkcd:red'})

            ax = fig1.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.title("Non-Metros vs Average Cost")
            plt.xlim=(0,6000)
            st.pyplot(fig1,size=(1,1))

        st.write(
        """
            As can be observed, Average cost for non-metro restaurants is in the range of Rs 300-500/- while the average cost for Metro-ordering is in the range of Rs. 700-900. 
            This difference in average cost between Metro & non-Metro restaurants can be primarily attributed to the higher cost of living in Metro cities.
        """)    

    if st.checkbox("Metros Aggregate Rating vs Average Cost for Two"):

        source = df_zomato_metro

        brush = alt.selection(type = 'interval')
        points = alt.Chart(source).mark_point().encode(
            x = 'aggregate_rating:Q',
            y = 'average_cost_for_two:Q',
            color = alt.condition(brush, 'city:N', alt.value('lightgrey'))
        ).add_selection(
            brush
        )
        bars = alt.Chart(source).mark_bar().encode(
            y = 'city:N',
            color = 'city:N',
            x = 'count(city):Q',
        ).transform_filter(
            brush
        )

        chart = points & bars
        st.altair_chart(points & bars)
        st.write(
        """
            The interactive graph for Aggregate-rating vs Average cost in Metros/Mega-cities provides us with multiple insights. Shared below are few:

            1. No restaurants found in 0.5 to 1.5 ratings.
            2. Chennai & Bangalore have the most restaurants with 2-2.5 ratings while Delhi & Kolkatta have the least number of such ratings. 
            3. Bangalore has the highest number of restaurants with 4.5-5 ratings range.
            4. Mumbai & New Delhi have the highest number of restaurants in the price range greater than Rs. 5000/-. Surprisingly, 1 of the online orders from Mumbai has an associated cost of Rs.30,000/-, which is considerably high for an online order.
        """)    

if (selected_type == "Establishment"):
    st.write(""" 
        ### Analysis with reference to establishments
        """)

    establishment_type = df_zomato["establishment"].unique()

    hl_str = ""
    for i in establishment_type:
        hl_str += str(i) + " "
    wordcloud = WordCloud(width = 800, height = 300, 
                    background_color ='white', 
                    min_font_size = 10, max_words=40).generate(hl_str) 

    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show()
    st.pyplot()

    if st.checkbox("Establishment type vs Restaurants Count"):
        labels = df_zomato_establishments.establishment
        values = df_zomato_establishments.restaurants_count

        fig = plt.figure(figsize=(4,4))
        plt.xlabel("Establishment type")
        plt.ylabel("Restaurants Count")
        plt.xticks(rotation='vertical')

        ax = fig.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        #Used the VIBGYOR pattern for colors..
        colors_list = ["Violet","Indigo","Blue","Green","Yellow","Orange","Red"]

        graph = plt.bar(labels,values,color=colors_list)
        plt.title("Establishment type vs Restaurants count")

        i=0
        for p in graph:
            wid = p.get_width()
            ht = p.get_height()
            x,y = p.get_xy()
            plt.text(x+wid/2,y+(ht*1.01),  str(values[i]), ha='center',weight='bold')
            i+=1
            
        st.pyplot(fig,size = (1,1)) 

        st.write(
        """
            The bar chart provides the distribution of restaurant counts based on the Establishment type (Quick Bites, Casual Dining, Cafe, Sweets, Bakery etc). As can be observed, 'Quick Bites' & 'Casual Dining'
            restaurants have a considerably higher number of restaurants in comparision with other establishment types.
        """)

    if st.checkbox("Average Cost for Two"):
        chart1=alt.Chart(df_zomato_establishment_subset).mark_bar().encode(
            x = alt.X('average_cost_for_two:Q', 
                    axis=alt.Axis(title='Average Cost for Two'), 
                    scale=alt.Scale(zero=False),
                    bin=alt.Bin(maxbins=20)),
            y = alt.Y('count():Q', 
                    axis=alt.Axis(title=''))
        ).properties(
            width=200,
            height=200
        ).facet(
            alt.Column('establishment:N', sort = alt.EncodingSortField(order=None)),
            align= 'all',
            padding=0,
            columns=4,
            spacing=0
        ).resolve_axis(
            x='independent',
            y='independent'
        ).resolve_scale(
            x='independent', 
            y='independent'
        )
        st.altair_chart(chart1)

        st.write(
        """
            The facet graphs provide a decent insight into the Average Cost with reference to the Establishment types. As can be noticed, Casual Dining establishments have a higher cost (Rs. 600-1200/-) associated
             in comparision with the other establishment types.
        """)
          
    if st.checkbox("Aggregate Rating vs Photo Count"):
       
        chart=alt.Chart(df_zomato_establishment_subset).mark_bar().encode(
            x = alt.X('aggregate_rating:Q', 
                    axis=alt.Axis(title='Aggregate rating'), 
                    scale=alt.Scale(zero=False),
                    bin=alt.Bin(maxbins=20)),
            y = alt.Y('photo_count:Q', 
                    axis=alt.Axis(title='Photo count'))
        ).properties(
            width=200,
            height=200
        ).facet(
            alt.Column('establishment:N', sort = alt.EncodingSortField(order=None)),
            align= 'all',
            padding=0,
            columns=4,
            spacing=0
        ).resolve_axis(
            x='independent',
            y='independent'
        ).resolve_scale(
            x='independent', 
            y='independent'
        )

        st.altair_chart(chart)

        st.write(
        """
            The facet graphs provide a decent insight into Aggregate rating vs Photo Count with reference to the Establishment types. As can be observed, the photo counts are much
            higher for the 3.5 to 5 range and also significantly higher for the Cafe & Casual Dining options...
        """)

if (selected_type == "Data Analysis Summary"):

    img = Image.open("zomato_logo.jpg")
    st.image(img)

    st.write(
          """
          Online ordering has become a major phenomenon in the recent days (especially post Covid) and more and more applications are being developed and used by people
          for various reasons like ordering food, clothing & apparels, groceries, retail stuff, healthcare consultations etc. In such a situation, it's more important to 
          make an educated reasoning into the key factors influencing decision-making and further enhance the usage of the corresponding applications.

          To summarize, the 'Zomato Indian Restaurants' dataset helped come up with interesting insights into online ordering with reference to various factors like 
          location, price, ratings, establishment type. It has lot more scope for further analysis. 
    """
    )

def model_summary(y_test,y_pred):
    accuracy = accuracy_score(y_test,y_pred)
    st.write("Accuracy")
    st.write(accuracy)
    st.write("Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.write(df)
    st.write("Confusion Matrix")    
    cm = confusion_matrix(y_test, y_pred)
    st.plotly_chart(px.imshow(cm,color_continuous_scale='Viridis', origin='lower',text_auto=True))

df_zomato =pd.read_csv("df_zomato_v1.csv") 
df_zomato_establishments = df_zomato.groupby("establishment").count()["res_id"].sort_values(ascending=False).head(7).rename_axis("establishment").reset_index(name="restaurants_count")
df_zomato["establishment"] = df_zomato["establishment"].apply(lambda x:x[2:-2])

df_zomato_establishment_subset = df_zomato[df_zomato.establishment.isin(["Quick Bites","Casual Dining","Café","Bakery","Dessert Parlour","Sweet Shop","Beverage Shop"])]
df_zomato_establishment_subset = df_zomato_establishment_subset[(df_zomato_establishment_subset['average_cost_for_two'] < 8000)]

le = LabelEncoder()
df_zomato_establishment_subset['city'] = le.fit_transform(df_zomato_establishment_subset['city'])
df_zomato_establishment_subset['establishment'] = le.fit_transform(df_zomato_establishment_subset['establishment'])

x = df_zomato_establishment_subset.iloc[:,[2,3,5,10,11]]
y = df_zomato_establishment_subset.iloc[:,13]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=0)

if (selected_type == "ML_Classification_Modeling"):
    st.write("Trying some Classifiication Models to accurately predict 'rating'")

    st.write("'rating' parameter is a categorical variable created to denote a restaurant rating by the users. The possible values are (0,1,2,3,4,5) where 5 denotes the highest rating")
    st.write("Some of the key parameters influencing rating are 'votes', 'city', 'establishment type' and 'average_cost_for_two'")
    st.write("For classification modeling, I used KNN, DecisionTree, Randon Forest & SVM model to verify the rating accuracy")
    st.write("Scaling was also used for input parameters during modeling. However, the streamlit app was focussed on predictions without scaling.")
    st.write("Users are given options to play around with 1 key hyperparameter per Model")
    st.write("Accuracy, Classification report & Confusion Matrix were used as the metrics in measuring a classification model's accuracy")
    st.write("As per the observations, SVM is taking the maximum time")
    st.write("Accuracy is observed to be highest for Random Forest & Decision tree classifiers and the value is around ~83%")

    type = st.radio(
        "Select type of algorithm",
        ("KNN", "Decision Tree","Random Forest","SVM")
    )

    if(type=="KNN"):
        st.write("K-nearest Neighbour")
        param = st.multiselect('Select features',['city', 'establishment', 'average_cost_for_two','votes','photo_count'],['city'])
        neighbor_count = st.slider('Number of Neighbors', 2, 10, 5)

        if(st.button('Model predictions & results, Confusion Matrix')):
            knn = KNeighborsClassifier(n_neighbors = neighbor_count,p=2)
            knn.fit(x_train.loc[:,param], y_train)
            y_pred = knn.predict(x_test.loc[:,param])
            model_summary(y_test,y_pred)

    if(type=="Decision Tree"):
        st.write("Decision Tree Classifier")          
        param = st.multiselect('Select features',['city', 'establishment', 'average_cost_for_two','votes','photo_count'],['city'])
        max_depth_DT = st.slider('Max Depth', 2, 20, 10)

        if(st.button('Model predictions & results, Confusion Matrix')):
            classifier_dTree = DecisionTreeClassifier(criterion='entropy', random_state=0,max_depth=max_depth_DT)
            classifier_dTree.fit(x_train.loc[:,param], y_train)
            y_pred = classifier_dTree.predict(x_test.loc[:,param])
            model_summary(y_test,y_pred)

    if(type=="Random Forest"):
        st.write("Random Forest Classifier")      
        param = st.multiselect('Select features',['city', 'establishment', 'average_cost_for_two','votes','photo_count'],['city'])
        n_estimators_RF = st.slider('N Estimators', 2, 100, 10)

        if(st.button('Model predictions & results, Confusion Matrix')):
            classifier_RF = RandomForestClassifier(criterion='entropy', random_state=0,n_estimators=n_estimators_RF)
            classifier_RF.fit(x_train.loc[:,param], y_train)
            y_pred = classifier_RF.predict(x_test.loc[:,param])
            model_summary(y_test,y_pred)

    if(type=="SVM"):
        st.write("Support Vector Machine")

        st.write("Support Vector Machine (SVM) are supervised machine learning models than analyze data for classification & regression analysis. ")
        param = st.multiselect('Select features',['city', 'establishment', 'average_cost_for_two','votes','photo_count'],['city'])
        c_value = st.slider('C', 1, 100, 10)

        if(st.button('Model predictions & results, Confusion Matrix')):
            #knn = KNeighborsClassifier(n_neighbors = neighbor_count,p=2)
            classifier_svm=SVC(kernel='linear', C=c_value)
            classifier_svm.fit(x_train.loc[:,param], y_train)
            y_pred = classifier_svm.predict(x_test.loc[:,param])
            model_summary(y_test,y_pred)


x = df_zomato_establishment_subset.iloc[:,[2,3,5,10,11]]
y = df_zomato_establishment_subset.iloc[:,13]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=0)

def model_summary_regression(y_test,y_pred):
    accuracy = r2_score(y_test,y_pred)
    st.write("Accuracy")
    st.write(accuracy)

if (selected_type == "ML_Regression_Modeling"):
    st.write("Trying some Regression models to predict 'aggregate_rating'")

    st.write("'aggregate_rating' parameter is a continuous variable and is provided with the dataset.")
    st.write("Some of the key parameters influencing aggregate_rating are 'votes', 'city', 'establishment type' and 'average_cost_for_two'")
    st.write("For Regression modeling, KNN, DecisionTree, Randon Forest & linear regression models are used to verify the aggregate_rating prediction accuracy")
    st.write("Users are given options to play around with 1 key hyper-parameter per Model")
    st.write("Accuracy is used as the metrics in measuring a regression model's accuracy")
    st.write("As per the observations, Linear regression model has the least accuracy")
    st.write("Random Forest & Decision tree regressors have the highest accuracy and the value is around ~89%")

    type = st.radio(
        "Select type of algorithm",
        ("Linear Regression", "KNN", "Decision Tree","Random Forest")
    )

    param = st.multiselect('Select features',['city', 'establishment', 'average_cost_for_two','votes','photo_count'],['votes'])

    if(type=="Linear Regression"):
        st.write("Linear Regression")
        if(st.button('Model predictions & accuracy')):
            reg=LinearRegression()
            reg.fit(x_train,y_train)
            y_pred=reg.predict(x_test)
            model_summary_regression(y_test,y_pred)

    if(type=="KNN"):
        st.write("KNN Regressor")
        neighbor_count = st.slider('Number of Neighbors', 2, 10, 5)

        if(st.button('Model predictions & accuracy')):
            knn = KNeighborsRegressor(n_neighbors = neighbor_count,p=2)
            knn.fit(x_train.loc[:,param], y_train)
            y_pred = knn.predict(x_test.loc[:,param])
            model_summary_regression(y_test,y_pred)

    if(type=="Decision Tree"):
        st.write("Decision Tree Regressor")          
        max_depth_DT = st.slider('Max Depth', 2, 20, 10)

        if(st.button('Model predictions & accuracy')):
            dTree = DecisionTreeRegressor(criterion='squared_error', random_state=0,max_depth=max_depth_DT)
            dTree.fit(x_train.loc[:,param], y_train)
            y_pred = dTree.predict(x_test.loc[:,param])
            model_summary_regression(y_test,y_pred)

    if(type=="Random Forest"):
        st.write("Random Forest Regressor")      
        n_estimators_RF = st.slider('N Estimators', 2, 100, 10)

        if(st.button('Model predictions & accuracy')):
            regressor_RF = RandomForestRegressor(criterion='squared_error', random_state=0,n_estimators=n_estimators_RF)
            regressor_RF.fit(x_train.loc[:,param], y_train)
            y_pred = regressor_RF.predict(x_test.loc[:,param])
            model_summary_regression(y_test,y_pred)

x = df_zomato_establishment_subset.iloc[:,[2,3,6,13]]
y = df_zomato_establishment_subset.iloc[:,5]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=0)   

if (selected_type == "ML_User_Predictions"):
    st.title("Predict Average Cost for Two")

    st.write("'average_cost_for_two' parameter is a continuous variable and is provided with the dataset. This variable contains the average cost for 2 persons, per order")
    st.write("Some of the key parameters influencing aggregate_rating are 'votes', 'city', 'establishment type' and 'rating'")
    st.write("Regression models - KNN, Random Forest & Decision Tree - are used to generate expected 'average_cost_for_two")
    st.write("Users are given drop-down options to provide inputs for parameters like rating, city, establishment etc")

    mod = st.selectbox('Select the model for prediction:.', ['KNN', 'Decision Tree', 'Random Forest'])

    city_name = st.selectbox('Select City',["Mumbai","Chennai:","New Delhi","Kolkata","Hyderabad","Bangalore"])

    if (city_name == 'Mumbai'):
        city = 56
    elif (city_name == 'Bangalore'):
        city = 8
    elif (city_name == 'Chennai'):
        city = 12
    elif (city_name == 'New Delhi'):
        city = 66    
    elif (city_name == 'Kolkata'):
        city = 46    
    elif (city_name == 'Hyderabad'):
        city = 31

    price_range = st.selectbox('Select price_range',[1,2,3,4])

    rating = st.selectbox('Select Rating',[0,2,3,4,5])

    #{'Bakery': 0, 'Beverage Shop': 1, 'Café': 2, 'Casual Dining': 3, 'Dessert Parlour': 4, 'Quick Bites': 5, 'Sweet Shop': 6}    
    establishment_name = st.selectbox('Select type of establishment',['Bakery','Beverage Shop','Café','Casual Dining','Dessert Parlour','Quick Bites', 'Sweet Shop'])

    if (establishment_name == 'Bakery'):
        establishment = 0
    elif (establishment_name == 'Beverage Shop'):
        establishment = 1
    elif (establishment_name == 'Café'):
        establishment = 2
    elif (establishment_name == 'Casual Dining'):
        establishment = 3
    elif (establishment_name == 'Dessert Parlour'):
        establishment = 4
    elif (establishment_name == 'Quick Bites'):
        establishment = 5
    elif (establishment_name == 'Sweet Shop'):
        establishment = 6

    pred = st.button("Predict")
    if pred:

        if mod == 'KNN':
            knn = KNeighborsRegressor(n_neighbors = 5,p=2)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)

            st.write("Average cost predicted is: ")
            st.write(knn.predict([[establishment,city, price_range,rating]]))

        if mod == 'Random Forest':
            regressor_RF = RandomForestRegressor(criterion='squared_error', random_state=0,n_estimators=n_estimators_RF)
            regressor_RF.fit(x_train, y_train)
            y_pred = regressor_RF.predict(x_test)

            st.write("Average cost predicted is: ")
            st.write(regressor_RF.predict([[establishment,city, price_range,rating]]))

        if mod == 'Decision Tree':
            dTree = DecisionTreeRegressor(criterion='squared_error', random_state=0,max_depth=10)
            dTree.fit(x_train, y_train)
            y_pred = dTree.predict(x_test)

            st.write("Average cost predicted is: ")
            st.write(dTree.predict([[establishment,city, price_range,rating]]))

if (selected_type == "ML Modeling & Prediction Summary"):
    img = Image.open("zomato_logo.jpg")
    st.image(img)

    st.write("ML Modeling & Predictions")
    st.write("Online ordering is a major phenomenon currently and has lot more scope for future. We just did a high-level usage of the ML models (Classification/Regression) in the final project,")
    st.write("There is a lot more scope for improvement, especially in terms of scaling, feature engineering and hyperparameter tuning or choosing the right ML models as per the requirements")
    st.write("Another key improvement will be in terms of coming up with more metrics to evaluate the efficiency of the ML algorithms")

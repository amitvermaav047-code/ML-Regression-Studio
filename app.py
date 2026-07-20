import numpy as np 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error , root_mean_squared_error
#--------------- PAGE CONFIG--------------------
st.set_page_config(
    page_title="ML Regression Studio",
    page_icon="🤖",
    layout="wide"
)
#--------------- Main Title------------------
st.title("🤖 Ml Regression Studio")
st.markdown("""
Welcome to **ML Regression Studio**.

Upload your dataset, clean the data, train regression models,
and evaluate results — all without writing any code.
""")
#--------------- Sidebar-----------------
st.sidebar.title("Navigation🧭")
st.sidebar.info("""
This Application allow you to:
                
✔ Upload Dataset

✔ Clean Data

✔ Train Regression Models

✔ Evaluate Performance

✔ Download Predictions
""")

#--------------------tabs------------------
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "Datasets 📁",
    "Data Cleaning 🧹",
    "Feature Selection 🎯",
    "Model Training 🤖",
    "Result result 📈",
    "Download 📥"
])

#-----------------------TAB-1---------------------------
with tab1:
    file = st.file_uploader("Upload your CSV file",type=["csv"])
    if file is not None:
        if(
            "file_name" not in st.session_state
            or st.session_state.file_name!=file.name
        ):
            file.seek(0)
            st.session_state.file_name=file.name
            st.session_state.df=pd.read_csv(file)
            st.session_state.original_df = st.session_state.df.copy()
        st.success("✅ CSV file uploaded successfully!")
        df = st.session_state.df
        st.header("Datasets 📁")
    #------------------- Preview Data----------------
        st.subheader("Data Preview 📰:")
        st.dataframe(df,use_container_width=True)
        row , col = df.shape
        col1 , col2 = st.columns(2)
        with col1:
            st.metric("Rows :",row)
            st.caption("Total Rows")
        with col2:
            st.metric("Columns :",col)
            st.caption("Total Columns")

#-------------------- Data Overview -----------------
        with st.expander("Dataset Information"):
            if st.checkbox("SHOW COLUMNS NAME"):
                st.write(df.columns)
            if st.checkbox("SHOW DATA TYPE"):
                st.write(df.dtypes)
            st.caption("Data schema and columns information")

    #---------------------Health Report-----------------
        num_col = len(df.select_dtypes(include="number").columns)
        str_col = len(df.select_dtypes(include="object").columns)
        missing = df.isnull().sum().sum()
        duplicate = df.duplicated().sum().sum()
        st.divider()
        st.subheader("Data Health Report 🩺")
        col1 , col2 , col3 , col4 = st.columns(4)
        with col1:
            st.metric("MISSING VALUES :",missing)
            st.caption("Total Missing values")
        with col2:
            st.metric("DUPLICATE VALUES :",duplicate)
            st.caption("Total Duplicated values")
        with col3:
            st.metric("NUMERIC COLUMNS :",num_col)
            st.caption("Total Numeric Columns")
        with col4:
            st.metric("STRING COLUMNS :",str_col)
            st.caption("Total String Columns")
        col1,col2 = st.columns(2)
        with col1:
            if missing==0:
                st.success("No Missing value found ✔")
            else:
                st.warning(f"{missing} Missing values found ⚠")
        with col2:
            if duplicate==0:
                st.success("No Duplicated value found ✔")
            else:
                st.warning(f"{duplicate} Missing values found ⚠")
        health = 100
        if missing>0:
            health-=30
        if duplicate>0:
            health-=30
        st.progress(health)
        st.caption(f'Dataset Health is {health}%')
    else:
        st.info("Please upload the CSV file")

#-----------------------TAB-2---------------------------
with tab2:

    if "df" not in st.session_state:
        st.warning("Please upload dataset first.")
        st.stop()
    df=st.session_state.df
    st.header("Data Cleaning 🧹")
    st.subheader("Missing Values")
    if "missing_cleaned" in st.session_state:
        st.success("Dataset Cleaned Successfully!")
        del st.session_state.missing_cleaned
    elif missing==0:
        st.success("No Missing value is Present")
    else:
        method = st.selectbox(
            "Select Method",
            ["~~Selct an Method~~","Drop Missing Values","Fill Missing Values"]
        )
        if method=="Drop Missing Values":
            if st.button("Drop Values"):
                st.session_state.df=df.dropna()
                st.session_state.missing_cleaned =True
                st.rerun()
        num_col=df.select_dtypes(include="number").columns
        if method=="Fill Missing Values":
            opt = st.selectbox(
                "Select Option of fill",
                ["~~Select an Option~~","Mean","Mode","Median"]
            )
            if opt!="~~Select an Option~~":
                if st.button("Apply Cleaning"):
                    if opt=="Mean":
                        for col in num_col:
                            df[col]=df[col].fillna(df[col].mean())
                    elif opt=="Mode":
                        for col in df.columns:
                            df[col]=df[col].fillna(df[col].mode()[0])
                    elif opt=="Median":
                        for col in num_col:
                            df[col]=df[col].fillna(df[col].median())
                    st.session_state.df=df
                    st.session_state.missing_cleaned =True
                    st.rerun()
    st.divider()
    st.subheader("Remove Duplicate")
    if "R_duplicate" in st.session_state:
        st.success("Duplicate Remove Successfully !")
        del st.session_state.R_duplicate
    elif duplicate == 0:
        st.success("No Duplicate is present !")

       
    else:
        if st.button("Drop Duplicate 🗑"):
            st.session_state.df=df.drop_duplicates()
            st.session_state.R_duplicate=True
            st.rerun()
    st.divider()
    cat_col = df.select_dtypes(include="object").columns
    if len(cat_col) == 0 :
        st.info("No Categorical Columns available")
    else:
        st.subheader("Categorical Encoding")
        method = st.selectbox(
            "Select Encoding Method",
            ["~~Select an option~~","Label Encoding","One Hot Encoding"]
        )
        if st.button("Apply Encoding"):
            if method!="~~Select an option~~":
                if method=="Label Encoding":
                    le = LabelEncoder()
                    for col in cat_col:
                        df[col] = le.fit_transform(df[col])
                    st.session_state.df = df
                    st.success("Label Encoding Applied Successfully")
                    st.rerun()
                elif method == "One Hot Encoding":
                    df = pd.get_dummies(
                        df,
                        columns=cat_col,
                        drop_first=True,
                        dtype=int
                    )
                    st.session_state.df = df
                    st.success("One Hot Encoding Applied Successfully!")
                    st.rerun()
    st.divider()
    st.subheader("Reset Data")
    if st.button("Reset Data",use_container_width=True):
        st.session_state.df = st.session_state.original_df.copy()
        st.success("Reset data Successfully")
        st.rerun()

       
#---------------------------TAB-3----------------------------------
with tab3:
    if "df" not in st.session_state:
        st.warning("Please upload dataset first.")
        st.stop()
    df = st.session_state.df
    numeric_col = df.select_dtypes(include="number").columns.tolist()
    st.header("Feature Selection 🎯")
    st.subheader("Select Model")
    model = st.selectbox(
        "Select Regression Model",
        ["~~Select an Model~~","Linear Regression","Multiple Linear Regression","polynomial Regression"]
    )
    if model=="Linear Regression":
        Ind = st.selectbox(
            "Select Independent Column",
            ["~~Select an Column~~"] + numeric_col
        )
        dep = st.selectbox(
            "Select Dependent Column",
            ["~~Select an Column~~"] + numeric_col
        )
    elif model=="Multiple Linear Regression":
        Ind = st.multiselect(
            "Select Independent Columns",
            numeric_col
        )
        dep = st.selectbox(
            "Select Dependent Column",
            ["~~Select an Column~~"] + numeric_col
        )
    elif model=="polynomial Regression":
        Ind = st.selectbox(
            "Select Independent Column",
            ["~~Select an Column~~"] + numeric_col
        )
        dep = st.selectbox(
            "Select Dependent Column",
            ["~~Select an Column~~"] + numeric_col
        )
        degree = st.slider(
            "Select Polynomial Degree",
            min_value=2,
            max_value=6,
            value=4
        )
    if model!="~~Select an Model~~":
        st.session_state.model=model
        if model=="Multiple Linear Regression":
            if len(Ind)>0:
                st.session_state.ind=Ind
        else:
            if Ind!="~~Select an Column~~":
                st.session_state.ind=Ind
        if dep!="~~Select an Column~~":
            st.session_state.dep=dep
        if model=="polynomial Regression":
            st.session_state.degree=degree

#------------------------------TAB-4----------------------------
with tab4:
    if "df" not in st.session_state:
        st.warning("Please upload dataset first")
        st.stop()
    if "model" not in st.session_state:
        st.warning("Please select Model first")
        st.stop()
    if "ind" not in st.session_state or "dep" not in st.session_state:
        st.warning("Please select Independent and Dependent columns first.")
        st.stop()
    df = st.session_state.df
    selected_model = st.session_state.model
    Ind = st.session_state.ind
    Dep = st.session_state.dep
    if selected_model=="polynomial Regression":
        degree = st.session_state.degree

    st.header("Model Training 🤖")
    st.subheader("Selected Configuration")
    col1 , col2 = st.columns(2)
    with col1:
        st.metric("Selecte model :",selected_model)
    with col2:
        st.metric("Train Test Split" , "80 : 20")
    col1 , col2 = st.columns(2)
    with col1:
        st.info(f"**INDEPENDENT :** {Ind}")
    with col2:
        st.info(f"**DEPENDENT :** {Dep}")
    if selected_model == "polynomial Regression":
        st.metric("Polynomial Degree", st.session_state.degree)
    st.divider()
    
    if df.isnull().sum().sum()>0:
        st.error("Please clean data first")
        st.stop()
    if st.button("Train Model",use_container_width=True):
        if selected_model == "Linear Regression" or selected_model =="polynomial Regression":    
            if Ind == Dep:
                st.error("Independent and Dependent columns cannot be same.")
                st.stop()
        elif selected_model=="Multiple Linear Regression":
            if Dep in Ind:
                st.error("Dependent column can't be selected as Independent")
                st.stop()
        with st.spinner("Trainning Model..."):
            if selected_model=="Linear Regression":
                X = df[[Ind]]
                Y = df[Dep]
                x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
                lr = LinearRegression()
                lr.fit(x_train,y_train)
                y_pred = lr.predict(x_test)
            elif selected_model=="Multiple Linear Regression":
                X = df[Ind]
                Y = df[Dep]
                x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
                lr = LinearRegression()
                lr.fit(x_train,y_train)
                y_pred = lr.predict(x_test)
            elif selected_model=="polynomial Regression":
                X = df[[Ind]]
                Y = df[Dep]  
                x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
                poly = PolynomialFeatures(degree=degree)
                x_train_poly = poly.fit_transform(x_train)
                x_test_poly = poly.transform(x_test)  
                lr = LinearRegression()
                lr.fit(x_train_poly,y_train)
                y_pred = lr.predict(x_test_poly)
            st.session_state.y_test=y_test
            st.session_state.y_pred=y_pred
            st.session_state.lr=lr
            st.session_state.model_trained=True
        st.success("Model Trained Successfully.")
        # st.balloons()


#----------------------------TAB-5-------------------------------
with tab5:
    st.header("Result 📈")
    if "model_trained" not in st.session_state:
        st.warning("Pleas train model first")
        st.stop()
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    col1 , col2 , col3 = st.columns(3)
    with col1:
        st.metric("R² Score is :",r2)
        st.caption("Predicted R² Score")
    with col2:
        st.metric("Average error :",mae)
        st.caption("Mean Absolute Error")
    with col3:
        st.metric("Squared error :",mse)
        st.caption("Mean Squared Error")
    st.subheader("Actual VS Predicted")
    result = pd.DataFrame({
        "Actual":y_test,
        "Predicted":y_pred
    })
    st.dataframe(result,use_container_width=True)
    st.session_state.result = result
    st.subheader("Actual VS Predict Graph")
    fig , ax = plt.subplots(figsize=(10,5))
    ax.scatter(y_test,y_pred,label="Prediction")
    ax.plot(
        [y_test.min(),y_test.max()],
        [y_test.min(),y_test.max()],
        "r--",
        linewidth=2
        )
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
#-----------------------------TAB-6-------------------------------
with tab6:
    st.header("Download 📥")
    st.subheader("Download Prediction")
    if "model_trained" not in st.session_state:
        st.warning("Please train Model first")
        st.stop()
    else:
        result = st.session_state.result
        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prediction CSV",
            data=csv,
            file_name="Prediction-Result.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Download the prediction results as a CSV file.")
#------------------ Developer ------------------        
    st.divider()
    st.caption("Developed by Amit Verma | Powered by Streamlit & Scikit-learn 🤖")
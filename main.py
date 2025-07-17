# שמור את הקובץ הזה כ: streamlit_app.py
# הרץ עם: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report, ConfusionMatrixDisplay
import pickle
import os
import kagglehub

# Download latest version
path = kagglehub.dataset_download("fmendes/fmendesdat263xdemos")

print("Path to dataset files:", path)
# הגדרות עיצוב
st.set_page_config(
    page_title="ניתוח נתוני אימון גופני",
    page_icon="💪",
    layout="wide"
)


# טעינת נתונים
@st.cache_data
def load_data():
    df_1 = pd.read_csv(path+"/exercise.csv")
    df_2 = pd.read_csv(path+"/calories.csv")
    df = pd.merge(df_1, df_2, on='User_ID')

    # הוספת רמות אימון
    def get_exercise_level(calories):
        if calories <= 79:
            return 0
        elif calories <= 138.:
            return 1
        else:
            return 2

    df['ExerciseLevel'] = df['Calories'].apply(get_exercise_level)
    df['ExerciseLevelName'] = df['ExerciseLevel'].map({0: 'נמוך', 1: 'בינוני', 2: 'גבוה'})
    # df['Gender'] = df['Gender'].map({'female': 'נקבה', 'male': 'זכר'})

    return df


# טעינת מודלים
@st.cache_resource
def train_models(df):
    # מודל רגרסיה
    X_reg = df[['Age', 'Duration', 'Heart_Rate', 'Body_Temp']].values
    y_reg = df['Calories'].values

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)

    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_reg_scaled, y_train_reg)

    # KNN Regression
    knn_reg = KNeighborsRegressor(n_neighbors=35)
    knn_reg.fit(X_train_reg_scaled, y_train_reg)

    # מודל סיווג
    X_clf = df[['Age', 'Duration', 'Heart_Rate', 'Body_Temp']].values
    y_clf = df['ExerciseLevel'].values

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    scaler_clf = StandardScaler()
    X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler_clf.transform(X_test_clf)

    knn_clf = KNeighborsClassifier(n_neighbors=25)
    knn_clf.fit(X_train_clf_scaled, y_train_clf)

    return {
        'lin_reg': lin_reg,
        'knn_reg': knn_reg,
        'knn_clf': knn_clf,
        'scaler_reg': scaler_reg,
        'scaler_clf': scaler_clf,
        'X_test_reg': X_test_reg_scaled,
        'y_test_reg': y_test_reg,
        'X_test_clf': X_test_clf_scaled,
        'y_test_clf': y_test_clf
    }


# טעינת נתונים
df = load_data()
models = train_models(df)

# כותרת ראשית
st.title("📊 פרויקט ניתוח נתוני אימון גופני")
st.markdown("---")

# תפריט צד
st.sidebar.title("ניווט")
page = st.sidebar.radio("בחר עמוד:", ["דף הבית", "ניתוח נתונים", "מודלים", "חיזוי"])

if page == "דף הבית":
    st.header("ברוכים הבאים לפרויקט!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### אודות הפרויקט
        פרויקט זה מנתח נתוני אימון גופני כדי להבין את הקשר בין:
        - **קצב לב** לבין **שריפת קלוריות**
        - **מאפיינים אישיים** לבין רמת האימון

        ### הדאטה
        הדאטה מכיל מידע על:
        - גיל, מין, גובה, משקל
        - משך אימון, קצב לב, טמפרטורת גוף
        - קלוריות שנשרפו
        """)

    with col2:
        st.metric("סה״כ נתונים", f"{len(df):,} רשומות")
        st.metric("עמודות", len(df.columns))
        st.metric("מספר יעדים (רגרסיה)", "1 (קלוריות)")
        st.metric("מספר יעדים (סיווג)", "3 (רמות אימון)")

elif page == "ניתוח נתונים":
    st.header("🔍 ניתוח נתונים")

    tab1, tab2, tab3, tab4 = st.tabs(["תיאור כללי", "יחסים", "קורלציות", "תובנות"])

    with tab1:
        st.subheader("תיאור כללי של הנתונים")
        st.dataframe(df[['Age', 'Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories',
                         'ExerciseLevelName']])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("קלוריות ממוצעות", f"{df['Calories'].mean():.1f}")
            st.metric("קצב לב ממוצע", f"{df['Heart_Rate'].mean():.1f}")
        with col2:
            st.metric("משך אימון ממוצע", f"{df['Duration'].mean():.1f} דקות")
            st.metric("גיל ממוצע", f"{df['Age'].mean():.1f} שנים")

    with tab2:
        st.subheader("קשרים בין משתנים")

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='Heart_Rate', y='Calories', hue='ExerciseLevel', palette='tab10', ax=ax1)
        ax1.set_title('Heart Rate vs Calories by Exercise Level')
        ax1.set_xlabel('Heart Rate (bpm)')
        ax1.set_ylabel('Calories Burned')
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='Gender', y='Calories', hue='Gender',ax=ax2)
        ax2.set_title('Calories Burned by Gender')
        ax2.set_xlabel('Gender')
        ax2.set_ylabel('Calories Burned')
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='Height', y='Weight', hue='Gender', palette='tab10', ax=ax3)
        ax3.set_title('Height vs Weight by Gender')
        ax3.set_xlabel('Height (cm)')
        ax3.set_ylabel('Weight (kg)')
        st.pyplot(fig3)
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='ExerciseLevel', y='Weight', hue='ExerciseLevel', palette='tab10', ax=ax4)
        plt.xlabel('Exercise Level')
        plt.ylabel('Weight')
        plt.title('Exercise Level vs Weight')
        st.pyplot(fig4)

        fig5, ax5 = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=df,
            x='Gender',
            y='Heart_Rate',
            hue='Gender',
            palette='tab10',
            ax=ax5
        )
        ax5.set_xlabel('Gender')
        ax5.set_ylabel('Heart Rate')
        ax5.set_title('Gender vs Heart Rate')
        st.pyplot(fig5)
    
    with tab3:
        st.subheader("Correlation Matrix")

        numeric_cols = df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='vlag',
                    fmt='.2f', center=0, ax=ax)
        ax.set_title('Correlation Matrix of Numeric Variables')
        st.pyplot(fig)
    with tab4:
        st.subheader("תובנות מרכזיות")

        st.markdown("""
        ### 💡 התגליות שלנו

        1. **קצב הלב משפיע על שריפת קלוריות** - יש קורלציה חזקה בין קצב הלב לבין כמות הקלוריות שנשרפו

        2. **הבדלי מין** - גברים נוטים לשרוף יותר קלוריות מאשר נשים באימון זהה

        3. **מבנה גוף** - גברים בדרך כלל גבוהים וכבדים יותר

        4. **מוטיבציה** - אנשים עם משקל גבוה נוטים לעבוד קשה יותר

        5. **קצב לב דומה** - למרות ההבדלים, קצב הלב הממוצע דומה בין המינים
        """)

elif page == "מודלים":
    st.header("🤖 ביצועי המודלים")

    tab1, tab2 = st.tabs(["רגרסיה", "סיווג"])

    with tab1:
        st.subheader("מודל רגרסיה - חיזוי קלוריות")

        col1, col2 = st.columns(2)

        # הערכה על סט הבדיקה
        y_pred_lin = models['lin_reg'].predict(models['X_test_reg'])
        y_pred_knn = models['knn_reg'].predict(models['X_test_reg'])

        with col1:
            st.metric("R² לינארי", f"{r2_score(models['y_test_reg'], y_pred_lin):.3f}")
            st.metric("R² KNN", f"{r2_score(models['y_test_reg'], y_pred_knn):.3f}")

        with col2:
            st.success("The KNN model outperforms the linear regression model by 2% in R² score.")

        # גרף השוואה
        fig, ax = plt.subplots(figsize=(10, 6))

        sample_size = min(100, len(models['y_test_reg']))
        indices = np.random.choice(len(models['y_test_reg']), sample_size, replace=False)

        x = np.arange(sample_size)
        width = 0.25

        ax.bar(x - width, models['y_test_reg'][indices], width, label='True value', alpha=0.8)
        ax.bar(x, y_pred_lin[indices], width, label='Linear Reg', alpha=0.8)
        ax.bar(x + width, y_pred_knn[indices], width, label='KNN', alpha=0.8)

        ax.set_xlabel('Sample index')
        ax.set_ylabel('Calories')
        ax.set_title('Predctions Comparison')
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.subheader("מודל סיווג - קביעת רמת אימון")

        y_pred_clf = models['knn_clf'].predict(models['X_test_clf'])
        accuracy = accuracy_score(models['y_test_clf'], y_pred_clf)

        st.metric("דיוק הסיווג", f"{accuracy:.2%}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            models['y_test_clf'], y_pred_clf,
            display_labels=['Low', 'Meduim', 'High'],
            cmap="Blues", ax=ax
        )
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.text("דוח סיווג:")
        st.text(classification_report(
            models['y_test_clf'], y_pred_clf,
            target_names=['נמוך', 'בינוני', 'גבוה']
        ))

elif page == "חיזוי":
    st.header("🔮 חיזוי אישי")

    st.markdown("הכנס את הפרטים שלך וקבל חיזוי מותאם אישית:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("גיל", min_value=10, max_value=80, value=30)
        duration = st.number_input("משך אימון (דקות)", min_value=1, max_value=120, value=30)

    with col2:
        heart_rate = st.number_input("קצב לב", min_value=50, max_value=200, value=120)
        body_temp = st.number_input("טמפרטורת גוף", min_value=36.0, max_value=43.0, value=37.5)

    gender = st.selectbox("מין", ["זכר", "נקבה"])
    height = st.number_input("גובה (ס״מ)", min_value=140, max_value=220, value=170)
    weight = st.number_input("משקל (ק״ג)", min_value=40, max_value=150, value=70)

    if st.button("חזה עכשיו!"):
        # חיזוי קלוריות
        features_reg = np.array([[age, duration, heart_rate, body_temp]])
        features_reg_scaled = models['scaler_reg'].transform(features_reg)

        calories_pred_lin = models['lin_reg'].predict(features_reg_scaled)[0]
        calories_pred_knn = models['knn_reg'].predict(features_reg_scaled)[0]

        # חיזוי רמת אימון
        features_clf = np.array([[age, duration, heart_rate, body_temp]])
        features_clf_scaled = models['scaler_clf'].transform(features_clf)

        exercise_level = models['knn_clf'].predict(features_clf_scaled)[0]
        level_names = {0: 'נמוך', 1: 'בינוני', 2: 'גבוה'}

        # הצגת תוצאות
        st.markdown("### 📊 התוצאות שלך:")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("קלוריות (רגרסיה לינארית)", f"{calories_pred_lin:.1f}")
            st.metric("קלוריות (KNN)", f"{calories_pred_knn:.1f}")

        with col2:
            st.metric("רמת אימון", level_names[exercise_level])

        # המלצות
        st.markdown("### 💡 המלצות:")

        if calories_pred_knn < 80:
            st.info("התאמנת בעצימות נמוכה. נסה להעלות את קצב הלב או להאריך את האימון.")
        elif calories_pred_knn < 138:
            st.success("התאמנת בעצימות בינונית - מצוין! המשך כך.")
        else:
            st.success("התאמנת בעצימות גבוהה! היזהר לא להתאמץ יותר מדי.")

        # BMI
        bmi = weight / ((height / 100) ** 2)
        st.metric("BMI", f"{bmi:.1f}")

        if bmi < 18.5:
            st.warning("BMI נמוך - ייתכן שתזדקק להגדלת צריכת קלוריות")
        elif bmi > 25:
            st.info("BMI גבוה - האימונים שלך עוזרים בניהול משקל")
        else:
            st.success("BMI תקין - המשך בשגרת האימונים הנוכחית")

# footer
st.markdown("---")
st.markdown("פותח על ידי שאולי ינון סולטן | פרויקט סיום שבוע 1 במחנה מדעי הנתונים 2025 טכניון")
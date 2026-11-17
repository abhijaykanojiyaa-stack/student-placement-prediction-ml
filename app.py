import streamlit as st
import pandas as pd
import joblib
from pandas.api.types import is_numeric_dtype, is_bool_dtype


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# 2. LOAD DATASET AND MODEL
# ============================================================

DATA_PATH = "data/student_placement_prediction_dataset_2026.csv"
MODEL_PATH = "placement_model.pkl"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)


# ============================================================
# 3. IDENTIFY MODEL FEATURES
# ============================================================

TARGET = "placement_status"

excluded_columns = {
    TARGET,
    "salary_package_lpa"
}

# Remove student ID
for col in df.columns:
    if "student" in col.lower() and "id" in col.lower():
        excluded_columns.add(col)

# Get features used during training
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
else:
    model_features = [
        col for col in df.columns
        if col not in excluded_columns
    ]

feature_columns = [
    col for col in model_features
    if col in df.columns
    and col not in excluded_columns
]


# ============================================================
# 4. CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9aa4b2;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🎓")
    st.markdown("## About Model")

    st.info(
        """
        **Student Placement Prediction**

        Machine Learning based prediction
        of student placement status.

        **Model:** Logistic Regression

        **Deployment:** Streamlit
        """
    )

    st.markdown("### Dataset Information")

    st.write(f"**Students:** {len(df):,}")
    st.write(f"**Features:** {len(feature_columns)}")
    st.write("**Target:** Placement Status")

    st.markdown("---")

    st.caption(
        "Academic machine-learning project demonstration."
    )


# ============================================================
# 6. MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Student Placement Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict student placement status using machine learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 7. DATASET SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Students", f"{len(df):,}")

with col2:
    st.metric("Input Features", len(feature_columns))

with col3:
    st.metric("Prediction Target", "Placement")

st.markdown("---")


# ============================================================
# 8. STUDENT INPUT SECTION
# ============================================================

st.markdown("## 👨‍🎓 Student Information")

st.write(
    "Enter the student's details below and click Predict Placement."
)


# ------------------------------------------------------------
# STUDENT NAME
# ------------------------------------------------------------

student_name = st.text_input(
    "Student Name",
    placeholder="Enter student name"
)


# ------------------------------------------------------------
# MODEL INPUTS
# ------------------------------------------------------------

user_input = {}

input_columns = st.columns(3)


for i, column in enumerate(feature_columns):

    with input_columns[i % 3]:

        display_name = column.replace("_", " ").title()

        # CATEGORICAL FEATURE
        if (
            not is_numeric_dtype(df[column])
            and not is_bool_dtype(df[column])
        ):

            values = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            values = sorted(values)

            if values:

                user_input[column] = st.selectbox(
                    display_name,
                    values,
                    key=f"input_{column}"
                )

        # BOOLEAN FEATURE
        elif is_bool_dtype(df[column]):

            user_input[column] = st.selectbox(
                display_name,
                [True, False],
                key=f"input_{column}"
            )

        # NUMERICAL FEATURE
        elif is_numeric_dtype(df[column]):

            minimum = float(df[column].min())
            maximum = float(df[column].max())
            default = float(df[column].median())

            # INTEGER
            if pd.api.types.is_integer_dtype(df[column]):

                user_input[column] = st.number_input(
                    display_name,
                    min_value=int(minimum),
                    max_value=int(maximum),
                    value=int(default),
                    step=1,
                    key=f"input_{column}"
                )

            # DECIMAL
            else:

                user_input[column] = st.number_input(
                    display_name,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=0.01,
                    key=f"input_{column}"
                )


# ============================================================
# 9. PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict = st.button(
    "🔮 Predict Placement",
    use_container_width=True,
    type="primary"
)


# ============================================================
# 10. PREDICTION RESULT
# ============================================================

if predict:

    try:

        # Create input for model
        input_df = pd.DataFrame([user_input])

        # Same column order as training
        input_df = input_df[feature_columns]

        # Prediction
        prediction = model.predict(input_df)[0]

        # Probability
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        prediction_text = str(prediction).strip().lower()


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction_text in [
            "1",
            "placed",
            "yes",
            "true"
        ]:

            result_text = "✅ PLACED"

            result_message = (
                "The model predicts that the student "
                "is likely to be placed."
            )

        else:

            result_text = "⚠️ NOT PLACED"

            result_message = (
                "The model predicts that the student "
                "is likely to be not placed."
            )


        # ----------------------------------------------------
        # FIND PLACED PROBABILITY
        # ----------------------------------------------------

        placed_index = None

        for index, value in enumerate(classes):

            if str(value).strip().lower() in [
                "1",
                "placed",
                "yes",
                "true"
            ]:

                placed_index = index
                break


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.markdown("---")
        st.markdown("## 📊 Prediction Result")

        if student_name.strip():
            st.write(f"**Student Name:** {student_name}")

        st.success(result_text)

        st.info(result_message)


        # ----------------------------------------------------
        # DISPLAY PROBABILITY
        # ----------------------------------------------------

        if placed_index is not None:

            placement_probability = (
                probabilities[placed_index] * 100
            )

            st.metric(
                "Placement Probability",
                f"{placement_probability:.2f}%"
            )

            st.progress(
                float(placement_probability / 100)
            )


    except Exception as e:

        st.error("Prediction error")
        st.code(str(e))


# ============================================================
# 11. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Student Placement ML • Machine Learning Deployment Project"
)
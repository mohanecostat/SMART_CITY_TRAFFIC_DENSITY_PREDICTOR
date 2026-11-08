import datetime
from pathlib import Path
 
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
 
 
#CONFIGURATION
 
st.set_page_config(
    page_title="Smart City Traffic Predictor",
    page_icon="🚦",
    layout="centered",
)
 
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Smart city traffic.csv"
HISTORY_FILE = BASE_DIR / "prediction_history.csv"
 
FEATURES = ["area", "hour", "day"]
TARGET = "congestion_level"
 
st.title("🚦 Smart City Traffic Density Predictor")
st.write(
    "Predict traffic congestion for a selected area, date and time, "
    "and receive suggestions for a less congested travel time."
)
 
 
#LOAD AND PREPARE DATA
 
@st.cache_data
def load_dataset(csv_path, file_version=None):
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place 'Smart city traffic.csv' beside this Python file."
        )
 
    data = pd.read_csv(csv_path)
 
    required = {"date", "time", "day", "area", TARGET}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(sorted(missing))
        )
 
    data = data.copy()
 
    data["parsed_date"] = pd.to_datetime(data["date"], errors="coerce")
    data["parsed_time"] = pd.to_datetime(
        data["time"].astype(str), format="%H:%M:%S", errors="coerce"
    )
 
    data["hour"] = data["parsed_time"].dt.hour
    data["area"] = data["area"].astype(str).str.strip()
    data["day"] = data["day"].astype(str).str.strip()
    data[TARGET] = data[TARGET].astype(str).str.strip()
 
    data = data.dropna(subset=FEATURES + [TARGET])
    data = data[data["area"] != ""]
    data = data[data["day"] != ""]
    data = data[data[TARGET] != ""]
 
    if data.empty:
        raise ValueError("No usable records were found in the dataset.")
 
    return data
 
 
#TRAIN MODEL
 
@st.cache_resource
def train_model(data):
    X = data[FEATURES].copy()
    y = data[TARGET].copy()
 
    class_counts = y.value_counts()
    stratify_values = y if class_counts.min() >= 2 else None
 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_values,
    )
 
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["area", "day"],
            ),
            ("numeric", "passthrough", ["hour"]),
        ]
    )
 
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
 
    pipeline.fit(X_train, y_train)
    test_predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)
 
    return pipeline, accuracy, len(X_train), len(X_test)
 
 
#LOAD MODEL 
 
try:
    data = load_dataset(DATA_FILE, DATA_FILE.stat().st_mtime_ns)
    model, accuracy, train_rows, test_rows = train_model(data)
    areas_list = sorted(data["area"].unique().tolist())
    available_days = set(data["day"].unique())
except Exception as error:
    st.error(f"Could not load or train the model: {error}")
    st.stop()
 
 
#MODEL INFORMATION 
 
with st.expander("Machine Learning Model Information"):
    st.write("**Algorithm:** Random Forest Classifier")
    st.write(f"**Dataset records used:** {len(data):,}")
    st.write(f"**Training records:** {train_rows:,}")
    st.write(f"**Testing records:** {test_rows:,}")
    st.write(f"**Test accuracy:** {accuracy:.2%}")
    st.warning(
        "This accuracy is measured on a held-out portion of the supplied "
        "historical dataset. It does not guarantee real-world accuracy, "
        "especially for dates or conditions not represented in the data."
    )
    st.write("**Dataset preview**")
    st.dataframe(data.head(10), use_container_width=True)
 
 
#SESSION STATE
 
if "page" not in st.session_state:
    st.session_state.page = "input"
 
if "suggestion_index" not in st.session_state:
    st.session_state.suggestion_index = 0
 
 
#INPUT PAGE
 
if st.session_state.page == "input":
    st.subheader("Enter Journey Details")
 
    with st.form("traffic_prediction_form"):
        selected_area = st.selectbox("Select Area / Location", areas_list)
 
        today = datetime.date.today()
        selected_date = st.date_input(
            "Select Date",
            value=today,
            min_value=today,
        )
 
        selected_time = st.time_input(
            "Select Time",
            value=datetime.datetime.now().time().replace(second=0, microsecond=0),
        )
 
        submitted = st.form_submit_button(
            "Predict Traffic Density",
            type="primary",
            use_container_width=True,
        )
 
    if submitted:
        chosen_datetime = datetime.datetime.combine(
            selected_date, selected_time
        )
 
        if chosen_datetime < datetime.datetime.now():
            st.error("You cannot select a past date and time.")
        else:
            st.session_state.selected_area = selected_area
            st.session_state.selected_date = selected_date
            st.session_state.selected_time = selected_time
            st.session_state.hour = selected_time.hour
            st.session_state.day_name = selected_date.strftime("%A")
            st.session_state.suggestion_index = 0
            st.session_state.page = "result"
            st.rerun()
 
 
#RESULT PAGE
 
elif st.session_state.page == "result":
    area = st.session_state.selected_area
    selected_date = st.session_state.selected_date
    selected_time = st.session_state.selected_time
    current_hour = st.session_state.hour
    day_name = st.session_state.day_name
 
    st.subheader("📊 Traffic Prediction Results")
    st.write(f"**Area:** {area}")
    st.write(f"**Date:** {selected_date.strftime('%d-%m-%Y')}")
    st.write(f"**Day:** {day_name}")
    st.write(f"**Time:** {selected_time.strftime('%H:%M')}")
    st.divider()
 
    input_data = pd.DataFrame(
        [{"area": area, "hour": current_hour, "day": day_name}],
        columns=FEATURES,
    )
 
    try:
        predicted_level = str(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(max(probabilities))
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()
 
    st.subheader("Predicted Congestion Level")
 
    normalized_level = predicted_level.strip().lower()
    if normalized_level in {"high", "severe"}:
        st.error(f"🔴 Traffic Status: **{predicted_level.upper()}**")
    elif normalized_level in {"medium", "moderate"}:
        st.warning(f"🟡 Traffic Status: **{predicted_level.upper()}**")
    else:
        st.success(f"🟢 Traffic Status: **{predicted_level.upper()}**")
 
    st.progress(confidence, text=f"Model confidence: {confidence:.1%}")
    st.caption(
        "Model confidence is not the same as guaranteed prediction accuracy."
    )
 
    #SMART TRIP ADVICE
 
    st.divider()
    st.subheader("💡 Smart Trip Advice")
 
    rank_map = {
        "low": 1,
        "medium": 2,
        "moderate": 2,
        "high": 3,
        "severe": 4,
    }
    current_rank = rank_map.get(normalized_level, 2)
 
    if current_rank <= 2:
        st.success(
            f"Traffic is **{predicted_level}** at this time. "
            "This is a reasonable time for your trip."
        )
    else:
        better_predictions = []
 
        # Check up to two hours before and after the selected hour.
        for hour in range(
            max(0, current_hour - 2),
            min(24, current_hour + 3),
        ):
            if hour == current_hour:
                continue
 
            alternative = pd.DataFrame(
                [{"area": area, "hour": hour, "day": day_name}],
                columns=FEATURES,
            )
 
            alternative_level = str(model.predict(alternative)[0])
            alternative_rank = rank_map.get(
                alternative_level.strip().lower(), 2
            )
 
            if alternative_rank <= 2:
                better_predictions.append(
                    (hour, alternative_level, abs(hour - current_hour))
                )
 
        better_predictions.sort(key=lambda item: (item[2], item[0]))
 
        if better_predictions:
            index = (
                st.session_state.suggestion_index
                % len(better_predictions)
            )
            alternative_hour, alternative_level, difference = (
                better_predictions[index]
            )
 
            direction = (
                "prepone" if alternative_hour < current_hour else "postpone"
            )
 
            st.info(
                f"Consider **{direction}** your trip by **{difference} hour(s)** "
                f"to around **{alternative_hour:02d}:00**. "
                f"The model predicts **{alternative_level}** congestion then."
            )
 
            if st.button("🔄 Give Another Suggestion"):
                st.session_state.suggestion_index += 1
                st.rerun()
        else:
            st.warning(
                "No Low or Medium traffic time was predicted within "
                "two hours of your selected time."
            )
 
    #SAVE PREDICTION HISTORY
 
    history_record = pd.DataFrame(
        [{
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "area": area,
            "date": selected_date.strftime("%Y-%m-%d"),
            "day": day_name,
            "time": selected_time.strftime("%H:%M"),
            "predicted_congestion": predicted_level,
            "model_confidence": round(confidence, 4),
        }]
    )
 
    if "saved_result_key" not in st.session_state:
        st.session_state.saved_result_key = None
 
    result_key = (
        area,
        selected_date.isoformat(),
        selected_time.strftime("%H:%M"),
    )
 
    if st.session_state.saved_result_key != result_key:
        try:
            if HISTORY_FILE.exists():
                old_history = pd.read_csv(HISTORY_FILE)
                history = pd.concat(
                    [old_history, history_record], ignore_index=True
                )
            else:
                history = history_record
 
            history.to_csv(HISTORY_FILE, index=False)
            st.session_state.saved_result_key = result_key
            st.success("Prediction saved to history.")
        except Exception as error:
            st.warning(f"Could not save prediction history: {error}")
 
    #HISTORY
 
    st.divider()
    st.subheader("Prediction History")
 
    if HISTORY_FILE.exists():
        try:
            history = pd.read_csv(HISTORY_FILE)
            st.dataframe(history.tail(10), use_container_width=True)
            st.download_button(
                "Download Prediction History",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.warning(f"Could not read prediction history: {error}")
    else:
        st.info("No prediction history available yet.")
 
    if st.button("⬅️ Check Another Trip", type="secondary"):
        st.session_state.page = "input"
        st.session_state.suggestion_index = 0
        st.session_state.pop("saved_result_key", None)
        st.rerun()
 
 
#CONTRIBUTE TRAVEL DATA
 
st.divider()
st.header("Contribute to the App with More Details of Your Travel")
st.write(
    "Help improve future predictions by sharing an observed traffic condition. "
    "No vehicle-count information is required."
)
 
with st.expander("Add a traffic record", expanded=False):
    with st.form("contribute_traffic_form"):
        contribution_city = st.text_input(
            "City", placeholder="Example: Kochi"
        )
        contribution_area = st.text_input(
            "Area / Road Name", placeholder="Example: MG Road"
        )
        contribution_date = st.date_input(
            "Date of observation", value=datetime.date.today(), key="contribution_date"
        )
        contribution_time = st.time_input(
            "Time of observation", value=datetime.datetime.now().time().replace(second=0, microsecond=0),
            key="contribution_time"
        )
        contribution_level = st.selectbox(
            "Observed traffic density",
            ["Low", "Medium", "High", "Severe"],
            help="Choose the traffic level you observed at that place and time."
        )
        contribution_submitted = st.form_submit_button(
            "Submit Travel Details", type="primary", use_container_width=True
        )
 
    if contribution_submitted:
        city_clean = contribution_city.strip()
        area_clean = contribution_area.strip()
 
        if not city_clean or not area_clean:
            st.error("Please enter both the city and area / road name.")
        else:
            try:
               
                original = pd.read_csv(DATA_FILE)
                city_matches = original["city"].astype(str).str.strip().str.casefold() == city_clean.casefold()
                area_matches = original["area"].astype(str).str.strip().str.casefold() == area_clean.casefold()
                matching_rows = original[city_matches & area_matches]
 
                if not matching_rows.empty:
                    route_id = str(matching_rows.iloc[0]["road_id"])
                    route_status = f"Existing route found. Reusing Route ID **{route_id}**."
                else:
                    route_numbers = (
                        pd.to_numeric(
                            original["road_id"].astype(str).str.extract(r"(\d+)")[0],
                            errors="coerce",
                        ).dropna()
                    )
                    next_number = int(route_numbers.max()) + 1 if not route_numbers.empty else 1
                    route_id = f"RD{next_number:04d}"
                    route_status = f"New area added with Route ID **{route_id}**."
 
                level_medians = original.groupby("congestion_level")["traffic_density"].median()
                if contribution_level in level_medians.index:
                    density_value = float(level_medians.loc[contribution_level])
                else:
                    density_defaults = {"Low": 15.0, "Medium": 45.0, "High": 74.0, "Severe": 99.0}
                    density_value = density_defaults[contribution_level]
 
                observed_date = contribution_date
                new_record = {
                    "date": observed_date.strftime("%m/%d/%Y"),
                    "time": contribution_time.strftime("%H:%M:%S"),
                    "day": observed_date.strftime("%A"),
                    "city": city_clean,
                    "area": area_clean,
                    "road_id": route_id,
                    "vehicle_count": pd.NA,
                    "car_count": pd.NA,
                    "bike_count": pd.NA,
                    "bus_count": pd.NA,
                    "truck_count": pd.NA,
                    "traffic_density": density_value,
                    "congestion_level": contribution_level,
                }
 
                updated = pd.concat(
                    [original, pd.DataFrame([new_record])], ignore_index=True
                )
                updated.to_csv(DATA_FILE, index=False)
 
                load_dataset.clear()
                train_model.clear()
                st.success("Thank you! Your traffic record has been added to the dataset.")
                st.info(route_status)
                st.caption(
                    f"Recorded {city_clean} — {area_clean}, "
                    f"{observed_date.strftime('%d-%m-%Y')} at "
                    f"{contribution_time.strftime('%H:%M')}, "
                    f"{contribution_level} traffic."
                )
                st.warning(
                    "The new record will be included when the model is next loaded. "
                    "Vehicle-count fields were left blank, not estimated."
                )
            except Exception as error:
                st.error(f"Could not add your contribution: {error}")
 
 
#FOOTER
 
st.divider()
st.caption(
    "Smart City Traffic Predictor"
)

import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

@st.cache_resource
def load_models():
    return joblib.load("gamma_models_selected.pkl")

loaded_models = load_models() # Load once and reuse

# Streamlit UI
st.title("Gamma Topic Prediction")

# User input
text_input = st.text_area("Enter text to classify:", "")

# Define a probability threshold (cutoff) for displaying topics
probability_cutoff = 0.2  # Adjust this threshold as needed

if st.button("Predict Topics"):
    if text_input.strip():
        predictions = {}
        shap_explanations = {}

        for topic, model_tuple in loaded_models.items():
            if not isinstance(model_tuple, tuple):
                print(f"Error: Model for topic {topic} is not a tuple! Got {type(model_tuple)}")
                continue

            vectorizer, classifier = model_tuple  # Unpack the tuple
            # Transform input text using the stored vectorizer
            X_input_transformed = vectorizer.transform([text_input])

            # Ensure format is consistent with model training
            if hasattr(classifier, "predict_proba"):  # Ensure the classifier has predict_proba
                # Get probability of topic presence
                prob = classifier.predict_proba(X_input_transformed)[0][1]  # Probability of topic presence
                if prob >= probability_cutoff:  # Apply threshold
                    predictions[topic] = prob

                    # SHAP explanation
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer.shap_values(X_input_transformed)[0]

                    # Get feature names
                    feature_names = vectorizer.get_feature_names_out()

                    # Extract only words from the input text
                    input_words = set(text_input.lower().split())  # Get unique words from input
                    word_contributions = {
                        feature_names[i]: shap_values[i]  # Shap value per word
                        for i in range(len(feature_names))
                        if feature_names[i] in input_words  # Keep only words from input text
                    }

                    shap_explanations[topic] = word_contributions

        # Display results
        st.subheader("Prediction Results:")
        if predictions:
            for topic, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{topic}**: {prob:.4f}")

                # Sort and get the top 10 SHAP feature contributions
                sorted_shap = sorted(shap_explanations[topic].items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                words, impacts = zip(*sorted_shap)

                # Matplotlib bar chart
                fig, ax = plt.subplots()
                ax.barh(words, impacts, color="skyblue")
                ax.set_xlabel("SHAP Value")
                ax.set_title(f"Top 10 Feature Contributions for {topic}")
                ax.invert_yaxis()  # Highest impact on top

                st.pyplot(fig)
                
        else:
            st.write("No topics met the probability cutoff.")

    else:
        st.warning("Please enter text before predicting.")

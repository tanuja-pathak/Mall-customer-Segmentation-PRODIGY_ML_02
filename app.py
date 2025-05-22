import gradio as gr
import joblib
import numpy as np

# Load the saved model for predictions
kmeans = joblib.load('mall_customer_segmentation.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the prediction function
def predict_cluster(gender, age, annual_income, spending_score):
        # Encode gender
        gender_encoded = label_encoder.transform([gender])[0]

        # Create input data
        data = np.array([[gender_encoded, age, annual_income, spending_score]])

        # Predict the cluster
        cluster = kmeans.predict(data)[0]

        # Return the result
        return f"The customer belongs to cluster: {cluster}"


# Create the Gradio interface
interface = gr.Interface(
    fn=predict_cluster,
    inputs=[
        gr.Dropdown(choices=['Male', 'Female'], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Annual Income (k$)"),
        gr.Number(label="Spending Score (1-100)"),
    ],
    outputs=gr.Textbox(label="Predicted Cluster"),
    title="Mall Customer Segmentation",
    description="Enter customer details to find their cluster."
)

# Launch the Gradio app
interface.launch(share=True)

from fastapi import FastAPI

app = FastAPI(
    title="Fraud Detection API",
    description="API for Fraud Detection using the Optimized LightGBM Model",
    version="1.0")


@app.get("/")
def home():
    return {
        "message": "Welcome to the Fraud Detection API"}

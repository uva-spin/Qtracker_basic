from dotenv import load_dotenv

# from mangum import Mangum

# from src.api.app import app

load_dotenv(override=True)

# For AWS Lambda compatibility
# handler = Mangum(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

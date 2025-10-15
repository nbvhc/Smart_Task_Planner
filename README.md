# Smart Task Planner

A Flask web application that uses Google's Gemini AI to break down project goals into actionable tasks.

## Features

- Flask backend with Gemini AI integration
- Interactive web interface for goal submission
- Automatic task breakdown with dependencies
- Development mode with stub responses
- Configurable model selection
- Error handling and timeouts

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Smart_Task_Planner
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Configuration

Environment variables:
- `GEMINI_API_KEY`: Your Google Gemini API key
- `MODEL_NAME`: (Optional) Specify a particular Gemini model to use
- `USE_STUB`: Set to "true" for development without API calls

## Running the Application

1. Start the Flask server:
```bash
cd app
python -m flask run
```

2. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Development Mode

To run in stub mode without making API calls:
```bash
export USE_STUB=true  # Linux/Mac
set USE_STUB=true    # Windows
python -m flask run
```

## API Endpoints

- `GET /`: Main web interface
- `GET /health`: API health check
- `POST /generate-plan`: Generate a task breakdown
  - Request body: `{"goal": "Your project goal here"}`
  - Response: `{"status": "SUCCESS", "tasks": [...]}`

## Project Structure

```
smart-task-planner/
├── app/
│   ├── app.py            # Flask application
│   ├── templates/        # HTML templates
│   │   └── index.html   # Main interface
│   └── static/          # Static assets (if any)
├── .env                 # Environment variables (not in git)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Testing

```bash
# Check API health
curl http://127.0.0.1:5000/health

# Generate a task plan
curl -X POST http://127.0.0.1:5000/generate-plan \
  -H "Content-Type: application/json" \
  -d '{"goal":"Build a website in one month"}'
```
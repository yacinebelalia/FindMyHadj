# FindMyHadj - Face Recognition API

A robust face recognition API built with FastAPI and InsightFace, designed to help identify and match faces in a database of Hajj pilgrims.

## Features

- Face detection and embedding extraction
- Face recognition and matching
- Age and gender detection
- RESTful API endpoints
- Docker containerization
- Supabase integration for data storage
- Comprehensive logging and error handling

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Supabase account and credentials

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FindMyHadj.git
cd FindMyHadj
```

2. Build and run with Docker:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /`: Check if the API is running
- Response: `{"message": "Face recognition API is running"}`

### Get All Data
- `GET /data`: Retrieve all records from the database
- Response: List of all records with face embeddings

### Get Face Embedding
- `POST /get_embedding`: Extract face embedding from an image
- Input: Image file
- Response: Face embedding vector

### Recognize Face
- `POST /recognize`: Match a face against the database
- Input: Image file
- Response: Matched person's information or "No matching identity found"

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
FindMyHadj/
├── App/
│   ├── face_api.py      # FastAPI application and endpoints
│   └── face_processor.py # Face detection and processing logic
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile          # Docker build instructions
├── requirements.txt    # Python dependencies
└── .env               # Environment variables (create this)
```

## Development

### Local Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
uvicorn App.face_api:app --reload
```

### Docker Development

The Docker setup includes:
- Hot-reload for development
- Volume mounting for live code updates
- Health checks
- Logging configuration
- Non-root user for security

## Logging

The application uses Python's logging module with the following levels:
- ERROR: Critical issues
- WARNING: Potential problems
- INFO: General operations
- DEBUG: Detailed debugging

Logs are available in the Docker container output and can be viewed using:
```bash
docker-compose logs -f
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License



## Support

For support, please [add contact information or issue reporting guidelines]
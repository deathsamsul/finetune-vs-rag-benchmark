"""
serving/ — FastAPI serving layer.

Start with:
    uvicorn serving.main:app --reload --port 8000

Endpoints:
    POST /query         single question
    POST /batch-query   multiple questions
    GET  /health        model load status
    GET  /metrics       RAGAS benchmark scores
    GET  /docs          Swagger UI
"""

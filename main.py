import os
import logging
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Public LLMs-txt API")

# ─────────────  CORS for “any origin”  ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # wildcard → every scheme+host is allowed
    allow_methods=["*"],       # allow GET, POST, PUT, DELETE, OPTIONS, …
    allow_headers=["*"],       # allow any non-simple request header
    allow_credentials=False,   # MUST be False with "*" (CORS spec)
    max_age=86400,             # cache pre-flight 24 h (optional)
)


# ─── 1. CONFIGURATION ────────────────────────────────────────────────────────
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
try:
   RATE_LIMIT = int(os.getenv("RATE_LIMIT", "10"))
except ValueError:
   raise ValueError("RATE_LIMIT must be an integer")


if not OPENAI_API_KEY:
   raise ValueError("Missing OPENAI_API_KEY in environment")
if not INTERNAL_API_KEY:
   raise ValueError("Missing INTERNAL_API_KEY in environment")


# ─── 2. LOGGER SETUP ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("llms_txt_api")


# ─── 3. OPENAI CLIENT ────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)


# ─── 4. AUTHENTICATION ───────────────────────────────────────────────────────
security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
   """
   Validates the Bearer token and returns the API key if valid.
   """
   if credentials.scheme.lower() != "bearer":
       raise HTTPException(status_code=401, detail="Invalid authentication scheme")
   token = credentials.credentials
   if token != INTERNAL_API_KEY:
       raise HTTPException(status_code=401, detail="Invalid API key")
   return token


# ─── 5. RATE LIMITING ─────────────────────────────────────────────────────────


def _get_api_key_from_request(request: Request) -> str:
   """
   Extract the bearer token from Authorization header for rate limiting key.
   """
   auth: str = request.headers.get("authorization", "")
   scheme, _, token = auth.partition(" ")
   return token if scheme.lower() == "bearer" else "anonymous"


limiter = Limiter(key_func=_get_api_key_from_request)
app = FastAPI(title="LLMs.txt Generator")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
   status_code=429, content={"detail": "Too Many Requests"}
))


# ─── 6. REQUEST MODEL ────────────────────────────────────────────────────────
class URLRequest(BaseModel):
   url: HttpUrl


# ─── 7. ROUTES ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/generate-llms-txt")
@limiter.limit(f"{RATE_LIMIT}/minute")
async def generate_llms(
   request: Request,
   request_data: URLRequest,
   api_key: str = Depends(verify_api_key)
) -> dict:
   """
   Generate an LLMs.txt summary for a given URL.
   Protected by API key and rate limited per key.
   """
   url = str(request_data.url)
   logger.info("Request by %s from %s for URL: %s", api_key, request.client.host, url)
   prompt = (
       f"Generate an LLMs.txt summary for the brand at {url}.\n"
       "Structure output in Markdown with headings: Brand, Description, Product Categories, "
       "Key Initiatives, Customer Resources, Optional."
   )
   try:
       response = client.responses.create(
           model="gpt-4.1",
           tools=[{"type": "web_search_preview"}],
           input=prompt
       )
       return {"llms_txt": response.output_text}
   except OpenAIError as e:
       logger.exception("OpenAI API error: %s", str(e))
       raise HTTPException(status_code=502, detail="Upstream OpenAI API error")
   except Exception:
       logger.exception("Unexpected error")
       raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check() -> dict:
   """Simple health check endpoint"""
   logger.info("Health check performed")
   return {"status": "ok"}


# ─── 8. RUNNER ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, uvicorn

    # DigitalOcean sets $PORT for you; default to 8000 locally
    port = int(os.getenv("PORT", 8000))
    # Listen on all interfaces, not just loopback
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,      # disable auto-reload in production
        log_level="info",
    )



"""Browser request protection for the single-user localhost application."""
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class LocalRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Cross-site reads can expose local projects; cross-site writes can spend
        # API credits or change configuration. CORS alone does not block writes.
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site requests are forbidden"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if origin != expected or parsed.username or parsed.password:
                return JSONResponse({"detail": "Origin must match this application"}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if request.headers.get("x-clipforge-request") != "1":
                return JSONResponse({"detail": "Missing local request header"}, status_code=403)
        return await call_next(request)

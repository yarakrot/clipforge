import sys
import unittest
from pathlib import Path

from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from local_security import LocalRequestMiddleware


class LocalSecurityTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.add_middleware(LocalRequestMiddleware)
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

        @app.get("/config")
        async def read_config():
            return {"ok": True}

        @app.post("/config")
        async def write_config():
            return {"ok": True}

        self.client = TestClient(app, base_url="http://127.0.0.1:8000")

    def test_local_read(self):
        self.assertEqual(self.client.get("/config").status_code, 200)

    def test_same_origin_write(self):
        response = self.client.post("/config", headers={"Origin": "http://127.0.0.1:8000", "X-ClipForge-Request": "1"})
        self.assertEqual(response.status_code, 200)

    def test_simple_write_without_header_is_blocked(self):
        self.assertEqual(self.client.post("/config", data={"key": "dummy"}).status_code, 403)

    def test_cross_origin_write_is_blocked_even_with_header(self):
        response = self.client.post("/config", headers={"Origin": "https://example.com", "X-ClipForge-Request": "1"})
        self.assertEqual(response.status_code, 403)

    def test_cross_site_read_is_blocked(self):
        self.assertEqual(self.client.get("/config", headers={"Sec-Fetch-Site": "cross-site"}).status_code, 403)

    def test_null_origin_is_blocked(self):
        self.assertEqual(self.client.get("/config", headers={"Origin": "null"}).status_code, 403)

    def test_different_port_is_blocked(self):
        self.assertEqual(self.client.get("/config", headers={"Origin": "http://127.0.0.1:9000"}).status_code, 403)

    def test_untrusted_host_is_blocked(self):
        self.assertEqual(self.client.get("/config", headers={"Host": "example.com"}).status_code, 400)

    def test_localhost_at_launcher_port_is_allowed(self):
        response = self.client.post("/config", headers={"Host": "localhost:8012", "Origin": "http://localhost:8012", "X-ClipForge-Request": "1"})
        self.assertEqual(response.status_code, 200)


class ApplicationSmokeTests(unittest.TestCase):
    def test_application_import_and_routes(self):
        from main import app

        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            for path in ("/", "/config", "/projects"):
                with self.subTest(path=path):
                    self.assertEqual(client.get(path).status_code, 200)
            self.assertEqual(client.post("/config/deepseek-key", json={"api_key": ""}).status_code, 403)
            self.assertEqual(client.post("/config/deepseek-key", headers={"X-ClipForge-Request": "1"}, json={"api_key": ""}).status_code, 400)
            self.assertEqual(client.post("/projects/not-a-uuid/load", headers={"X-ClipForge-Request": "1"}).status_code, 400)
            self.assertEqual(client.get("/download/missing.mp4").status_code, 404)


if __name__ == "__main__":
    unittest.main()

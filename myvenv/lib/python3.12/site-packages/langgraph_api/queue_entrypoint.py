# ruff: noqa: E402
import os

if not (
    (disable_truststore := os.getenv("DISABLE_TRUSTSTORE"))
    and disable_truststore.lower() == "true"
):
    import truststore  # noqa: F401

    truststore.inject_into_ssl()  # noqa: F401

import asyncio
import http.server
import json
import logging.config
import os
import pathlib

import structlog
import uvloop

from langgraph_runtime.lifespan import lifespan

logger = structlog.stdlib.get_logger(__name__)


async def healthcheck_server():
    port = int(os.getenv("PORT", "8080"))
    ok = json.dumps({"status": "ok"}).encode()
    ok_len = str(len(ok))

    class HealthHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Skip logging for /ok endpoint
            if self.path == "/ok":
                return
            # Log other requests normally
            super().log_message(format, *args)

        def do_GET(self):
            if self.path == "/ok":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", ok_len)
                self.end_headers()
                self.wfile.write(ok)
            else:
                self.send_error(http.HTTPStatus.NOT_FOUND)

    with http.server.ThreadingHTTPServer(("0.0.0.0", port), HealthHandler) as httpd:
        logger.info(f"Health server started at http://0.0.0.0:{port}")
        try:
            await asyncio.to_thread(httpd.serve_forever)
        finally:
            httpd.shutdown()


async def entrypoint():
    from langgraph_api import logging as lg_logging

    lg_logging.set_logging_context({"entrypoint": "python-queue"})
    tasks: set[asyncio.Task] = set()
    # start simple http server for health checks
    tasks.add(asyncio.create_task(healthcheck_server()))
    # start queue and associated tasks
    async with lifespan(None, with_cron_scheduler=False, taskset=tasks):
        # run forever, error if any tasks fail
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            await logger.awarning("Queue entrypoint cancelled", exc_info=True)


if __name__ == "__main__":
    # set up logging
    with open(pathlib.Path(__file__).parent.parent / "logging.json") as file:
        loaded_config = json.load(file)
        logging.config.dictConfig(loaded_config)

    # set up uvloop
    uvloop.install()

    # run the entrypoint
    asyncio.run(entrypoint())

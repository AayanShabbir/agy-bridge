"""CLI entrypoint for running agy-bridge as a host daemon."""
import os
import uvicorn

from agy_bridge.config import BridgeConfig
from agy_bridge.bootstrap import build_bridge_app


def main():
    host = os.environ.get("AGY_BRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("AGY_BRIDGE_PORT", "8790"))

    config = BridgeConfig(host=host, port=port)
    app = build_bridge_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

"""
FastAPI application for the Vendor Negotiation Gym.
Uses create_app() from openenv-core for spec-compliant endpoint generation.
"""
import uvicorn
from openenv.core.env_server import create_app

from server.environment import VendorNegotiationEnvironment
from models import NegotiationAction, NegotiationObservation

app = create_app(
    VendorNegotiationEnvironment,
    NegotiationAction,
    NegotiationObservation,
    env_name="vendor-negotiation-gym",
)


@app.get("/", include_in_schema=False)
def index():
    return {
        "name": "vendor-negotiation-gym",
        "status": "running",
        "description": (
            "OpenEnv environment for multi-dimensional enterprise contract "
            "negotiation under partial observability."
        ),
        "endpoints": {
            "health": "GET /health",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
        },
        "tasks": [
            "deal_qualification",
            "multi_term_negotiation",
            "strategic_contract_close",
        ],
        "repo": "https://github.com/atharva-deopujari/openenv-vendor-negotiation-gym",
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

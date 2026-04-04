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


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

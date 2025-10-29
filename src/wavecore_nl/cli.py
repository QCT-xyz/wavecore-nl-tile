import json
import click
from .config import settings


@click.group()
def main() -> None:
    """WaveCore-NL Tile CLI"""
    pass


@main.command()
def show_config() -> None:
    """Print current runtime config from .env or defaults."""
    print(
        json.dumps(
            {
                "backend": settings.backend,
                "modes": settings.modes,
                "alpha": settings.alpha,
                "depth": settings.depth,
                "shots": settings.shots,
                "onnx_opset": settings.onnx_opset,
                "log_level": settings.log_level,
                "seed": settings.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Produce one external native MODIS tile using the portable frozen models."""
from production_cli import parser, configure


def main():
    p = parser("tile")
    a = p.parse_args()
    try:
        config = configure(a, "tile")
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        p.error(str(exc))
    if not a.check_only:
        from portable_runtime import produce
        produce(config)


if __name__ == "__main__":
    main()

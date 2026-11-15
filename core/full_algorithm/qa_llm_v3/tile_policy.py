"""Inference allowlist, separate from the immutable eight-tile fitting population."""
import json
from pathlib import Path

_registry = json.loads((Path(__file__).resolve().parents[2] / 'tile_registry.json').read_text())
TRAINING_TILES = tuple(_registry['training'])
INDEPENDENT_TILES = tuple(_registry['independent'])
PRODUCTION_TILES = TRAINING_TILES + INDEPENDENT_TILES
if len(set(PRODUCTION_TILES)) != len(PRODUCTION_TILES):
    raise ValueError('Training and independent tile populations must be disjoint')


def validate_production_tiles(text):
    tiles = text.split(',')
    if not tiles or len(set(tiles)) != len(tiles) or not set(tiles) <= set(PRODUCTION_TILES):
        raise ValueError('Invalid production tile selection')
    return tiles

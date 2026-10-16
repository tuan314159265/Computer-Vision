"""Load project input images, preferring user datasets over bundled test assets."""
from pathlib import Path

from lib import utils
from lib.gradient_domain import list_images, load_rgb


def _named_images(folder, prefix):
    """Return images with a named prefix in deterministic order."""
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
        and path.stem.lower().startswith(prefix.lower())
    )


def load_panorama_set(data_dir, set_name, demo_seed):
    """Load a panorama set from data/<set_name>, then known test_assets if absent."""
    data_dir = Path(data_dir)
    folder = data_dir / set_name
    files = list_images(folder)

    if len(files) < 2:
        fallback_prefix = {'pano_set1': 'boat', 'pano_set2': 'budapest'}.get(set_name)
        fallback_files = _named_images(data_dir / 'test_assets', fallback_prefix) if fallback_prefix else []
        if len(fallback_files) >= 2:
            files = fallback_files
            source = f'test_assets/{fallback_prefix}*'
        else:
            return utils.make_demo_panorama(seed=demo_seed, n_images=4), 'DEMO'
    else:
        source = str(folder)

    return [load_rgb(path) for path in files], f'REAL: {source} ({len(files)} ảnh)'


def load_gradient_pair(data_dir, output_dir):
    """Load the gradient-editing images, falling back to test assets then DEMO."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    candidates = [data_dir / 'gradient', data_dir / 'test_assets' / 'gradient']

    for folder in candidates:
        background_path = folder / 'background.jpg'
        source_path = folder / 'source.jpg'
        if background_path.exists() and source_path.exists():
            mask_path = folder / 'mask.png'
            return (
                load_rgb(background_path),
                load_rgb(source_path, max_side=500),
                mask_path if mask_path.exists() else None,
                f'REAL: {folder}',
            )

    background, source, mask = utils.make_demo_gradient_pair()
    mask_path = output_dir / 'demo_mask.png'
    import cv2
    cv2.imwrite(str(mask_path), mask)
    return background, source, mask_path, 'DEMO'


def load_project_inputs(data_dir, output_dir):
    """Load gradient inputs and both panorama sets from the project data tree."""
    background, source, mask_path, gradient_status = load_gradient_pair(data_dir, output_dir)
    panorama_sets = {}
    statuses = {'gradient': gradient_status}

    for index, set_name in enumerate(('pano_set1', 'pano_set2')):
        images, status = load_panorama_set(data_dir, set_name, demo_seed=11 + index * 11)
        panorama_sets[set_name] = images
        statuses[set_name] = status

    return background, source, mask_path, panorama_sets, statuses

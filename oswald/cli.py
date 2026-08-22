import click
import sys

from oswald.bootstrap import configure_torch_cuda_allocator, preload_nvidia_libraries


class LazyGroup(click.Group):
    def __init__(self, *args, **kwargs):
        self._lazy_commands = kwargs.pop("lazy_commands", {})
        super().__init__(*args, **kwargs)

    def list_commands(self, ctx):
        return sorted(list(self._lazy_commands.keys()) + list(self.commands.keys()))

    def get_command(self, ctx, name):
        if name in self._lazy_commands:
            import_path, func_name = self._lazy_commands[name]
            try:
                mod = __import__(import_path, fromlist=[func_name])
                return getattr(mod, func_name)
            except Exception as e:
                click.echo(f"Error loading command '{name}': {e}", err=True)
                sys.exit(1)
        return super().get_command(ctx, name)


lazy_commands = {
    "train": ("terrain_diffusion.training.train", "main"),
    "build-base-dataset": (
        "terrain_diffusion.data.preprocessing.build_base_dataset",
        "process_base_dataset",
    ),
    "build-encoded-dataset": (
        "terrain_diffusion.data.preprocessing.build_encoded_dataset",
        "process_encoded_dataset",
    ),
    "define-splits": (
        "terrain_diffusion.data.preprocessing.define_splits",
        "split_dataset",
    ),
    "explore": ("terrain_diffusion.inference.explorer.server", "main"),
    "onnx-export": ("terrain_diffusion.onnx.export", "main"),
    "azgaar-to-tiff": ("terrain_diffusion.inference.utils.azgaar_to_tiff", "main"),
    "generate-map": ("terrain_diffusion.inference.generate_map", "main"),
    "tiff-stats": ("terrain_diffusion.inference.utils.tiff_stats", "main"),
}


@click.group(cls=LazyGroup, lazy_commands=lazy_commands)
def cli():
    """Oswald CLI - Main entry point for terrain workflows"""


def main():
    preload_nvidia_libraries()
    configure_torch_cuda_allocator()
    cli()

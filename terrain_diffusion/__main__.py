import click
import sys
import os

# Set memory management environment variable to reduce fragmentation
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class LazyGroup(click.Group):
    def __init__(self, *args, **kwargs):
        self._lazy_commands = kwargs.pop('lazy_commands', {})
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
    # Training commands
    'train': ('terrain_diffusion.training.train', 'main'),
    'save-model': ('terrain_diffusion.training.save_model', 'save_model'),
    
    # Data preprocessing commands
    'build-base-dataset': ('terrain_diffusion.data.preprocessing.build_base_dataset', 'process_base_dataset'),
    'build-encoded-dataset': ('terrain_diffusion.data.preprocessing.build_encoded_dataset', 'process_encoded_dataset'),
    'define-splits': ('terrain_diffusion.data.preprocessing.define_splits', 'split_dataset'),
    
    # Inference commands
    'explore': ('terrain_diffusion.inference.explorer.server', 'main'),
    'generate': ('terrain_diffusion.inference.world_generator', 'main'),
    'api': ('terrain_diffusion.inference.api', 'main'),
    'mc-api': ('terrain_diffusion.inference.minecraft_api', 'main'),
    'onnx-export': ('terrain_diffusion.onnx.export', 'main'),
    'azgaar-to-tiff': ('terrain_diffusion.inference.utils.azgaar_to_tiff', 'main'),
    'tiff-export': ('terrain_diffusion.inference.tiff_export', 'main'),
    'generate-map': ('terrain_diffusion.inference.generate_map', 'main'),
    'tiff-stats': ('terrain_diffusion.inference.utils.tiff_stats', 'main'),
}

@click.group(cls=LazyGroup, lazy_commands=lazy_commands)
def cli():
    """Terrain Diffusion CLI - Main entry point for all commands"""
    pass

if __name__ == '__main__':
    cli()

import click
import rasterio
import numpy as np

def calculate_stats(path, bins=10):
    if isinstance(path, np.ndarray):
        data = path.astype(np.float32)
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return None
        return {
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "median": float(np.median(valid_data)),
            "std": float(np.std(valid_data)),
            "percentiles": {p: float(np.percentile(valid_data, p)) for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]},
            "histogram": np.histogram(valid_data, bins=bins)
        }

    with rasterio.open(path) as ds:
        ov_factors = ds.overviews(1)
        if ov_factors:
            # Use the most decimated overview for speed
            factor = ov_factors[-1]
            data = ds.read(1, out_shape=(ds.height // factor, ds.width // factor))
        else:
            # Sample 1024x1024 if no overviews
            data = ds.read(1, out_shape=(1024, 1024))
            
        data = data.astype(np.float32)
        if ds.nodata is not None:
            data[data == ds.nodata] = np.nan
        
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return None
            
        stats = {
            "min": np.min(valid_data),
            "max": np.max(valid_data),
            "mean": np.mean(valid_data),
            "median": np.median(valid_data),
            "std": np.std(valid_data),
            "percentiles": {p: np.percentile(valid_data, p) for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]},
            "histogram": np.histogram(valid_data, bins=bins)
        }
        return stats

@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--bins", default=10, help="Number of bins for the histogram")
def main(path, bins):
    """Display statistics and a text-based histogram for a TIFF file."""
    stats = calculate_stats(path, bins=bins)
    if stats is None:
        click.echo("No valid data found in TIFF.")
        return

    click.echo(f"Statistics for {path}:")
    click.echo(f"  Min:    {stats['min']:.2f}")
    click.echo(f"  Max:    {stats['max']:.2f}")
    click.echo(f"  Mean:   {stats['mean']:.2f}")
    click.echo(f"  Median: {stats['median']:.2f}")
    click.echo(f"  Std:    {stats['std']:.2f}")
    click.echo("\nPercentiles:")
    for p, val in sorted(stats['percentiles'].items()):
        click.echo(f"  {p:3d}%: {val:8.2f}")

    hist, bin_edges = stats['histogram']
    click.echo("\nHistogram:")
    max_h = np.max(hist)
    for i in range(len(hist)):
        bar = "*" * int(hist[i] / max_h * 40) if max_h > 0 else ""
        click.echo(f"  [{bin_edges[i]:8.1f}, {bin_edges[i+1]:8.1f}]: {bar} ({hist[i]})")

if __name__ == "__main__":
    main()

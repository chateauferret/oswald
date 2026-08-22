# oswald

Curating global terrain using terrain-diffusion and other algorithms.

## `terrain_diffusion/` upstream sync

The `terrain_diffusion/` directory is bootstrapped as a git subtree from the
`terrain_diffusion/` package inside
[`xandergos/terrain-diffusion`](https://github.com/xandergos/terrain-diffusion)
on `master`.

Oswald-specific files live outside that subtree-driven package flow, notably:

- `configs/`
- `legends/`
- `make_terrain.ipynb`
- `oswald/`

The Oswald-owned runtime helpers now live under `oswald/`. The files kept under
`terrain_diffusion/` for `__main__.py`, `__init__.py`, and `paths.py` are thin
compatibility wrappers so existing imports and `python -m terrain_diffusion`
continue to work while the actual custom logic stays outside the subtree-owned
package.

### One-time remote setup

Git remotes are local clone configuration, so each clone needs:

```bash
git remote add upstream https://github.com/xandergos/terrain-diffusion.git
git fetch upstream master
```

### Pull upstream package updates

The upstream repository root contains more than the Python package, so syncing
the package subtree requires splitting `upstream/master` down to its
`terrain_diffusion/` directory first and then merging that split history:

```bash
git fetch upstream master
split_commit=$(git subtree split --prefix=terrain_diffusion refs/remotes/upstream/master)
git subtree merge --prefix=terrain_diffusion "$split_commit" --squash
```

Using `git subtree pull --prefix terrain_diffusion upstream master --squash`
directly would import the upstream repository root under `terrain_diffusion/`,
which would create a nested `terrain_diffusion/terrain_diffusion/...` layout.

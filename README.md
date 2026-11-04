# ror-to-blender

Convert Rigs of Rods / OGRE vehicle mods (trains, trucks, cars, trailers, boats, aircraft) to a `.glb` file you can import into Blender, with textures.

One Python file. No game install needed.

## Quick start

1. Install [Python 3.8+](https://www.python.org/downloads/) and the two libraries:
   ```
   pip install numpy pillow
   ```
2. Run the script:
   ```
   python ror2glb.py
   ```
3. A window opens. Select the folder that contains the mod (the `.train` / `.truck` file, `.mesh`, `.material` and `.dds` files). Subfolders are searched too.
4. If the mod has several vehicle files or configurations, pick one from the list.
5. The `.glb` is saved in the folder you picked. In Blender: **File > Import > glTF 2.0 (.glb/.gltf)**.

You can also drag a mod folder or `.zip` onto `ror2glb.py`.

## Command line

```
python ror2glb.py <mod folder or .zip> [vehicle file] [options]
```

| Option | What it does |
| --- | --- |
| `-o FILE` | Output path for the `.glb` |
| `--config NAME` | Use a `sectionconfig` variant (repeatable). Default: the first one |
| `--list-configs` | Print the available configurations and exit |
| `--no-wheels` | Skip meshwheels / axle meshes |
| `--wheel-axis x\|y\|z\|auto` | Local axis of wheel meshes (default: auto-detect) |
| `--max-tex N` | Downscale textures so the longest side is at most N pixels |
| `--no-textures` | Export geometry only |
| `--version` | Print the version |

## What it converts

- Props and flexbodies, placed from their reference nodes, offsets and rotations
- Meshwheels and meshwheels2 (axles and wheelsets), at their rest position
- Materials from every `.material` file, including inheritance, alpha blend / alpha rejection and emissive
- `managedmaterials` textures
- `.dds`, `.png`, `.tga`, `.jpg` textures, embedded in the `.glb`

## Limits

- Parts come in at their rest position. Nothing is animated or rigged (wheels, suspension, pantographs, doors).
- Flexbodies are exported as static meshes.
- Collision geometry (`submesh`, `cab`) is ignored; only visible meshes are exported.
- Materials that live in the game's own files (not in the mod) are not available and get a grey colour. The script lists them.
- Textures need Pillow's DDS support. If textures come out grey, run `pip install -U pillow`. Very new compression formats (for example BC7) may still fail; the script lists these.

## Troubleshooting

The script prints a summary at the end: parts exported, files it could not find, textures it could not decode, and meshes with problems. Start there.

## License

MIT. See [LICENSE](LICENSE).

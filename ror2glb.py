#!/usr/bin/env python3
"""Rigs of Rods / OGRE vehicle mod -> glTF binary (.glb) for Blender.

Single-file version (OGRE .mesh reader included).

usage:
    python ror2glb.py                      (opens a window: pick the mod folder)
    python ror2glb.py <mod folder or .zip> [vehicle file] [options]
    (you can also drag a folder onto the script)

Reads every OGRE .mesh the vehicle file places (props, flexbodies, meshwheels),
applies .material / managedmaterials textures (.dds etc. -> PNG, embedded) and
puts each part where the vehicle file says (rest position, not animated).

Options:
    -o FILE            output .glb path
    --config NAME      pick a sectionconfig (repeatable). Default: first one.
    --list-configs     print the sectionconfigs in the vehicle file and exit
    --no-wheels        skip meshwheels / axle meshes
    --wheel-axis A     x|y|z|auto  local axis of wheel meshes (default auto)
    --max-tex N        downscale textures so the longest side is <= N px
    --no-textures      skip textures entirely

Needs: Python 3.8+, numpy, Pillow  (pip install numpy pillow)
Import the .glb in Blender with File > Import > glTF 2.0.
"""
import argparse
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image


# =============================================================================
#  OGRE .mesh reader
# =============================================================================
SEM_POSITION, SEM_NORMAL, SEM_TEXCOORD = 1, 4, 7


class SubMesh:
    def __init__(self):
        self.material = ''
        self.uses_shared = False
        self.indices = np.zeros(0, np.int64)
        self.op = 4                 # 4 = triangle list, 5 = strip, 6 = fan
        self.vcount = 0
        self.attrs = None           # {(semantic, index): float32 array (n, k)}


def read_mesh(path):
    """Return (version_string, [SubMesh, ...])."""
    with open(path, 'rb') as f:
        b = f.read()
    if len(b) < 8:
        raise ValueError('file too small')
    hid = struct.unpack_from('<H', b, 0)[0]
    if hid == 0x0010:
        raise ValueError('big-endian .mesh files are not supported')
    if hid != 0x1000:
        raise ValueError('not an OGRE .mesh file (header 0x%x)' % hid)
    nl = b.index(b'\n', 2)
    version = b[2:nl].decode('latin1')
    n = len(b)

    def chunk(p):
        if p + 6 > n:
            return 0, 0
        return struct.unpack_from('<HI', b, p)

    def cstr(p):
        e = b.index(b'\n', p)
        return b[p:e].decode('latin1'), e + 1

    def read_geometry(p, end):
        vcount = struct.unpack_from('<I', b, p)[0]
        p += 4
        decl, bufs = [], {}
        while p + 6 <= end:
            cid, ln = chunk(p)
            if ln < 6:
                break
            q = p + 6
            if cid == 0x5100:                       # vertex declaration
                while q + 6 <= p + ln:
                    c2, l2 = chunk(q)
                    if l2 < 6:
                        break
                    if c2 == 0x5110:                # one element
                        decl.append(struct.unpack_from('<5H', b, q + 6))
                    q += l2
            elif cid == 0x5200:                     # vertex buffer
                bind, vsize = struct.unpack_from('<2H', b, q)
                q += 4
                c2, l2 = chunk(q)
                if c2 == 0x5210:
                    bufs[bind] = (vsize, b[q + 6:q + l2])
            p += ln
        return vcount, decl, bufs

    def decode(vcount, decl, bufs):
        out = {}
        for src, typ, sem, off, idx in decl:
            nf = {0: 1, 1: 2, 2: 3, 3: 4}.get(typ)      # float1..float4 only
            if nf is None or src not in bufs:
                continue
            vsize, data = bufs[src]
            if len(data) < vcount * vsize or off + 4 * nf > vsize:
                continue
            raw = np.frombuffer(data, np.uint8, vcount * vsize).reshape(vcount, vsize)
            col = np.ascontiguousarray(raw[:, off:off + 4 * nf]).view('<f4').reshape(vcount, nf)
            out[(sem, idx)] = col.astype(np.float32)
        return out

    shared = None
    subs = []
    p = nl + 1
    cid, ln = chunk(p)
    if cid != 0x3000:
        raise ValueError('missing M_MESH chunk')
    q = p + 6 + 1                                   # skip "skeletally animated" bool
    while q + 6 <= n:
        cid, ln2 = chunk(q)
        if ln2 < 6:
            break
        r = q + 6
        if cid == 0x4000:                           # submesh
            s = SubMesh()
            s.material, r = cstr(r)
            s.uses_shared = bool(b[r]); r += 1
            ic = struct.unpack_from('<I', b, r)[0]; r += 4
            i32 = b[r]; r += 1
            dt = '<u4' if i32 else '<u2'
            s.indices = np.frombuffer(b, dt, ic, r).astype(np.int64)
            r += ic * (4 if i32 else 2)
            # Some exporters write a too-short submesh length, so walk children by id.
            while r + 6 <= n:
                c3, l3 = chunk(r)
                if l3 < 6:
                    break
                if c3 == 0x5000 and not s.uses_shared:
                    vc, decl, bufs = read_geometry(r + 6, r + l3)
                    s.attrs, s.vcount = decode(vc, decl, bufs), vc
                elif c3 == 0x4010:
                    s.op = struct.unpack_from('<H', b, r + 6)[0]
                elif c3 not in (0x4100, 0x4200):
                    break
                r += l3
            subs.append(s)
            q = r
            continue
        if cid == 0x5000:                           # shared geometry
            vc, decl, bufs = read_geometry(r, q + ln2)
            shared = (vc, decode(vc, decl, bufs))
        q += ln2
    for s in subs:
        if s.uses_shared and shared:
            s.vcount, s.attrs = shared
    return version, subs


def to_triangles(idx, op):
    """Convert an index list of any OGRE operation type to a triangle list."""
    if op == 4:
        return idx[:len(idx) // 3 * 3]
    if len(idx) < 3:
        return idx[:0]
    if op == 5:                                     # triangle strip
        a, b, c = idx[:-2], idx[1:-1], idx[2:]
        flip = (np.arange(len(a)) % 2 == 1)[:, None]
        tri = np.where(flip, np.stack([b, a, c], 1), np.stack([a, b, c], 1))
    elif op == 6:                                   # triangle fan
        tri = np.stack([np.full(len(idx) - 2, idx[0]), idx[1:-1], idx[2:]], 1)
    else:
        return idx[:0]                              # points / lines: ignored
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    return tri[keep].reshape(-1)


# =============================================================================
#  OGRE .material / script parser
# =============================================================================
class Blk:
    __slots__ = ('w', 'kids')

    def __init__(self, w):
        self.w, self.kids = w, []


def parse_script(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r'//[^\n]*', '', text)
    root = Blk([]); stack = [root]; last = None
    for line in text.splitlines():
        for tok in re.findall(r'\{|\}|[^{}]+', line):
            tok = tok.strip()
            if not tok:
                continue
            if tok == '{':
                stack.append(last if last is not None else Blk([]))
                last = None
            elif tok == '}':
                if len(stack) > 1:
                    stack.pop()
                last = None
            else:
                w = [x.strip('"') for x in re.findall(r'"[^"]*"|\S+', tok)]
                last = Blk(w)
                stack[-1].kids.append(last)
    return root


def _floats(a):
    out = []
    for x in a:
        try:
            out.append(float(x))
        except ValueError:
            break
    return out


def _material_props(blk):
    d = {}

    def walk(b):
        for k in b.kids:
            if not k.w:
                walk(k); continue
            key, a = k.w[0].lower(), k.w[1:]
            if key == 'texture_unit':
                if 'tex' not in d:
                    for c in k.kids:
                        if c.w and c.w[0].lower() in ('texture', 'anim_texture') and len(c.w) > 1:
                            d['tex'] = c.w[1]
                            break
            elif key == 'scene_blend' and a:
                if a[0].lower() in ('alpha_blend', 'add', 'src_alpha'):
                    d['blend'] = True
            elif key == 'alpha_rejection' and len(a) > 1:
                try:
                    if int(a[1]) > 0:
                        d['cutoff'] = int(a[1]) / 255.0
                except ValueError:
                    pass
            elif key in ('emissive', 'self_illumination'):
                f = _floats(a)
                if f and sum(f[:3]) > 0:
                    d['emissive'] = True
            elif key == 'diffuse':
                f = _floats(a)
                if len(f) >= 3:
                    d['diffuse'] = (f + [1.0])[:4] if len(f) == 3 else f[:4]
            walk(k)
    walk(blk)
    return d


def parse_materials(paths):
    """Parse one or more .material files -> {name: {tex, blend, cutoff, emissive, diffuse}}."""
    raw = {}
    for path in paths:
        try:
            with open(path, encoding='latin1') as f:
                root = parse_script(f.read())
        except OSError:
            continue
        for blk in root.kids:
            if len(blk.w) >= 2 and blk.w[0].lower() == 'material':
                name, _, parent = ' '.join(blk.w[1:]).partition(':')
                raw[name.strip()] = (parent.strip(), _material_props(blk))
    mats = {}

    def resolve(name, seen=()):
        if name in mats:
            return mats[name]
        if name not in raw or name in seen:
            return {}
        parent, own = raw[name]
        d = dict(resolve(parent, seen + (name,))) if parent else {}
        d.update(own)
        mats[name] = d
        return d
    for name in raw:
        resolve(name)
    return mats


# =============================================================================
#  Vehicle file (.train / .truck / .car / ...)
# =============================================================================
VEHICLE_EXTS = ('.train', '.truck', '.car', '.trailer', '.load', '.airplane', '.boat')
_IDENT = re.compile(r'^[A-Za-z_]\w*$')


def _tokens(s):
    return [t for t in re.split(r'\s*,\s*|\s+', s.strip()) if t]


def _read_vehicle_lines(path, selected=None):
    """Return ({section: [lines]}, [sectionconfig names]). Honors section/end_section."""
    with open(path, encoding='latin1') as f:
        text = f.read().lstrip('\xef\xbb\xbf')
    lines = []
    for l in text.splitlines():
        s = re.split(r'\s*(?:;|//)', l, maxsplit=1)[0].strip()
        if s:
            lines.append(s)
    cfgs = []
    for s in lines:
        w = s.split()
        if w[0].lower() == 'sectionconfig' and len(w) >= 3:
            cfgs.append(' '.join(w[2:]))
    if selected:
        sel = {x.lower() for x in selected}
    else:
        sel = {cfgs[0].lower()} if cfgs else set()

    sections, cur, skip, in_comment = {}, None, False, False
    for s in lines:
        w = s.split(); k = w[0].lower()
        if in_comment:
            in_comment = k != 'end_comment'
            continue
        if k == 'comment':
            in_comment = True; continue
        if k == 'end':
            break
        if k == 'sectionconfig':
            continue
        if k == 'section':
            names = {x.lower() for x in re.split(r'[,\s]+', ' '.join(w[2:])) if x}
            skip = not (names & sel)
            continue
        if k == 'end_section':
            skip = False; continue
        if skip:
            continue
        if len(w) == 1 and _IDENT.match(s):
            cur = k; sections.setdefault(cur, []); continue
        if cur:
            sections[cur].append(s)
    return sections, cfgs


def _find_mesh_token(tokens):
    for t in tokens:
        if t.lower().endswith('.mesh'):
            return t
    return None


def parse_vehicle(path, selected=None):
    sections, cfgs = _read_vehicle_lines(path, selected)
    nodes = {}
    for sec in ('nodes', 'nodes2'):
        for s in sections.get(sec, []):
            t = _tokens(s)
            if len(t) >= 4:
                try:
                    nodes[t[0]] = np.array([float(t[1]), float(t[2]), float(t[3])])
                except ValueError:
                    pass
    managed = {}
    for s in sections.get('managedmaterials', []):
        t = _tokens(s)
        if len(t) >= 3:
            managed[t[0]] = {'type': t[1].lower(), 'tex': t[2]}
    items = []
    for sec in ('props', 'flexbodies'):
        for s in sections.get(sec, []):
            t = _tokens(s)
            mesh = _find_mesh_token(t[9:]) if len(t) >= 10 else None
            if mesh:
                try:
                    items.append(dict(kind=sec, ref=t[0], x=t[1], y=t[2],
                                      off=[float(v) for v in t[3:6]],
                                      rot=[float(v) for v in t[6:9]], mesh=mesh))
                except ValueError:
                    pass
    wheels = []
    for sec in ('meshwheels', 'meshwheels2'):
        for s in sections.get(sec, []):
            t = _tokens(s)
            mesh = _find_mesh_token(t[6:]) if len(t) >= 7 else None
            if mesh:
                wheels.append(dict(n1=t[4], n2=t[5], mesh=mesh,
                                   side=t[13] if len(t) > 13 and t[13].lower() in ('l', 'r') else 'r'))
    return nodes, managed, items, wheels, cfgs


# =============================================================================
#  Transforms
# =============================================================================
def rot_xyz(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def prop_matrix(nodes, it):
    """Same frame RoR builds for props/flexbodies: X = ref->x node, Y = normal, Z = X x Y."""
    ref = nodes[it['ref']]
    dx, dy = nodes[it['x']] - ref, nodes[it['y']] - ref
    normal = np.cross(dy, dx)
    nl, xl = np.linalg.norm(normal), np.linalg.norm(dx)
    if nl < 1e-9 or xl < 1e-9:
        return None
    normal /= nl
    xa = dx / xl
    za = np.cross(xa, normal)
    frame = np.column_stack([xa, normal, za])
    M = np.eye(4)
    M[:3, :3] = frame @ rot_xyz(*it['rot'])
    M[:3, 3] = ref + it['off'][0] * dx + it['off'][1] * dy + it['off'][2] * normal
    return M


def wheel_axis_guess(pos):
    """Wheel axis = the 'odd one out' extent (width differs from the two equal diameters).
    Also works for a whole axle (long axle + two equal wheel diameters)."""
    e = pos.max(0) - pos.min(0)
    score = [abs(e[i] - (e[(i + 1) % 3] + e[(i + 2) % 3]) / 2) for i in range(3)]
    return int(np.argmax(score))


def wheel_matrix(nodes, w, axis_local):
    a, b = nodes[w['n1']], nodes[w['n2']]
    ax = b - a
    L = np.linalg.norm(ax)
    if L < 1e-9:
        return None
    ax /= L
    if w['side'].lower().startswith('l'):
        ax = -ax
    src = np.zeros(3); src[axis_local] = 1
    v = np.cross(src, ax); c = float(src @ ax)
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            R = np.eye(3)
        else:                                       # 180 degrees about a perpendicular axis
            R = np.diag([1, -1, -1]) if axis_local == 2 else np.diag([-1, -1, 1])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + (K @ K) * (1.0 / (1.0 + c))
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = (a + b) / 2
    return M


# =============================================================================
#  glTF binary writer
# =============================================================================
class GLB:
    def __init__(s):
        s.bin = bytearray(); s.bv = []; s.acc = []; s.meshes = []; s.nodes = []
        s.mats = []; s.tex = []; s.img = []; s.matidx = {}; s.texidx = {}

    def _view(s, data, target=None):
        while len(s.bin) % 4:
            s.bin.append(0)
        d = {'buffer': 0, 'byteOffset': len(s.bin), 'byteLength': len(data)}
        s.bin += data
        if target:
            d['target'] = target
        s.bv.append(d)
        return len(s.bv) - 1

    def accessor(s, arr, ctype, comp, target, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = {'bufferView': s._view(arr.tobytes(), target), 'componentType': comp,
             'count': int(len(arr)), 'type': ctype}
        if minmax:
            a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
        s.acc.append(a)
        return len(s.acc) - 1

    def texture(s, png_bytes, key):
        if key in s.texidx:
            return s.texidx[key]
        s.img.append({'bufferView': s._view(png_bytes), 'mimeType': 'image/png', 'name': key})
        s.tex.append({'source': len(s.img) - 1, 'sampler': 0})
        s.texidx[key] = len(s.tex) - 1
        return s.texidx[key]

    def save(s, path, roots):
        g = {'asset': {'version': '2.0', 'generator': 'ror2glb'}, 'scene': 0,
             'scenes': [{'nodes': roots}], 'buffers': [{'byteLength': len(s.bin)}]}
        for key, val in (('nodes', s.nodes), ('meshes', s.meshes), ('materials', s.mats),
                         ('accessors', s.acc), ('bufferViews', s.bv)):
            if val:
                g[key] = val
        if s.tex:
            g['textures'] = s.tex; g['images'] = s.img
            g['samplers'] = [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}]
        js = json.dumps(g, separators=(',', ':')).encode()
        js += b' ' * (-len(js) % 4)
        b = bytes(s.bin); b += b'\0' * (-len(b) % 4)
        out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(b))
        out += struct.pack('<II', len(js), 0x4E4F534A) + js
        out += struct.pack('<II', len(b), 0x004E4942) + b
        with open(path, 'wb') as f:
            f.write(out)


# =============================================================================
#  Conversion
# =============================================================================
TEX_EXTS = ('.dds', '.png', '.jpg', '.jpeg', '.tga', '.bmp')


def index_files(folder):
    """Case-insensitive index of every file under folder: basename -> path."""
    idx = {}
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            idx.setdefault(f.lower(), os.path.join(root, f))
    return idx


def ask_choice(title, options, label):
    print('\n' + title)
    for i, o in enumerate(options, 1):
        print('  %d) %s' % (i, label(o)))
    while True:
        s = input('Number [1]: ').strip() or '1'
        if s.isdigit() and 1 <= int(s) <= len(options):
            return options[int(s) - 1]
        print('Please type a number from the list.')


def pick_folder():
    """Open a folder picker (falls back to typing a path if tkinter is missing)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askdirectory(title='Select the mod folder (with .train, .mesh, .material, .dds)')
        root.destroy()
        return path
    except Exception:
        return input('Paste the path of the mod folder (or .zip): ').strip().strip('"')


def convert(folder, vehicle=None, out=None, configs=None, wheels_on=True,
            wheel_axis='auto', max_tex=0, textures=True, list_configs=False,
            interactive=False):
    files = index_files(folder)
    if vehicle:
        vpath = vehicle if os.path.exists(vehicle) else files.get(os.path.basename(vehicle).lower())
        cands = [vpath] if vpath else []
    else:
        cands = sorted((p for n, p in files.items() if n.endswith(VEHICLE_EXTS)),
                       key=lambda p: (p.count(os.sep), p))
    if not cands:
        sys.exit('error: no vehicle file (%s) found' % '/'.join(VEHICLE_EXTS))
    train = cands[0]
    if len(cands) > 1:
        if interactive:
            train = ask_choice('Several vehicle files found, which one?', cands, os.path.basename)
        else:
            print('note: several vehicle files found, using %s (others: %s)' %
                  (os.path.basename(train), ', '.join(os.path.basename(c) for c in cands[1:])))

    nodes, managed, items, wheels, cfgs = parse_vehicle(train, configs)
    if interactive and len(cfgs) > 1 and not configs and not list_configs:
        configs = [ask_choice('This vehicle has several configurations, which one?', cfgs, str)]
        nodes, managed, items, wheels, cfgs = parse_vehicle(train, configs)
    if list_configs:
        print('configs:', ', '.join(cfgs) if cfgs else '(none)')
        return None
    if cfgs:
        print('config:', ', '.join(configs) if configs else cfgs[0], '| available:', ', '.join(cfgs))

    mats = parse_materials([p for n, p in files.items() if n.endswith('.material')])
    mats_ci = {k.lower(): v for k, v in mats.items()}
    for k, v in managed.items():
        mats.setdefault(k, {'tex': v['tex'], 'blend': 'transparent' in v['type']})
        mats_ci.setdefault(k.lower(), mats[k])

    glb = GLB()
    missing, bad_tex, bad_mesh = set(), set(), {}
    png_cache, mesh_cache = {}, {}

    def find_file(name):
        name = os.path.basename(name.replace('\\', '/')).lower()
        if name in files:
            return files[name]
        stem = os.path.splitext(name)[0]
        for e in TEX_EXTS:
            if stem + e in files:
                return files[stem + e]
        return None

    def tex_png(name):
        if name in png_cache:
            return png_cache[name]
        res = None
        p = find_file(name)
        if not p:
            missing.add(name)
        else:
            try:
                with Image.open(p) as im:
                    im = im.convert('RGBA')
                    if max_tex and max(im.size) > max_tex:
                        im.thumbnail((max_tex, max_tex), Image.LANCZOS)
                    bio = io.BytesIO(); im.save(bio, 'PNG'); res = bio.getvalue()
            except Exception:
                bad_tex.add(name)
        png_cache[name] = res
        return res

    def material(name):
        if name in glb.matidx:
            return glb.matidx[name]
        d = mats.get(name) or mats_ci.get(name.lower()) or {}
        pbr = {'metallicFactor': 0.0, 'roughnessFactor': 0.8}
        m = {'name': name, 'doubleSided': True, 'pbrMetallicRoughness': pbr}
        png = tex_png(d['tex']) if (textures and d.get('tex')) else None
        if png:
            t = glb.texture(png, d['tex'])
            pbr['baseColorTexture'] = {'index': t}
            if d.get('emissive'):
                m['emissiveTexture'] = {'index': t}; m['emissiveFactor'] = [1, 1, 1]
        else:
            pbr['baseColorFactor'] = d.get('diffuse') or [0.6, 0.6, 0.6, 1.0]
        if d.get('blend'):
            m['alphaMode'] = 'BLEND'
        elif d.get('cutoff') is not None:
            m['alphaMode'] = 'MASK'; m['alphaCutoff'] = d['cutoff']
        glb.mats.append(m)
        glb.matidx[name] = len(glb.mats) - 1
        return glb.matidx[name]

    def load_mesh(meshname):
        """-> (glb mesh index, positions) or None."""
        key = meshname.lower()
        if key in mesh_cache:
            return mesh_cache[key]
        mesh_cache[key] = None
        p = find_file(meshname)
        if not p:
            missing.add(meshname); return None
        try:
            _, subs = read_mesh(p)
        except Exception as e:
            bad_mesh[meshname] = str(e); return None
        prims, allpos = [], []
        for s in subs:
            P = s.attrs.get((SEM_POSITION, 0)) if s.attrs else None
            if P is None or P.shape[1] < 3:
                continue
            tri = to_triangles(s.indices, s.op)
            if len(tri) == 0:
                continue
            if tri.max() >= len(P) or tri.min() < 0:
                bad_mesh[meshname] = 'submesh has out-of-range indices (skipped)'
                continue
            P = P[:, :3]
            att = {'POSITION': glb.accessor(P, 'VEC3', 5126, 34962, True)}
            N = s.attrs.get((SEM_NORMAL, 0))
            if N is not None and N.shape[1] >= 3:
                att['NORMAL'] = glb.accessor(N[:, :3], 'VEC3', 5126, 34962)
            T = s.attrs.get((SEM_TEXCOORD, 0))
            if T is not None and T.shape[1] >= 2:
                att['TEXCOORD_0'] = glb.accessor(T[:, :2], 'VEC2', 5126, 34962)
            idx = glb.accessor(tri.astype(np.uint32), 'SCALAR', 5125, 34963)
            prims.append({'attributes': att, 'indices': idx, 'material': material(s.material)})
            allpos.append(P)
        if not prims:
            return None
        glb.meshes.append({'name': os.path.splitext(meshname)[0], 'primitives': prims})
        mesh_cache[key] = (len(glb.meshes) - 1, np.vstack(allpos))
        return mesh_cache[key]

    groups = {'props': [], 'flexbodies': [], 'wheels': []}

    def add_node(group, meshname, M, label):
        mi = load_mesh(meshname)
        if mi is None or M is None:
            return False
        glb.nodes.append({'name': label, 'mesh': mi[0], 'matrix': M.T.flatten().tolist()})
        groups[group].append(len(glb.nodes) - 1)
        return True

    skipped = 0
    for it in items:
        if not all(it[k] in nodes for k in ('ref', 'x', 'y')):
            skipped += 1; continue
        if not add_node(it['kind'], it['mesh'], prop_matrix(nodes, it), os.path.splitext(it['mesh'])[0]):
            pass
    if wheels_on:
        for w in wheels:
            if w['n1'] not in nodes or w['n2'] not in nodes:
                skipped += 1; continue
            mi = load_mesh(w['mesh'])
            if mi is None:
                continue
            ax = 'xyz'.index(wheel_axis) if wheel_axis in ('x', 'y', 'z') else wheel_axis_guess(mi[1])
            add_node('wheels', w['mesh'], wheel_matrix(nodes, w, ax), 'wheel_' + os.path.splitext(w['mesh'])[0])

    roots = []
    for g, ids in groups.items():
        if ids:
            glb.nodes.append({'name': g, 'children': ids})
            roots.append(len(glb.nodes) - 1)
    if not roots:
        sys.exit('error: nothing to export (no usable .mesh files found for this vehicle file)')

    out = out or os.path.join(folder, os.path.splitext(os.path.basename(train))[0] + '.glb')
    glb.save(out, roots)
    nparts = sum(len(v) for v in groups.values())
    print('wrote %s - %d parts, %d meshes, %d materials, %d textures' %
          (out, nparts, len(glb.meshes), len(glb.mats), len(glb.tex)))
    if skipped:
        print('skipped %d part(s) that reference unknown nodes' % skipped)
    if missing:
        print('files not found (skipped or untextured):', ', '.join(sorted(missing)))
    if bad_tex:
        print('textures that could not be decoded (Pillow lacks that DDS format?):', ', '.join(sorted(bad_tex)))
    for k, v in bad_mesh.items():
        print('mesh problem in %s: %s' % (k, v))
    return out


def main():
    ap = argparse.ArgumentParser(description='Rigs of Rods mod -> glTF (.glb) for Blender')
    ap.add_argument('folder', nargs='?', help='mod folder or .zip (omit to pick a folder in a window)')
    ap.add_argument('vehicle', nargs='?', help='vehicle file name (.train/.truck/...), optional')
    ap.add_argument('-o', '--out')
    ap.add_argument('--config', action='append', help='sectionconfig name (repeatable)')
    ap.add_argument('--list-configs', action='store_true')
    ap.add_argument('--no-wheels', action='store_true')
    ap.add_argument('--wheel-axis', default='auto', choices=['auto', 'x', 'y', 'z'])
    ap.add_argument('--max-tex', type=int, default=0)
    ap.add_argument('--no-textures', action='store_true')
    a = ap.parse_args()

    interactive = a.folder is None          # started by double-click / no arguments
    src = a.folder or pick_folder()
    if not src:
        print('No folder selected.')
        if interactive:
            input('Press Enter to close...')
        return
    tmp = None
    try:
        if os.path.isfile(src) and zipfile.is_zipfile(src):
            tmp = tempfile.mkdtemp(prefix='ror2glb_')
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            folder = tmp
            out = a.out or os.path.splitext(os.path.abspath(src))[0] + '.glb'
        elif os.path.isdir(src):
            folder, out = src, a.out
        else:
            sys.exit('error: %s is not a folder or zip file' % src)
        convert(folder, a.vehicle, out, a.config, not a.no_wheels, a.wheel_axis,
                a.max_tex, not a.no_textures, a.list_configs, interactive)
    except SystemExit as e:
        if interactive and e.code:
            print(e.code)
            input('Press Enter to close...')
            return
        raise
    except Exception:
        if interactive:
            import traceback
            traceback.print_exc()
            input('Press Enter to close...')
            return
        raise
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if interactive:
        input('\nDone. In Blender: File > Import > glTF 2.0. Press Enter to close...')


if __name__ == '__main__':
    main()

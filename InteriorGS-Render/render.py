'''
ADOBE CONFIDENTIAL
Copyright 2026 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
'''

import bpy
import os
import sys
import math
import json
import numpy as np
import time
import random
import os
import subprocess
from datetime import datetime
from tqdm import tqdm
import urllib.request
import urllib.error
import shutil
from scipy.spatial.transform import Rotation as R_scipy


import multiprocessing
from multiprocessing.pool import Pool as BasePool
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass


addon_name = "blender-addon"
addon_path = "gaussian-splatting-blender-addon/blender-addon.zip"


HF_REPO = "spatialverse/InteriorGS"
HF_TOKEN = ""


dataset_base_path = "datasets"
output_base_dir = "output"


local_path_dir = "datasets"


PROCESS_LOG_FILE = "process.log"


panorama_type = 'EQUIRECTANGULAR'


fps = 30
target_speed = 0.02
min_keyframe_distance = target_speed


target_frames = 800
min_frames_for_backward = 800




class NoDaemonProcess(multiprocessing.Process):

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(BasePool):

    def Process(self, *args, **kwds):
        proc = super().Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


def parse_sys_argv():


    if '--' in sys.argv:
        script_args = sys.argv[sys.argv.index('--') + 1:]
    else:
        script_args = []


    config = {
        'datasets': [],
        'dataset_file': None,
        'base_path': dataset_base_path,
        'output_dir': output_base_dir,
        'target_frames': target_frames,
        'dry_run': False,
        'hf_token': HF_TOKEN,
        'hf_repo': HF_REPO,
        'splits': [],
        'total_split': None
    }


    i = 0
    while i < len(script_args):
        arg = script_args[i]

        if arg == '--datasets':

            i += 1
            while i < len(script_args) and not script_args[i].startswith('--'):
                config['datasets'].append(script_args[i])
                i += 1
            continue

        elif arg == '--dataset-file':
            if i + 1 < len(script_args):
                config['dataset_file'] = script_args[i + 1]
                i += 2
            else:
                print(f"Error: --dataset-file requires an argument")
                sys.exit(1)
            continue

        elif arg == '--base-path':
            if i + 1 < len(script_args):
                config['base_path'] = script_args[i + 1]
                i += 2
            else:
                print(f"Error: --base-path requires an argument")
                sys.exit(1)
            continue

        elif arg == '--output-dir':
            if i + 1 < len(script_args):
                config['output_dir'] = script_args[i + 1]
                i += 2
            else:
                print(f"Error: --output-dir requires an argument")
                sys.exit(1)
            continue

        elif arg == '--target-frames':
            if i + 1 < len(script_args):
                try:
                    config['target_frames'] = int(script_args[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: --target-frames must be an integer")
                    sys.exit(1)
            else:
                print(f"Error: --target-frames requires an argument")
                sys.exit(1)
            continue

        elif arg == '--dry-run':
            config['dry_run'] = True
            i += 1
            continue

        elif arg == '--hf-token':
            if i + 1 < len(script_args):
                config['hf_token'] = script_args[i + 1]
                i += 2
            else:
                print(f"Error: --hf-token requires an argument")
                sys.exit(1)
            continue

        elif arg == '--split':

            i += 1
            while i < len(script_args) and not script_args[i].startswith('--'):
                try:
                    split_num = int(script_args[i])
                    config['splits'].append(split_num)
                    i += 1
                except ValueError:
                    print(f"Error: --split must be an integer, got: {script_args[i]}")
                    sys.exit(1)

            if not config['splits']:
                print(f"Error: --split requires at least one argument")
                sys.exit(1)
            continue

        elif arg == '--total-split':
            if i + 1 < len(script_args):
                try:
                    config['total_split'] = int(script_args[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: --total-split must be an integer")
                    sys.exit(1)
            else:
                print(f"Error: --total-split requires an argument")
                sys.exit(1)
            continue

        elif arg == '--help' or arg == '-h':
            print_help()
            sys.exit(0)

        else:
            print(f"Warning: unknown argument '{arg}' will be ignored")
            i += 1


    if config['dataset_file']:
        try:
            with open(config['dataset_file'], 'r') as f:
                datasets = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            config['datasets'] = datasets
            print(f"✓ Read {len(datasets)} datasets from file {config['dataset_file']}")
        except Exception as e:
            print(f"Error: unable to read dataset file {config['dataset_file']}: {e}")
            sys.exit(1)


    if not config['datasets']:
        print("Error: no datasets specified")
        print("Use --help to see usage information")
        sys.exit(1)

    return config


def print_help():

    help_text = """
Blender 3DGS batch renderer - supports downloading datasets from Hugging Face

Usage:
  blender --background --python script.py -- [options]

Note: you must use -- to separate Blender arguments from script arguments

Options:
  --datasets ID1 ID2 ...        Dataset ID list, separated by spaces
  --dataset-file FILE           Path to a text file containing dataset IDs (one ID per line)
  --base-path PATH              Local dataset cache path (default: /mnt/localssd/tmp/blender_datasets)
  --output-dir PATH             Base output directory (default: /mnt/localssd/output)
  --target-frames N             Target video frame count (default: 800)
  --hf-token TOKEN              Hugging Face access token (required for private repos)
  --split N1 N2 ...             Split ID list (supports multiple values for multiprocessing)
  --total-split N               Total number of splits (for multi-server parallelism)
  --dry-run                     Only validate dataset files, do not render
  --help, -h                    Show this help message

Examples:
  # Render a single dataset (automatically downloaded from HF)
  blender --background --python script.py -- --datasets 0001_839920

  # Render multiple datasets
  blender --background --python script.py -- --datasets 0001_839920 0002_123456 0003_789012

  # Use a Hugging Face token (private repo)
  blender --background --python script.py -- --datasets 0001_839920 --hf-token hf_xxxxx

  # Read the dataset list from a file
  blender --background --python script.py -- --dataset-file datasets.txt

  # Parallel processing with multiple processes (process splits 1 and 2 out of 200 total)
  blender --background --python script.py -- --dataset-file datasets.txt --split 1 2 --total-split 200

  # Validation mode (no rendering)
  blender --background --python script.py -- --datasets 0001_839920 --dry-run

  # Custom configuration
  blender --background --python script.py -- \\
    --datasets 0001_839920 \\
    --target-frames 1000 \\
    --output-dir /path/to/output \\
    --base-path /custom/cache/dir

Bash script example:
  # Create a dataset list file
  echo "0001_839920" > datasets.txt
  echo "0002_123456" >> datasets.txt

  # Run rendering (datasets will be downloaded automatically)
  blender --background --python script.py -- --dataset-file datasets.txt

Notes:
  - Datasets will be downloaded automatically from https://huggingface.co/datasets/spatialverse/InteriorGS
  - You need to install first: npm install -g @playcanvas/splat-transform
  - Downloaded datasets will be cached under the directory specified by --base-path
  - A video will be generated automatically after rendering
"""
    print(help_text)


def install_and_enable_addon(addon_path, addon_name):

    try:

        if addon_name in bpy.context.preferences.addons.keys():
            print(f"✓ Add-on '{addon_name}' is already enabled, skipping installation")
            return


        delay = random.uniform(60, 80)
        print(f"Waiting {delay:.2f} seconds to avoid multiprocessing conflicts...")
        time.sleep(delay)

        print(f"Installing add-on: {addon_path}")


        try:
            bpy.ops.preferences.addon_install(overwrite=True, filepath=addon_path)
        except Exception as install_error:

            error_msg = str(install_error)
            if "File exists" in error_msg or "already exists" in error_msg.lower():
                print(f"⚠ Add-on directory already exists (possibly created by another process), continuing to enable the add-on...")
            else:
                raise

        print(f"Enabling add-on: {addon_name}")
        bpy.ops.preferences.addon_enable(module=addon_name)
        print(f"✓ Add-on '{addon_name}' installed and enabled successfully.")

    except Exception as e:

        if addon_name in bpy.context.preferences.addons.keys():
            print(f"✓ Add-on '{addon_name}' is already usable even though installation reported an error")
            return

        print(f"✗ Add-on setup failed: {e}", file=sys.stderr)
        raise


def setup_gpu_rendering():
    print("=" * 50)
    print("Configuring Cycles rendering (automatically selecting available devices)...")

    try:
        bpy.ops.preferences.addon_enable(module='cycles')
        print("✓ Cycles add-on enabled")
    except:
        print("⚠ Cycles add-on may already be enabled")

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except KeyError:
        print("✗ Error: unable to access Cycles preferences, falling back to CPU rendering.")
        scene.cycles.device = 'CPU'
        return


    prefs.compute_device_type = 'CUDA'
    try:
        prefs.refresh_devices()
    except AttributeError:
        try:
            prefs.get_devices()
        except:
            pass

    cuda_devs = [d for d in prefs.devices if d.type == 'CUDA']

    if cuda_devs:
        for device in prefs.devices:
            device.use = (device.type == 'CUDA')
        scene.cycles.device = 'GPU'
        print(f"✓ Enabled {len(cuda_devs)} CUDA device(s) for rendering.")
    else:
        print("⚠ No CUDA device found, automatically switching to CPU rendering.")
        for device in prefs.devices:
            device.use = (device.type == 'CPU')
        scene.cycles.device = 'CPU'

    print(f"\n✓ Final device in use: {scene.cycles.device}")
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    print("=" * 50 + "\n")


def verify_gpu_usage():

    scene = bpy.context.scene
    print("\n" + "=" * 50)
    print("Final render configuration check:")
    print(f"  Render engine: {scene.render.engine}")
    print(f"  Cycles device: {scene.cycles.device}")

    prefs = bpy.context.preferences.addons['cycles'].preferences
    print(f"  Compute device type: {prefs.compute_device_type}")

    print(f"\n  Enabled devices (use=True):")
    for dev in prefs.devices:
        if dev.use:
            print(f"    ✓ {dev.name} ({dev.type})")

    print(f"\n  Disabled devices (use=False):")
    for dev in prefs.devices:
        if not dev.use:
            print(f"    ✗ {dev.name} ({dev.type})")

    cpu_enabled = any(d.use and d.type == 'CPU' for d in prefs.devices)
    if cpu_enabled:
        print("\n  ⚠ Warning: CPU device is still enabled!")

    cuda_enabled = any(d.use and d.type == 'CUDA' for d in prefs.devices)
    if not cuda_enabled:
        print("\n  ⚠ Warning: no CUDA device is enabled!")

    print("=" * 50 + "\n")


def clear_scene():

    print("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    print("Scene cleared.")


def import_gaussian_splatting(ply_filepath):

    try:
        print(f"Importing 3DGS file: {ply_filepath}")
        bpy.ops.object.import_gaussian_splatting(filepath=ply_filepath)
        print(f"✓ Successfully imported PLY file: {ply_filepath}")
    except Exception as e:
        print(f"✗ Import failed: {e}", file=sys.stderr)
        raise

    obj = bpy.context.active_object

    if obj is None or "gaussian_splatting" not in obj:
        raise Exception("Gaussian Splatting object not found")

    print(f"✓ Found object: {obj.name}")

    geo_mod = obj.modifiers.get("Geometry Nodes")
    if geo_mod is None:
        raise Exception("Geometry Nodes modifier not found")

    geo_tree = geo_mod.node_group

    print("\nGeometry node info:")
    print(f"  Node tree: {geo_tree.name}")
    print(f"  Node count: {len(geo_tree.nodes)}")

    if hasattr(geo_mod, 'execution_mode'):
        print(f"  Execution mode: {geo_mod.execution_mode}")

    boolean_node = geo_tree.nodes.get("Boolean")
    if boolean_node:
        boolean_node.boolean = False
        print("✓ Point-cloud mode disabled (using ellipsoids)")
    else:
        print("⚠ Warning: Boolean node not found")

    random_value_node = geo_tree.nodes.get("Random Value")
    if random_value_node and "Probability" in random_value_node.inputs:
        random_value_node.inputs["Probability"].default_value = 1.0
        print("✓ Display percentage set to 100%")
    else:
        print("⚠ Warning: Random Value node or Probability input not found")

    return obj


def load_and_transform_path(json_path):

    print(f"Reading path file: {json_path}")

    try:
        with open(json_path, 'r') as f:
            path_data = json.load(f)
    except Exception as e:
        print(f"✗ Failed to read path file: {e}", file=sys.stderr)
        raise


    transformed_points = []
    for i, point in enumerate(path_data):
        x = point['x']
        y = point['y']
        z = 1.4


        blender_coords = (x, z, -y)
        transformed_points.append(blender_coords)

    print(f"✓ Successfully loaded {len(transformed_points)} path points")
    return transformed_points


def calculate_distance(p1, p2):

    return math.sqrt(
        (p2[0] - p1[0])**2 +
        (p2[1] - p1[1])**2 +
        (p2[2] - p1[2])**2
    )


def select_keyframe_points(points, target_speed, min_distance, fps):

    if len(points) == 0:
        return []


    if len(points) == 1:
        print(f"\n⚠ Warning: the path file contains only one point; a static video will be created from it")
        return [(0, points[0])]

    selected = [(0, points[0])]
    accumulated_distance = 0.0

    print(f"\nKeyframe selection parameters:")
    print(f"  Target speed: {target_speed} meters/frame")
    print(f"  Minimum keyframe distance: {min_distance} meters")
    print(f"  Frame rate: {fps} fps")

    for i in range(1, len(points)):

        segment_distance = calculate_distance(points[i-1], points[i])
        accumulated_distance += segment_distance


        if accumulated_distance >= min_distance:
            selected.append((i, points[i]))
            accumulated_distance = 0.0
            print(f"  Selected point {i}: accumulated distance reached the threshold")


    if selected[-1][0] != len(points) - 1:
        selected.append((len(points) - 1, points[-1]))
        print(f"  Added endpoint {len(points) - 1}")

    print(f"\n✓ Selected {len(selected)} keyframe points from {len(points)} points")
    print(f"  Skipped {len(points) - len(selected)} overly dense points")

    return selected


def prepare_animation_path(selected_points, target_frames, min_frames_for_backward, target_speed):

    if len(selected_points) == 0:
        return [], 0
    if len(selected_points) == 1:

        return selected_points, target_frames

    def seg_frames(p0, p1):
        dist = calculate_distance(p0, p1)
        return int(dist / target_speed)


    forward_frames = 0
    new_selected = [selected_points[0]]
    for i in range(len(selected_points) - 1):
        if seg_frames(selected_points[i][1], selected_points[i+1][1]) >= 1:
            forward_frames += seg_frames(selected_points[i][1], selected_points[i+1][1])
            new_selected.append(selected_points[i+1])

    full_points = new_selected.copy()
    full_frames = forward_frames
    while full_frames < min_frames_for_backward:
        new_selected.reverse()
        full_points.extend(new_selected[1:])
        full_frames += forward_frames

    print(full_points)

    return full_points, full_frames


def setup_camera_animation(animation_points, fps, target_speed, dataset_id, target_frames_param=800, is_static=False):

    print(f"\nSetting up camera animation...")

    if len(animation_points) == 0:
        print("⚠ Warning: animation point count is 0, skipping")
        return None, 0


    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add(location=animation_points[0][1], rotation=(0, 0, 0))
        camera = bpy.data.objects["Camera"]

    camera_data = camera.data


    camera_data.type = 'PANO'
    camera_data.panorama_type = panorama_type
    camera.data.latitude_min = math.radians(-90)
    camera.data.latitude_max = math.radians(90)
    camera.data.longitude_min = math.radians(-180)
    camera.data.longitude_max = math.radians(180)


    bpy.context.scene.camera = camera


    if camera.animation_data:
        camera.animation_data_clear()


    if is_static or len(animation_points) == 1:
        print(f"  Creating static video")
        camera.location = animation_points[0][1]
        camera.keyframe_insert(data_path="location", frame=1)
        camera.keyframe_insert(data_path="location", frame=target_frames_param)

        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = target_frames_param

        print(f"\n✓ Static camera animation setup complete")
        print(f"  Position: {animation_points[0][1]}")
        print(f"  Total frames: {target_frames_param}")
        print(f"  Total duration: {target_frames_param / fps:.2f} seconds")

        return camera, target_frames_param


    current_frame = 1

    print(f"  Keyframe setup:")
    print(f"    Frame {current_frame}: source point {animation_points[0][0]}, position {animation_points[0][1]}")

    camera.location = animation_points[0][1]
    camera.keyframe_insert(data_path="location", frame=current_frame)

    for i in range(1, len(animation_points)):
        dist = calculate_distance(animation_points[i-1][1], animation_points[i][1])
        frames = max(1, int(dist / target_speed))

        current_frame += frames

        if (current_frame > target_frames_param):
            print(f"    Reached target frame count {target_frames_param}, stopping further keyframe insertion")
            camera.location = tuple((target_frames_param - current_frame) * target_speed / dist * (np.array(animation_points[i][1]) - np.array(animation_points[i-1][1])) + np.array(animation_points[i-1][1]))
            camera.keyframe_insert(data_path="location", frame=target_frames_param)
            break

        camera.location = animation_points[i][1]
        camera.keyframe_insert(data_path="location", frame=current_frame)

        print(f"    Frame {current_frame}: source point {animation_points[i][0]}, position {animation_points[i][1]} (distance: {dist:.3f}m, frames: {frames})")


    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = target_frames_param


    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


    if current_frame > target_frames_param:
        bpy.context.scene.frame_end = target_frames_param
        current_frame = target_frames_param
    elif current_frame < target_frames_param:
        camera.keyframe_insert(data_path="location", frame=target_frames_param)
        bpy.context.scene.frame_end = target_frames_param
        current_frame = target_frames_param

    print(f"\n✓ Camera animation setup complete")
    print(f"  Total frames: {current_frame}")
    print(f"  Total duration: {current_frame / fps:.2f} seconds")

    return camera, current_frame


def setup_lighting():

    print("Setting up lighting...")
    world = bpy.context.scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Strength'].default_value = 1.0
        bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    print("✓ Lighting configured")


def setup_render_settings():

    scene = bpy.context.scene

    scene.render.resolution_x = 2880
    scene.render.resolution_y = 1440
    scene.render.resolution_percentage = 100
    scene.render.fps = fps


    scene.render.threads_mode = 'AUTO'
    cpu_count = os.cpu_count()
    print(f"✓ Render thread mode: AUTO (system has {cpu_count} CPU cores)")

    print(f"✓ Render resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")
    print(f"✓ Frame rate: {scene.render.fps} fps")

    scene.cycles.samples = 4
    scene.cycles.adaptive_sampling_threshold = 0.05
    scene.cycles.adaptive_min_samples = 8
    scene.cycles.use_denoising = True

    print(f"✓ Cycles samples: {scene.cycles.samples}")


def download_and_prepare_dataset(dataset_id, base_path, hf_repo, hf_token=None):

    dataset_dir = os.path.join(base_path, dataset_id)
    compressed_ply = os.path.join(dataset_dir, "3dgs_compressed.ply")
    decompressed_ply = os.path.join(dataset_dir, "3dgs_uncompressed.ply")
    path_json = os.path.join(dataset_dir, "path.json")


    if os.path.exists(decompressed_ply) and os.path.exists(path_json):
        print(f"✓ Dataset {dataset_id} already exists in local cache")
        return True


    os.makedirs(dataset_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting dataset download: {dataset_id}")
    print(f"{'='*60}")


    if not os.path.exists(compressed_ply):
        filename = "3dgs_compressed.ply"
        url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{dataset_id}/{filename}"

        print(f"Downloading: {filename}")
        print(f"  URL: {url}")

        try:

            request = urllib.request.Request(url)
            if hf_token:
                request.add_header("Authorization", f"Bearer {hf_token}")


            with urllib.request.urlopen(request) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 8192

                with open(compressed_ply, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)


                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r  Progress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')

                print()
                print(f"✓ Download complete: {filename}")

        except urllib.error.HTTPError as e:
            print(f"\n✗ Download failed ({filename}): HTTP {e.code} - {e.reason}")
            if e.code == 401:
                print("  Tip: you may need to provide --hf-token to access a private repository")
            return False
        except Exception as e:
            print(f"\n✗ Download failed ({filename}): {e}")
            return False


    if not os.path.exists(path_json):

        local_path_json = os.path.join(local_path_dir, dataset_id, "path.json")

        if os.path.exists(local_path_json):
            print(f"✓ Copied path.json from local source: {local_path_json}")
            shutil.copy2(local_path_json, path_json)
        else:

            filename = "path.json"
            url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{dataset_id}/{filename}"

            print(f"Downloading: {filename}")
            print(f"  URL: {url}")

            try:

                request = urllib.request.Request(url)
                if hf_token:
                    request.add_header("Authorization", f"Bearer {hf_token}")


                with urllib.request.urlopen(request) as response:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    chunk_size = 8192

                    with open(path_json, 'wb') as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)


                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\r  Progress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')

                    print()
                    print(f"✓ Download complete: {filename}")

            except urllib.error.HTTPError as e:
                print(f"\n✗ Download failed ({filename}): HTTP {e.code} - {e.reason}")
                print(f"  It was also not found locally: {local_path_json}")
                return False
            except Exception as e:
                print(f"\n✗ Download failed ({filename}): {e}")
                print(f"  It was also not found locally: {local_path_json}")
                return False


    if not os.path.exists(decompressed_ply):
        print(f"\nDecompressing 3DGS file...")
        print(f"  Input: {compressed_ply}")
        print(f"  Output: {decompressed_ply}")

        try:

            result = subprocess.run(
                ["splat-transform", compressed_ply, decompressed_ply],
                capture_output=True,
                text=True,
                check=True
            )

            print(f"✓ Decompression complete")


        except subprocess.CalledProcessError as e:
            print(f"\n✗ Decompression failed:")
            print(f"  Return code: {e.returncode}")
            print(f"  Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"\n✗ Error: splat-transform command not found")
            print(f"  Please make sure it is installed: npm install -g @playcanvas/splat-transform")
            return False
        except Exception as e:
            print(f"\n✗ Decompression failed: {e}")
            return False

    print(f"\n{'='*60}")
    print(f"✓ Dataset {dataset_id} is ready")
    print(f"{'='*60}\n")

    return True


def blender_to_original_location(blender_location):

    x_b, y_b, z_b = blender_location
    return np.array([x_b, -z_b, y_b])


def blender_to_original_rotation(blender_rotation_euler):


    rot_blender = R_scipy.from_euler('xyz', blender_rotation_euler, degrees=False)
    R_blender = rot_blender.as_matrix()


    coord_transform = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])


    coord_transform_inv = coord_transform.T


    R_original = coord_transform_inv @ R_blender @ coord_transform


    rot_original = R_scipy.from_matrix(R_original)
    euler_original = rot_original.as_euler('xyz', degrees=False)

    return euler_original


def get_3x4_RT_matrix_from_blender_original_coords(camera):


    location, rotation = camera.matrix_world.decompose()[:2]


    location_original = blender_to_original_location(location)


    R_blender = np.array(rotation.to_matrix())


    coord_transform = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])


    R_original = coord_transform.T @ R_blender @ coord_transform


    t_original = -R_original @ location_original

    return R_original, t_original, location_original


def get_3x4_RT_matrix_from_blender(camera):


    R_BlenderView_to_OpenCVView = np.diag([1, -1, -1])

    location, rotation = camera.matrix_world.decompose()[:2]
    R_BlenderView = np.array(rotation.to_matrix().transposed())

    T_BlenderView = -1.0 * R_BlenderView @ np.array(location)

    R_OpenCV = R_BlenderView_to_OpenCVView @ R_BlenderView
    T_OpenCV = R_BlenderView_to_OpenCVView @ T_BlenderView

    return R_OpenCV, T_OpenCV


def get_transform_matrix_from_blender(camera):


    R_world_to_cam, t_world_to_cam, location_world = get_3x4_RT_matrix_from_blender_original_coords(camera)


    R_cam_to_world = R_world_to_cam.T
    t_cam_to_world = -R_cam_to_world @ t_world_to_cam


    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_cam_to_world
    transform_matrix[:3, 3] = t_cam_to_world

    return transform_matrix


def rvec_from_rotation_matrix(R):

    rot = R_scipy.from_matrix(R)
    rvec = rot.as_rotvec()
    return rvec


camera_positions_data = {}


render_progress_bar = None


def evaluate_camera_at_frame(camera, frame):


    blender_location = np.array([camera.location.x, camera.location.y, camera.location.z])
    blender_rotation = np.array([camera.rotation_euler.x, camera.rotation_euler.y, camera.rotation_euler.z])


    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:

            if fcurve.data_path == 'location':
                value = fcurve.evaluate(frame)
                if fcurve.array_index == 0:
                    blender_location[0] = value
                elif fcurve.array_index == 1:
                    blender_location[1] = value
                elif fcurve.array_index == 2:
                    blender_location[2] = value


            elif fcurve.data_path == 'rotation_euler':
                value = fcurve.evaluate(frame)
                if fcurve.array_index == 0:
                    blender_rotation[0] = value
                elif fcurve.array_index == 1:
                    blender_rotation[1] = value
                elif fcurve.array_index == 2:
                    blender_rotation[2] = value

    return blender_location, blender_rotation


def compute_RT_from_location_rotation(blender_location, blender_rotation):


    location_original = blender_to_original_location(blender_location)


    rot_blender = R_scipy.from_euler('xyz', blender_rotation, degrees=False)
    R_blender = rot_blender.as_matrix()


    coord_transform = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ])


    R_original = coord_transform.T @ R_blender @ coord_transform


    t_original = -R_original @ location_original

    return R_original, t_original, location_original


def save_camera_position_for_frame(camera, scene, dataset_id, frame):

    global camera_positions_data


    if dataset_id not in camera_positions_data:
        camera_positions_data[dataset_id] = {}


    image_name = f"pano_camera0/frame_{frame:04d}.png"

    try:

        blender_location, blender_rotation = evaluate_camera_at_frame(camera, frame)


        original_location = blender_to_original_location(blender_location)
        original_rotation = blender_to_original_rotation(blender_rotation)


        R_world_to_cam, t_world_to_cam, location_world = compute_RT_from_location_rotation(
            blender_location, blender_rotation
        )


        R_cam_to_world = R_world_to_cam.T
        t_cam_to_world = -R_cam_to_world @ t_world_to_cam

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_cam_to_world
        transform_matrix[:3, 3] = t_cam_to_world
        transform_matrix = transform_matrix[[1, 2, 0, 3]]


        rvec = rvec_from_rotation_matrix(R_world_to_cam)


        width = scene.render.resolution_x
        height = scene.render.resolution_y


        camera_positions_data[dataset_id][image_name] = {
            "image_id": frame,
            "image_name": image_name,
            "camera_id": 0,
            "width": width,
            "height": height,

            "R": R_world_to_cam.tolist(),
            "t": t_world_to_cam.tolist(),
            "rvec": rvec.tolist(),
            "transform_matrix": transform_matrix.tolist(),

            "location": {
                "x": float(original_location[0]),
                "y": float(original_location[1]),
                "z": float(original_location[2])
            },
            "rotation": {
                "x": float(original_rotation[0]),
                "y": float(original_rotation[1]),
                "z": float(original_rotation[2])
            },

            "blender_location": {
                "x": float(blender_location[0]),
                "y": float(blender_location[1]),
                "z": float(blender_location[2])
            },
            "blender_rotation": {
                "x": float(blender_rotation[0]),
                "y": float(blender_rotation[1]),
                "z": float(blender_rotation[2])
            },
            "frame": frame,
            "dataset_id": dataset_id
        }

    except Exception as e:
        print(f"⚠ Warning: failed to save camera position for frame {frame}: {e}")
        import traceback
        traceback.print_exc()


def save_all_camera_positions(camera, scene, dataset_id):

    global camera_positions_data


    if dataset_id not in camera_positions_data:
        camera_positions_data[dataset_id] = {}
    else:
        camera_positions_data[dataset_id] = {}

    print(f"\n======== Start recording camera positions for dataset {dataset_id} ========")


    start_frame = scene.frame_start
    end_frame = scene.frame_end
    total_frames = end_frame - start_frame + 1

    print(f"  Frame range: {start_frame} - {end_frame} ({total_frames} frames total)")
    print(f"  Using fcurve evaluation (no need to change the scene frame)")


    for frame in tqdm(range(start_frame, end_frame + 1),
                     desc=f"Record camera positions {dataset_id}",
                     unit="frame",
                     ncols=100):
        save_camera_position_for_frame(camera, scene, dataset_id, frame)

    print(f"======== Camera position recording complete ({total_frames} frames total) ========")


def write_camera_positions_to_file(output_file, dataset_id):

    global camera_positions_data

    try:

        if dataset_id not in camera_positions_data:
            print(f"\n⚠ Warning: no recorded data found for dataset {dataset_id}")
            return

        dataset_data = camera_positions_data[dataset_id]


        output_data = {
            "coordinate_convention": "OpenCV: x_cam = R * X_world + t (Rcw/tcw)",
            "twc_convention": "Nerfstudio: X_world -> cam with inverse(Twc); stored transform_matrix = Twc (cam-to-world)",
            "dataset_id": dataset_id,
            "num_images": len(dataset_data),
            "per_image": dataset_data
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Camera positions for dataset {dataset_id} saved to: {output_file}")
        print(f"  Recorded {len(dataset_data)} frames")
        print(f"  Format: OpenCV coordinate system with transform_matrix (Twc)")
    except Exception as e:
        print(f"\n✗ Failed to save camera positions: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


@bpy.app.handlers.persistent
def render_init_handler(scene):

    global camera_positions_data
    global render_progress_bar


    dataset_id = scene.get("current_dataset_id", "unknown")


    if dataset_id not in camera_positions_data:
        camera_positions_data[dataset_id] = {}
    else:

        camera_positions_data[dataset_id] = {}

    print(f"\n======== Start rendering dataset {dataset_id} ========")


    total_frames = scene.frame_end - scene.frame_start + 1
    render_progress_bar = tqdm(
        total=total_frames,
        desc=f"Render {dataset_id}",
        unit="frame",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    print()


@bpy.app.handlers.persistent
def render_pre_handler(scene):

    global render_progress_bar

    frame = scene.frame_current
    camera = scene.camera
    if camera:
        loc = camera.location

        if render_progress_bar is not None:
            render_progress_bar.update(1)
            render_progress_bar.set_postfix({
                'Frame': frame,
                'X': f'{loc.x:.2f}',
                'Y': f'{loc.y:.2f}',
                'Z': f'{loc.z:.2f}'
            })


@bpy.app.handlers.persistent
def render_complete_handler(scene):

    global render_progress_bar

    if render_progress_bar is not None:
        render_progress_bar.close()
        render_progress_bar = None

    print("\n======== Rendering complete ========")


@bpy.app.handlers.persistent
def render_cancel_handler(scene):

    global render_progress_bar

    if render_progress_bar is not None:
        render_progress_bar.close()
        render_progress_bar = None

    print("\n======== Rendering cancelled ========")


def register_render_handlers():

    handlers_to_register = [
        (bpy.app.handlers.render_init, render_init_handler),
        (bpy.app.handlers.render_pre, render_pre_handler),
        (bpy.app.handlers.render_complete, render_complete_handler),
        (bpy.app.handlers.render_cancel, render_cancel_handler),
    ]

    for handler_list, handler_func in handlers_to_register:
        if handler_func in handler_list:
            handler_list.remove(handler_func)
        handler_list.append(handler_func)

    print("✓ Render handlers registered")


def create_video_from_frames(frames_dir, output_video_path, fps=30):

    def encoder_available(name: str) -> bool:
        try:
            out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
            return name in out
        except Exception:
            return False

    print(f"\n{'='*60}")
    print(f"Merging video...")
    print(f"  Input directory: {frames_dir}")
    print(f"  Output video: {output_video_path}")
    print(f"  Frame rate: {fps} fps")

    input_pattern = os.path.join(frames_dir, "frame_%04d.png")
    base = ["ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps), "-start_number", "1",
            "-i", input_pattern, "-pix_fmt", "yuv420p"]


    candidates = []
    if encoder_available("libx264"):
        candidates.append(base + ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", output_video_path])
    if encoder_available("libopenh264"):

        candidates.append(base + ["-c:v", "libopenh264", "-b:v", "10M", "-g", str(fps*2), output_video_path])

    candidates.append(base + ["-c:v", "mpeg4", "-q:v", "2", output_video_path])

    last_err = ""
    for idx, cmd in enumerate(candidates, 1):
        print(f"  Trying encoder [{idx}/{len(candidates)}]: {' '.join(cmd[cmd.index('-c:v'):cmd.index('-c:v')+2])}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"✓ Video created successfully: {output_video_path}")
            try:
                size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
                print(f"  Video size: {size_mb:.2f} MB")
            except Exception:
                pass
            return True
        else:

            print("  This encoder failed, falling back...")
            last_err = f"Command: {' '.join(cmd)}\nError output: {proc.stderr}"

    print(f"\n✗ Video creation failed (all candidates failed):\n{last_err}")
    return False


def log_progress(dataset_id, split, total_split, processed_count, remaining_count, log_file):

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} | ID: {dataset_id} | Split: {split}/{total_split} | Processed: {processed_count} | Remaining: {remaining_count}\n"


        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)


        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        print(f"\n✓ Progress logged to: {log_file}")
        print(f"  {log_entry.strip()}")

    except Exception as e:
        print(f"\n⚠ Warning: failed to log progress: {e}")


def validate_dataset(dataset_id, base_path, hf_repo=None, hf_token=None):

    ply_file = os.path.join(base_path, dataset_id, "3dgs_uncompressed.ply")
    path_json = os.path.join(base_path, dataset_id, "path.json")


    local_exists = os.path.exists(ply_file) and os.path.exists(path_json)

    if local_exists:
        return True, ""


    return True, ""


def process_single_dataset(dataset_id, base_path, output_dir, target_frames_param, hf_repo, hf_token, split=None, total_split=None, processed_count=None, remaining_count=None):

    print(f"\n{'='*60}")
    print(f"Start processing dataset: {dataset_id}")
    print(f"{'='*60}")

    try:

        ply_file = os.path.join(base_path, dataset_id, "3dgs_uncompressed.ply")
        path_json = os.path.join(base_path, dataset_id, "path.json")
        dataset_output_dir = os.path.join(output_dir, dataset_id)


        if not (os.path.exists(ply_file) and os.path.exists(path_json)):
            print(f"Dataset files not found locally, downloading from Hugging Face...")
            success = download_and_prepare_dataset(dataset_id, base_path, hf_repo, hf_token)
            if not success:
                return False, f"Failed to download dataset from Hugging Face"


        if not os.path.exists(ply_file):
            return False, f"PLY file does not exist: {ply_file}"

        if not os.path.exists(path_json):
            return False, f"Path JSON file does not exist: {path_json}"


        pano_camera_dir = os.path.join(dataset_output_dir, "pano_camera0")
        if not os.path.exists(pano_camera_dir):
            os.makedirs(pano_camera_dir)
            print(f"✓ Created output directory: {pano_camera_dir}")


        clear_scene()


        all_path_points = load_and_transform_path(path_json)


        selected_points = select_keyframe_points(
            all_path_points,
            target_speed,
            min_keyframe_distance,
            fps
        )

        if len(selected_points) == 0:
            return False, f"No keyframe points were selected"


        animation_points, estimated_frames = prepare_animation_path(
            selected_points,
            target_frames_param,
            min_frames_for_backward,
            target_speed
        )


        gs_object = import_gaussian_splatting(ply_file)


        is_static = (len(animation_points) == 1)
        camera, total_frames = setup_camera_animation(
            animation_points,
            fps,
            target_speed,
            dataset_id,
            target_frames_param,
            is_static=is_static
        )

        if camera is None:
            return False, f"Camera animation setup failed"

        scene = bpy.context.scene


        scene["current_dataset_id"] = dataset_id


        output_filename = os.path.join(pano_camera_dir, "frame_####.png")
        scene.render.filepath = output_filename

        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGB'
        scene.render.image_settings.compression = 15
        scene.render.image_settings.color_depth = '8'

        print(f"\nRender output: {scene.render.filepath}")
        print(f"Output format: PNG sequence")


        print(f"\nPreparing to record camera positions (optimized: no per-frame scene updates required)")
        save_all_camera_positions(camera, scene, dataset_id)


        transforms_file = os.path.join(dataset_output_dir, "transforms.json")
        write_camera_positions_to_file(transforms_file, dataset_id)


        print(f"\n======== Start rendering dataset {dataset_id} ========")
        bpy.ops.render.render(animation=True)

        print(f"======== Dataset {dataset_id} rendering complete! ========")


        video_output_path = os.path.join(dataset_output_dir, "video.mp4")
        video_success = create_video_from_frames(pano_camera_dir, video_output_path, fps=fps)

        if not video_success:
            print(f"⚠ Warning: video creation failed, but rendered frames were saved")


        if split is not None and total_split is not None:
            log_progress(
                dataset_id,
                split,
                total_split,
                processed_count if processed_count is not None else 0,
                remaining_count if remaining_count is not None else 0,
                PROCESS_LOG_FILE
            )

        return True, "Rendering succeeded"

    except Exception as e:
        import traceback
        error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\n✗ Dataset {dataset_id} {error_msg}", file=sys.stderr)


        try:
            transforms_file = os.path.join(dataset_output_dir, "transforms.json")
            write_camera_positions_to_file(transforms_file, dataset_id)
        except:
            pass

        return False, error_msg


def process_split_worker(split_id, datasets, config):

    current_process = multiprocessing.current_process()
    current_process.name = f"Split-{split_id}"


    sys.stdout.flush()

    print(f"\n{'='*60}", flush=True)
    print(f"[Split {split_id}] Process started (PID: {os.getpid()})", flush=True)
    print(f"[Split {split_id}] Will process {len(datasets)} datasets", flush=True)
    print(f"{'='*60}\n", flush=True)

    try:

        try:

            cpu_count = os.cpu_count()
            if hasattr(os, 'sched_setaffinity'):

                os.sched_setaffinity(0, range(cpu_count))
                print(f"[Split {split_id}] ✓ CPU affinity set to use all {cpu_count} cores", flush=True)
        except Exception as e:
            print(f"[Split {split_id}] ⚠ Unable to set CPU affinity: {e}", flush=True)


        print(f"[Split {split_id}] Initializing Blender environment...", flush=True)
        install_and_enable_addon(addon_path, addon_name)
        setup_gpu_rendering()
        setup_render_settings()
        register_render_handlers()
        print(f"[Split {split_id}] ✓ Blender environment initialization complete", flush=True)
    except Exception as e:
        print(f"[Split {split_id}] ✗ Initialization failed: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return (split_id, 0, [(f"Initialization failed", str(e))])

    success_count = 0
    failed_datasets = []

    for idx, dataset_id in enumerate(datasets, start=1):
        print(f"\n[Split {split_id}] Progress: [{idx}/{len(datasets)}] Processing dataset: {dataset_id}", flush=True)

        try:
            success, message = process_single_dataset(
                dataset_id,
                config['base_path'],
                config['output_dir'],
                config['target_frames'],
                config['hf_repo'],
                config['hf_token'],
                split=split_id,
                total_split=config['total_split'],
                processed_count=idx,
                remaining_count=len(datasets) - idx
            )

            if success:
                success_count += 1
                print(f"[Split {split_id}] ✓ {dataset_id} processed successfully", flush=True)
            else:
                failed_datasets.append((dataset_id, message))
                print(f"[Split {split_id}] ✗ {dataset_id} processing failed: {message}", flush=True)

        except Exception as e:
            import traceback
            error_msg = f"Fatal error: {str(e)}\n{traceback.format_exc()}"
            print(f"[Split {split_id}] ✗ {dataset_id} {error_msg}", file=sys.stderr, flush=True)
            failed_datasets.append((dataset_id, error_msg))

    print(f"\n[Split {split_id}] Process complete: succeeded {success_count}/{len(datasets)}", flush=True)
    sys.stdout.flush()
    return (split_id, success_count, failed_datasets)


def main():
    config = parse_sys_argv()

    print(f"\n{'='*60}")
    print(f"Blender 3DGS batch renderer (multiprocessing version)")
    print(f"{'='*60}")
    print(f"There are {len(config['datasets'])} datasets to process")
    print(f"Target frame count: {config['target_frames']} frames")
    if config['splits'] and config['total_split'] is not None:
        print(f"Split configuration: {config['splits']} / {config['total_split']}")
        print(f"Using {len(config['splits'])} processes for parallel processing")
    print(f"Hugging Face repository: {config['hf_repo']}")
    print(f"Local cache path: {config['base_path']}")
    print(f"Output path: {config['output_dir']}")
    print(f"Progress log: {PROCESS_LOG_FILE}")
    print(f"\nOptimizations:")
    print(f"  ✓ Record camera positions with fcurve evaluation (no per-frame scene updates)")
    print(f"  ✓ Render multiple splits in parallel with Python multiprocessing")
    print(f"  ✓ Reduce scene update overhead to improve processing speed")
    print(f"{'='*60}\n")


    print("Validating datasets...")
    valid_datasets = []
    invalid_datasets = []

    for dataset_id in config['datasets']:
        is_valid, error_msg = validate_dataset(
            dataset_id,
            config['base_path'],
            config['hf_repo'],
            config['hf_token']
        )
        if is_valid:
            valid_datasets.append(dataset_id)
            print(f"  ✓ {dataset_id}: valid")
        else:
            invalid_datasets.append((dataset_id, error_msg))
            print(f"  ✗ {dataset_id}: {error_msg}")

    if invalid_datasets:
        print(f"\n⚠ Warning: {len(invalid_datasets)} invalid dataset(s):")
        for dataset_id, error_msg in invalid_datasets:
            print(f"  - {dataset_id}: {error_msg}")

    if not valid_datasets:
        print("\n✗ Error: no valid datasets available to process")
        sys.exit(1)

    print(f"\n✓ Validation complete: {len(valid_datasets)}/{len(config['datasets'])} datasets are valid\n")


    if config['dry_run']:
        print("=== DRY RUN mode - no rendering will be performed ===")
        sys.exit(0)


    if config['splits'] and config['total_split']:
        print(f"\nAssigning datasets by split...")


        split_datasets = {split_id: [] for split_id in config['splits']}

        for dataset_id in valid_datasets:

            dataset_index = valid_datasets.index(dataset_id)
            assigned_split = (dataset_index % config['total_split']) + 1


            if assigned_split in config['splits']:
                split_datasets[assigned_split].append(dataset_id)


        for split_id in config['splits']:
            count = len(split_datasets[split_id])
            print(f"  Split {split_id}: {count} datasets")


        split_datasets = {k: v for k, v in split_datasets.items() if v}

        if not split_datasets:
            print("\n✗ Error: no datasets were assigned to the specified splits")
            sys.exit(1)

        print(f"\nLaunching {len(split_datasets)} processes for parallel rendering...\n")


        tasks = [
            (split_id, datasets, config)
            for split_id, datasets in split_datasets.items()
        ]


        with NoDaemonPool(processes=len(split_datasets)) as pool:
            results = pool.starmap(process_split_worker, tasks)


        total_success = 0
        all_failed = []

        for split_id, success_count, failed_datasets in results:
            total_success += success_count
            all_failed.extend([(f"Split{split_id}:{ds}", msg) for ds, msg in failed_datasets])


        print(f"\n\n{'='*60}")
        print(f"Multiprocess batch processing complete!")
        print(f"{'='*60}")

        total_processed = sum(len(datasets) for datasets in split_datasets.values())
        print(f"✓ Successfully processed in total: {total_success}/{total_processed} datasets")
        print(f"  Used {len(split_datasets)} parallel processes")

        if all_failed:
            print(f"\n✗ Failed datasets ({len(all_failed)}):")
            for dataset_id, error_msg in all_failed:
                print(f"  - {dataset_id}")
                first_line = error_msg.split('\n')[0]
                print(f"    {first_line}")
        else:
            print(f"\n✓ All datasets were processed successfully!")

        print(f"{'='*60}\n")

    else:

        print("\nSingle-process sequential mode...\n")


        install_and_enable_addon(addon_path, addon_name)
        setup_gpu_rendering()
        verify_gpu_usage()
        setup_render_settings()
        register_render_handlers()


        success_count = 0
        failed_datasets = []
        total_count = len(valid_datasets)


        for idx, dataset_id in enumerate(valid_datasets, start=1):
            print(f"\n\n{'#'*60}")
            print(f"Progress: [{idx}/{len(valid_datasets)}] Processing dataset: {dataset_id}")
            print(f"{'#'*60}\n")

            try:

                remaining_count = total_count - idx + 1

                success, message = process_single_dataset(
                    dataset_id,
                    config['base_path'],
                    config['output_dir'],
                    config['target_frames'],
                    config['hf_repo'],
                    config['hf_token'],
                    split=None,
                    total_split=None,
                    processed_count=idx,
                    remaining_count=remaining_count
                )
                if success:
                    success_count += 1
                else:
                    failed_datasets.append((dataset_id, message))
            except Exception as e:
                import traceback
                error_msg = f"Fatal error: {str(e)}\n{traceback.format_exc()}"
                print(f"\n✗ {error_msg}", file=sys.stderr)
                failed_datasets.append((dataset_id, error_msg))


        print(f"\n\n{'='*60}")
        print(f"Batch processing complete!")
        print(f"{'='*60}")
        print(f"✓ Successfully processed: {success_count}/{len(valid_datasets)} datasets")

        if failed_datasets:
            print(f"✗ Failed datasets ({len(failed_datasets)}):")
            for dataset_id, error_msg in failed_datasets:
                print(f"  - {dataset_id}")

                first_line = error_msg.split('\n')[0]
                print(f"    {first_line}")
        else:
            print(f"✓ All datasets were processed successfully!")

        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

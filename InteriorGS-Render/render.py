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
from datetime import datetime
import subprocess
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R_scipy

import multiprocessing
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

PROCESS_LOG_FILE = "process.log"

panorama_type = 'EQUIRECTANGULAR'

fps = 30
target_speed = 0.02
min_keyframe_distance = target_speed

target_frames = 800
min_frames_for_backward = 800

from multiprocessing.pool import Pool as BasePool

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
                    print(f"Error: --split must be integer, got: {script_args[i]}")
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
            print(f"Warning: unknown argument '{arg}', will be ignored")
            i += 1

    if config['dataset_file']:
        try:
            with open(config['dataset_file'], 'r') as f:
                datasets = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            config['datasets'] = datasets
            print(f"Loaded {len(datasets)} datasets from {config['dataset_file']}")
        except Exception as e:
            print(f"Error: failed to read dataset file {config['dataset_file']}: {e}")
            sys.exit(1)

    if not config['datasets']:
        print("Error: no datasets specified")
        print("Use --help for usage information")
        sys.exit(1)

    return config


def print_help():
    help_text = """
Blender 3DGS Batch Rendering Tool

Usage:
  blender --background --python script.py -- [options]

Note: Must use -- to separate Blender args from script args

Options:
  --datasets ID1 ID2 ...        Dataset ID list, space separated
  --dataset-file FILE           Text file containing dataset IDs (one per line)
  --base-path PATH              Local dataset cache path (default: datasets)
  --output-dir PATH             Output base directory (default: output)
  --target-frames N             Target video frame count (default: 800)
  --hf-token TOKEN              Hugging Face access token (for private repos)
  --split N1 N2 ...             Split numbers (supports multiple for parallel processing)
  --total-split N               Total number of splits (for multi-server parallel)
  --dry-run                     Only validate dataset files, don't render
  --help, -h                    Show this help message

Examples:
  blender --background --python script.py -- --datasets 0001_839920

  blender --background --python script.py -- --dataset-file datasets.txt

  blender --background --python script.py -- --dataset-file datasets.txt --split 1 2 --total-split 200

  blender --background --python script.py -- --datasets 0001_839920 --dry-run
"""
    print(help_text)


def install_and_enable_addon(addon_path, addon_name):
    try:
        if addon_name in bpy.context.preferences.addons:
            print(f"Addon '{addon_name}' already enabled")
            return True

        if not os.path.exists(addon_path):
            print(f"Error: addon not found: {addon_path}")
            return False

        try:
            bpy.ops.preferences.addon_install(filepath=addon_path)
            print(f"Addon installed: {addon_name}")
        except Exception as e:
            print(f"Warning: addon install failed (might be already installed): {e}")

        try:
            bpy.ops.preferences.addon_enable(module=addon_name)
            print(f"Addon enabled: {addon_name}")
            return True
        except Exception as e:
            print(f"Error enabling addon: {e}")
            return False

    except Exception as e:
        print(f"Error installing addon: {e}")
        import traceback
        traceback.print_exc()
        return False


def setup_gpu_rendering():
    try:
        scene = bpy.context.scene
        scene.cycles.device = 'GPU'
        
        prefs = bpy.context.preferences
        cprefs = prefs.addons.get('cycles')
        
        if cprefs:
            cprefs = cprefs.preferences
            cprefs.compute_device_type = 'CUDA'
            
            for device in cprefs.devices:
                if device.type == 'CUDA':
                    device.use = True
                    print(f"Enabled GPU: {device.name}")
        
        bpy.context.scene.render.engine = 'CYCLES'
        print("GPU rendering configured")
        return True
        
    except Exception as e:
        print(f"Error setting up GPU: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_gpu_usage():
    try:
        scene = bpy.context.scene
        
        print("\nGPU Configuration:")
        print(f"  Render engine: {scene.render.engine}")
        print(f"  Device: {scene.cycles.device}")
        
        prefs = bpy.context.preferences
        cprefs = prefs.addons.get('cycles')
        
        if cprefs:
            cprefs = cprefs.preferences
            print(f"  Compute device type: {cprefs.compute_device_type}")
            
            enabled_gpus = [d.name for d in cprefs.devices if d.use and d.type == 'CUDA']
            if enabled_gpus:
                print(f"  Enabled GPUs: {', '.join(enabled_gpus)}")
            else:
                print("  Warning: No GPUs enabled")
        
        return True
        
    except Exception as e:
        print(f"Error verifying GPU: {e}")
        return False


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
    
    print("Scene cleared")


def import_gaussian_splatting(ply_filepath):
    try:
        print(f"\nImporting 3DGS from: {ply_filepath}")
        
        if not os.path.exists(ply_filepath):
            print(f"Error: PLY file not found: {ply_filepath}")
            return None
        
        bpy.ops.object.add_gaussian_splatting(filepath=ply_filepath)
        
        gs_object = bpy.context.selected_objects[0] if bpy.context.selected_objects else None
        
        if gs_object:
            print(f"3DGS imported successfully: {gs_object.name}")
        else:
            print("Warning: No object selected after import")
        
        return gs_object
    
    except Exception as e:
        print(f"Error importing 3DGS: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_and_transform_path(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        points = []
        
        for item in data:
            if 'pos' in item:
                pos = item['pos']
                
                x_blender = pos[2]
                y_blender = pos[0]
                z_blender = pos[1]
                
                points.append((x_blender, y_blender, z_blender))
        
        print(f"Loaded {len(points)} path points from {json_path}")
        return points
    
    except Exception as e:
        print(f"Error loading path: {e}")
        return []


def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)


def select_keyframe_points(points, target_speed, min_distance, fps):
    if len(points) == 0:
        return []
    
    selected = [points[0]]
    
    for i in range(1, len(points)):
        dist = calculate_distance(selected[-1], points[i])
        if dist >= min_distance:
            selected.append(points[i])
    
    print(f"Selected {len(selected)} keyframe points (from {len(points)} total)")
    return selected


def prepare_animation_path(selected_points, target_frames, min_frames_for_backward, target_speed):
    if len(selected_points) == 0:
        return [], 0
    
    if len(selected_points) == 1:
        return selected_points, 1
    
    total_distance = 0
    for i in range(len(selected_points) - 1):
        dist = calculate_distance(selected_points[i], selected_points[i+1])
        total_distance += dist
    
    estimated_frames = int(total_distance / target_speed)
    
    animation_points = selected_points[:]
    
    if estimated_frames < min_frames_for_backward:
        print(f"Path too short ({estimated_frames} frames), adding backward path")
        backward_points = selected_points[-2:0:-1]
        animation_points.extend(backward_points)
        estimated_frames = estimated_frames * 2
    
    print(f"Animation path prepared: {len(animation_points)} points, ~{estimated_frames} frames")
    return animation_points, estimated_frames


def setup_camera_animation(animation_points, fps, target_speed, dataset_id, target_frames_param=800, is_static=False):
    scene = bpy.context.scene
    
    if bpy.context.scene.camera:
        camera = bpy.context.scene.camera
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        scene.camera = camera
    
    camera.name = f"PanoramaCamera_{dataset_id}"
    camera.data.type = panorama_type
    
    scene.frame_start = 1
    
    if is_static:
        scene.frame_end = target_frames_param
        loc = animation_points[0]
        camera.location = loc
        camera.rotation_euler = (math.radians(90), 0, 0)
        camera.keyframe_insert(data_path="location", frame=1)
        camera.keyframe_insert(data_path="rotation_euler", frame=1)
        print(f"Static camera setup at {loc}, {target_frames_param} frames")
        return camera, target_frames_param
    
    def seg_frames(p0, p1):
        dist = calculate_distance(p0, p1)
        n = max(1, int(dist / target_speed))
        return n
    
    total_seg_frames = 0
    for i in range(len(animation_points) - 1):
        n = seg_frames(animation_points[i], animation_points[i+1])
        total_seg_frames += n
    
    total_frames = total_seg_frames
    
    scene.frame_end = total_frames
    
    frame = 1
    for i, point in enumerate(animation_points):
        camera.location = point
        camera.rotation_euler = (math.radians(90), 0, 0)
        
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
        
        if i < len(animation_points) - 1:
            n = seg_frames(animation_points[i], animation_points[i+1])
            frame += n
    
    for fcurve in camera.animation_data.action.fcurves:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'LINEAR'
    
    print(f"Camera animation setup: {len(animation_points)} keyframes, {total_frames} total frames")
    return camera, total_frames


def setup_lighting():
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.object
    sun.data.energy = 1.0
    print("Lighting setup complete")


def setup_render_settings():
    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    
    print(f"Render settings: {scene.render.resolution_x}x{scene.render.resolution_y}, {scene.cycles.samples} samples")


def download_and_prepare_dataset(dataset_id, base_path, hf_repo, hf_token=None):
    from huggingface_hub import hf_hub_download
    
    dataset_dir = os.path.join(base_path, dataset_id)
    ply_file = os.path.join(dataset_dir, "3dgs_uncompressed.ply")
    path_json = os.path.join(dataset_dir, "path.json")
    
    if os.path.exists(ply_file) and os.path.exists(path_json):
        print(f"Dataset already downloaded: {dataset_id}")
        return True
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    try:
        print(f"\n{'='*60}")
        print(f"Downloading dataset from Hugging Face...")
        print(f"  Repo: {hf_repo}")
        print(f"  Dataset: {dataset_id}")
        print(f"  Local: {dataset_dir}")
        print(f"{'='*60}\n")
        
        try:
            compressed_ply = f"{dataset_id}/3dgs.ply"
            downloaded_compressed = hf_hub_download(
                repo_id=hf_repo,
                filename=compressed_ply,
                repo_type="dataset",
                token=hf_token,
                local_dir=base_path
            )
            print(f"Downloaded: {compressed_ply}")
        except Exception as e_comp:
            print(f"Compressed PLY not found: {e_comp}")
            uncompressed_ply = f"{dataset_id}/3dgs_uncompressed.ply"
            downloaded_uncompressed = hf_hub_download(
                repo_id=hf_repo,
                filename=uncompressed_ply,
                repo_type="dataset",
                token=hf_token,
                local_dir=base_path
            )
            print(f"Downloaded: {uncompressed_ply}")
        
        path_json_hf = f"{dataset_id}/path.json"
        downloaded_path = hf_hub_download(
            repo_id=hf_repo,
            filename=path_json_hf,
            repo_type="dataset",
            token=hf_token,
            local_dir=base_path
        )
        print(f"Downloaded: {path_json_hf}")
        
        compressed_file = os.path.join(dataset_dir, "3dgs.ply")
        if os.path.exists(compressed_file) and not os.path.exists(ply_file):
            print(f"\nDecompressing PLY...")
            print(f"  Input: {compressed_file}")
            print(f"  Output: {ply_file}")
            
            try:
                result = subprocess.run(
                    ["splat-transform", "decompress", compressed_file, ply_file],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Decompression complete")
                
                try:
                    os.remove(compressed_file)
                    print(f"Removed compressed file")
                except Exception as e_rm:
                    print(f"Warning: failed to remove compressed file: {e_rm}")
                    
            except subprocess.CalledProcessError as e:
                print(f"Error decompressing PLY: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                return False
            except FileNotFoundError:
                print("Error: splat-transform not found")
                print("Install: npm install -g @playcanvas/splat-transform")
                return False
        
        if not os.path.exists(ply_file):
            print(f"Error: PLY file not found after download: {ply_file}")
            return False
        
        if not os.path.exists(path_json):
            print(f"Error: path.json not found after download: {path_json}")
            return False
        
        print(f"\nDataset ready: {dataset_id}")
        return True
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def blender_to_original_location(blender_location):
    x_blender, y_blender, z_blender = blender_location
    x_orig = y_blender
    y_orig = z_blender
    z_orig = x_blender
    return (x_orig, y_orig, z_orig)


def blender_to_original_rotation(blender_rotation_euler):
    rx_b, ry_b, rz_b = blender_rotation_euler
    
    R_x_b = R_scipy.from_euler('x', rx_b, degrees=False).as_matrix()
    R_y_b = R_scipy.from_euler('y', ry_b, degrees=False).as_matrix()
    R_z_b = R_scipy.from_euler('z', rz_b, degrees=False).as_matrix()
    
    R_blender = R_z_b @ R_y_b @ R_x_b
    
    P_yz = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    
    R_original = P_yz @ R_blender @ P_yz.T
    
    r_orig = R_scipy.from_matrix(R_original)
    euler_orig = r_orig.as_euler('xyz', degrees=False)
    
    return tuple(euler_orig)


def get_3x4_RT_matrix_from_blender_original_coords(camera):
    blender_location = camera.location
    blender_rotation = camera.rotation_euler
    
    location_original = blender_to_original_location(blender_location)
    
    rotation_original_euler = blender_to_original_rotation(blender_rotation)
    
    R_original = R_scipy.from_euler('xyz', rotation_original_euler, degrees=False).as_matrix()
    
    t_original = np.array(location_original, dtype=np.float64).reshape(3, 1)
    
    RT = np.hstack([R_original, t_original])
    
    return RT


def get_3x4_RT_matrix_from_blender(camera):
    blender_location = camera.location
    blender_rotation_euler = camera.rotation_euler
    
    rx_b, ry_b, rz_b = blender_rotation_euler
    R_x_b = R_scipy.from_euler('x', rx_b, degrees=False).as_matrix()
    R_y_b = R_scipy.from_euler('y', ry_b, degrees=False).as_matrix()
    R_z_b = R_scipy.from_euler('z', rz_b, degrees=False).as_matrix()
    R_blender = R_z_b @ R_y_b @ R_x_b
    
    t_blender = np.array(blender_location, dtype=np.float64).reshape(3, 1)
    
    RT = np.hstack([R_blender, t_blender])
    
    return RT


def get_transform_matrix_from_blender(camera):
    blender_location = camera.location
    blender_rotation_euler = camera.rotation_euler
    
    rx_b, ry_b, rz_b = blender_rotation_euler
    R_x_b = R_scipy.from_euler('x', rx_b, degrees=False).as_matrix()
    R_y_b = R_scipy.from_euler('y', ry_b, degrees=False).as_matrix()
    R_z_b = R_scipy.from_euler('z', rz_b, degrees=False).as_matrix()
    R_blender = R_z_b @ R_y_b @ R_x_b
    
    t_blender = np.array(blender_location, dtype=np.float64).reshape(3, 1)
    
    Twc = np.eye(4, dtype=np.float64)
    Twc[:3, :3] = R_blender
    Twc[:3, 3:4] = t_blender
    
    return Twc


def rvec_from_rotation_matrix(R):
    r = R_scipy.from_matrix(R)
    rvec = r.as_rotvec()
    return rvec


def evaluate_camera_at_frame(camera, frame):
    fcurves_location = [
        fc for fc in camera.animation_data.action.fcurves
        if fc.data_path == 'location'
    ]
    fcurves_rotation = [
        fc for fc in camera.animation_data.action.fcurves
        if fc.data_path == 'rotation_euler'
    ]
    
    location = []
    for fc in sorted(fcurves_location, key=lambda x: x.array_index):
        location.append(fc.evaluate(frame))
    
    rotation_euler = []
    for fc in sorted(fcurves_rotation, key=lambda x: x.array_index):
        rotation_euler.append(fc.evaluate(frame))
    
    return tuple(location), tuple(rotation_euler)


def compute_RT_from_location_rotation(blender_location, blender_rotation):
    rx_b, ry_b, rz_b = blender_rotation
    R_x_b = R_scipy.from_euler('x', rx_b, degrees=False).as_matrix()
    R_y_b = R_scipy.from_euler('y', ry_b, degrees=False).as_matrix()
    R_z_b = R_scipy.from_euler('z', rz_b, degrees=False).as_matrix()
    R_blender = R_z_b @ R_y_b @ R_x_b
    
    t_blender = np.array(blender_location, dtype=np.float64).reshape(3, 1)
    
    RT = np.hstack([R_blender, t_blender])
    
    Twc = np.eye(4, dtype=np.float64)
    Twc[:3, :3] = R_blender
    Twc[:3, 3:4] = t_blender
    
    return RT, Twc


camera_positions_data = {}

def save_camera_position_for_frame(camera, scene, dataset_id, frame):
    global camera_positions_data
    
    if dataset_id not in camera_positions_data:
        camera_positions_data[dataset_id] = {}
    
    blender_location, blender_rotation_euler = evaluate_camera_at_frame(camera, frame)
    
    RT, Twc = compute_RT_from_location_rotation(blender_location, blender_rotation_euler)
    
    frame_filename = f"pano_camera0/frame_{frame:04d}.png"
    
    camera_positions_data[dataset_id][frame_filename] = {
        "transform_matrix": Twc.tolist(),
        "location": list(blender_location),
        "rotation": list(blender_rotation_euler),
        "frame": frame
    }


def save_all_camera_positions(camera, scene, dataset_id):
    global camera_positions_data
    
    print(f"\n{'='*60}")
    print(f"Recording camera positions (optimized: no per-frame scene update)")
    
    if dataset_id not in camera_positions_data:
        camera_positions_data[dataset_id] = {}
    
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    total_frames = end_frame - start_frame + 1
    
    with tqdm(total=total_frames, desc=f"Recording {dataset_id}", unit="frame", ncols=100) as pbar:
        for frame in range(start_frame, end_frame + 1):
            save_camera_position_for_frame(camera, scene, dataset_id, frame)
            pbar.update(1)
    
    print(f"{'='*60}")
    print(f"Camera position recording complete ({total_frames} frames)")
    print(f"{'='*60}")


def write_camera_positions_to_file(output_file, dataset_id):
    global camera_positions_data

    try:
        if dataset_id not in camera_positions_data:
            print(f"\nWarning: no recorded data for dataset {dataset_id}")
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

        print(f"\nDataset {dataset_id} camera positions saved to: {output_file}")
        print(f"  Recorded {len(dataset_data)} frames")
        print(f"  Format: OpenCV coordinate system with transform_matrix (Twc)")
    except Exception as e:
        print(f"\nError saving camera positions: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


render_progress_bar = None

@bpy.app.handlers.persistent
def render_init_handler(scene):
    global camera_positions_data
    global render_progress_bar

    dataset_id = scene.get("current_dataset_id", "unknown")

    if dataset_id not in camera_positions_data:
        camera_positions_data[dataset_id] = {}
    else:
        camera_positions_data[dataset_id] = {}

    print(f"\n{'='*60}")
    print(f"Starting render for dataset {dataset_id}")
    print(f"{'='*60}")

    total_frames = scene.frame_end - scene.frame_start + 1
    render_progress_bar = tqdm(
        total=total_frames,
        desc=f"Rendering {dataset_id}",
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
                'frame': frame,
                'pos': f"({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f})"
            })


@bpy.app.handlers.persistent
def render_complete_handler(scene):
    global render_progress_bar
    
    if render_progress_bar is not None:
        render_progress_bar.close()
        render_progress_bar = None
    
    print(f"\n{'='*60}")
    print(f"Render complete!")
    print(f"{'='*60}\n")


@bpy.app.handlers.persistent
def render_cancel_handler(scene):
    global render_progress_bar
    
    if render_progress_bar is not None:
        render_progress_bar.close()
        render_progress_bar = None
    
    print(f"\nRender cancelled")


def register_render_handlers():
    bpy.app.handlers.render_init.append(render_init_handler)
    bpy.app.handlers.render_pre.append(render_pre_handler)
    bpy.app.handlers.render_complete.append(render_complete_handler)
    bpy.app.handlers.render_cancel.append(render_cancel_handler)
    print("Render handlers registered")


def create_video_from_frames(frames_dir, output_video_path, fps=30):
    import os
    import subprocess

    def encoder_available(name: str) -> bool:
        try:
            out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
            return name in out
        except Exception:
            return False

    print(f"\n{'='*60}")
    print(f"Creating video...")
    print(f"  Input dir: {frames_dir}")
    print(f"  Output: {output_video_path}")
    print(f"  FPS: {fps}")

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
            print(f"Video created successfully: {output_video_path}")
            try:
                size_mb = os.path.getsize(output_video_path) / (1024 * 1024)
                print(f"  Video size: {size_mb:.2f} MB")
            except Exception:
                pass
            return True
        else:
            print("  Encoder failed, trying fallback...")
            last_err = f"Command: {' '.join(cmd)}\nError: {proc.stderr}"

    print(f"\nVideo creation failed (all encoders failed):\n{last_err}")
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
        
        print(f"Progress logged: {log_entry.strip()}")
    except Exception as e:
        print(f"Warning: failed to log progress: {e}")


def validate_dataset(dataset_id, base_path, hf_repo=None, hf_token=None):
    ply_file = os.path.join(base_path, dataset_id, "3dgs_uncompressed.ply")
    path_json = os.path.join(base_path, dataset_id, "path.json")
    
    ply_exists = os.path.exists(ply_file)
    json_exists = os.path.exists(path_json)
    
    if ply_exists and json_exists:
        return True, "Valid"
    
    if hf_repo:
        return True, "Will download from HF"
    
    if not ply_exists:
        return False, "PLY file not found"
    
    if not json_exists:
        return False, "path.json not found"
    
    return False, "Unknown error"


def process_single_dataset(dataset_id, base_path, output_dir, target_frames_param, hf_repo, hf_token, 
                           split=None, total_split=None, processed_count=None, remaining_count=None):
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
            return False, f"PLY file not found: {ply_file}"

        if not os.path.exists(path_json):
            return False, f"path.json not found: {path_json}"

        pano_camera_dir = os.path.join(dataset_output_dir, "pano_camera0")
        if not os.path.exists(pano_camera_dir):
            os.makedirs(pano_camera_dir)
            print(f"Created output directory: {pano_camera_dir}")

        clear_scene()

        all_path_points = load_and_transform_path(path_json)

        selected_points = select_keyframe_points(
            all_path_points,
            target_speed,
            min_keyframe_distance,
            fps
        )

        if len(selected_points) == 0:
            return False, f"No keyframe points selected"

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

        print(f"\nPreparing to record camera positions (optimized: no per-frame scene update)")
        save_all_camera_positions(camera, scene, dataset_id)

        transforms_file = os.path.join(dataset_output_dir, "transforms.json")
        write_camera_positions_to_file(transforms_file, dataset_id)

        print(f"\n{'='*60}")
        print(f"Starting render for dataset {dataset_id}")
        print(f"{'='*60}")
        bpy.ops.render.render(animation=True)

        print(f"{'='*60}")
        print(f"Dataset {dataset_id} render complete!")
        print(f"{'='*60}")

        video_output_path = os.path.join(dataset_output_dir, "video.mp4")
        video_success = create_video_from_frames(pano_camera_dir, video_output_path, fps=fps)

        if not video_success:
            print(f"Warning: video creation failed, but rendered frames saved")

        if split is not None and total_split is not None:
            log_progress(
                dataset_id,
                split,
                total_split,
                processed_count if processed_count is not None else 0,
                remaining_count if remaining_count is not None else 0,
                PROCESS_LOG_FILE
            )

        return True, "Render successful"

    except Exception as e:
        import traceback
        error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
        print(f"\nDataset {dataset_id} {error_msg}", file=sys.stderr)

        try:
            transforms_file = os.path.join(dataset_output_dir, "transforms.json")
            write_camera_positions_to_file(transforms_file, dataset_id)
        except:
            pass

        return False, error_msg


def process_split_worker(split_id, datasets, config):
    import multiprocessing
    
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
                print(f"[Split {split_id}] Set CPU affinity: using all {cpu_count} cores", flush=True)
        except Exception as e:
            print(f"[Split {split_id}] Cannot set CPU affinity: {e}", flush=True)
        
        print(f"[Split {split_id}] Initializing Blender environment...", flush=True)
        install_and_enable_addon(addon_path, addon_name)
        setup_gpu_rendering()
        setup_render_settings()
        register_render_handlers()
        print(f"[Split {split_id}] Blender environment initialized", flush=True)
    except Exception as e:
        print(f"[Split {split_id}] Initialization failed: {e}", file=sys.stderr, flush=True)
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
                print(f"[Split {split_id}] {dataset_id} processed successfully", flush=True)
            else:
                failed_datasets.append((dataset_id, message))
                print(f"[Split {split_id}] {dataset_id} processing failed: {message}", flush=True)
                
        except Exception as e:
            import traceback
            error_msg = f"Critical error: {str(e)}\n{traceback.format_exc()}"
            print(f"[Split {split_id}] {dataset_id} {error_msg}", file=sys.stderr, flush=True)
            failed_datasets.append((dataset_id, error_msg))
    
    print(f"\n[Split {split_id}] Process complete: success {success_count}/{len(datasets)}", flush=True)
    sys.stdout.flush()
    return (split_id, success_count, failed_datasets)


def main():
    import multiprocessing
    
    config = parse_sys_argv()

    print(f"\n{'='*60}")
    print(f"Blender 3DGS Batch Rendering Tool (Multi-process Version)")
    print(f"{'='*60}")
    print(f"Total datasets: {len(config['datasets'])}")
    print(f"Target frames: {config['target_frames']}")
    if config['splits'] and config['total_split'] is not None:
        print(f"Split config: {config['splits']} / {config['total_split']}")
        print(f"Will use {len(config['splits'])} processes for parallel processing")
    print(f"Hugging Face repo: {config['hf_repo']}")
    print(f"Local cache path: {config['base_path']}")
    print(f"Output path: {config['output_dir']}")
    print(f"Progress log: {PROCESS_LOG_FILE}")
    print(f"\nOptimizations:")
    print(f"  - Using fcurve evaluation for camera position recording (no per-frame scene update)")
    print(f"  - Python multiprocessing for parallel rendering of multiple splits")
    print(f"  - Reduced scene update overhead, improved processing speed")
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
            print(f"  {dataset_id}: Valid")
        else:
            invalid_datasets.append((dataset_id, error_msg))
            print(f"  {dataset_id}: {error_msg}")

    if invalid_datasets:
        print(f"\nWarning: {len(invalid_datasets)} invalid datasets:")
        for dataset_id, error_msg in invalid_datasets:
            print(f"  - {dataset_id}: {error_msg}")

    if not valid_datasets:
        print("\nError: no valid datasets to process")
        sys.exit(1)

    print(f"\nValidation complete: {len(valid_datasets)}/{len(config['datasets'])} valid datasets\n")

    if config['dry_run']:
        print("=== DRY RUN MODE - No rendering will be performed ===")
        sys.exit(0)

    if config['splits'] and config['total_split']:
        print(f"\n{'='*60}")
        print(f"Starting multi-process parallel rendering")
        print(f"Splits to process: {config['splits']}")
        print(f"{'='*60}\n")
        
        datasets_per_split = {}
        total = len(valid_datasets)
        total_splits = config['total_split']
        
        for split_num in config['splits']:
            start_idx = int((split_num - 1) * total / total_splits)
            end_idx = int(split_num * total / total_splits)
            datasets_per_split[split_num] = valid_datasets[start_idx:end_idx]
            print(f"Split {split_num}: {len(datasets_per_split[split_num])} datasets (indices {start_idx} to {end_idx-1})")
        
        print(f"\n{'='*60}")
        print(f"Creating process pool with {len(config['splits'])} workers")
        print(f"{'='*60}\n")
        
        tasks = [(split_id, datasets_per_split[split_id], config) for split_id in config['splits']]
        
        num_workers = len(config['splits'])
        with NoDaemonPool(processes=num_workers) as pool:
            results = pool.starmap(process_split_worker, tasks)
        
        print(f"\n{'='*60}")
        print(f"All processes complete")
        print(f"{'='*60}\n")
        
        total_success = 0
        all_failed = []
        for split_id, success_count, failed_datasets in results:
            total_success += success_count
            all_failed.extend([(split_id, ds_id, msg) for ds_id, msg in failed_datasets])
            print(f"Split {split_id}: {success_count} successful")
        
        print(f"\nFinal results:")
        print(f"  Total successful: {total_success}")
        print(f"  Total failed: {len(all_failed)}")
        
        if all_failed:
            print(f"\nFailed datasets:")
            for split_id, ds_id, msg in all_failed[:10]:
                print(f"  [Split {split_id}] {ds_id}: {msg[:100]}...")
            if len(all_failed) > 10:
                print(f"  ... and {len(all_failed) - 10} more")
    
    else:
        print(f"\n{'='*60}")
        print(f"Starting single-process sequential rendering")
        print(f"{'='*60}\n")
        
        install_and_enable_addon(addon_path, addon_name)
        setup_gpu_rendering()
        verify_gpu_usage()
        setup_lighting()
        setup_render_settings()
        register_render_handlers()
        
        success_count = 0
        failed_datasets = []
        
        for idx, dataset_id in enumerate(valid_datasets, start=1):
            print(f"\n{'='*60}")
            print(f"Processing dataset [{idx}/{len(valid_datasets)}]: {dataset_id}")
            print(f"{'='*60}")
            
            success, message = process_single_dataset(
                dataset_id,
                config['base_path'],
                config['output_dir'],
                config['target_frames'],
                config['hf_repo'],
                config['hf_token']
            )
            
            if success:
                success_count += 1
                print(f"\n{dataset_id} processed successfully")
            else:
                failed_datasets.append((dataset_id, message))
                print(f"\n{dataset_id} processing failed: {message}")
        
        print(f"\n{'='*60}")
        print(f"All datasets processed")
        print(f"{'='*60}")
        print(f"Successful: {success_count}/{len(valid_datasets)}")
        print(f"Failed: {len(failed_datasets)}")
        
        if failed_datasets:
            print(f"\nFailed datasets:")
            for dataset_id, error_msg in failed_datasets:
                print(f"  - {dataset_id}: {error_msg[:200]}...")
    
    print(f"\n{'='*60}")
    print(f"Rendering complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
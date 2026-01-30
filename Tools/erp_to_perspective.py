#!/usr/bin/env python3
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

import argparse
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm


class ERPToPerspective:
    
    DIRECTIONS = {
        'forward': (180, 0),
        'backward': (0, 0),
        'left': (-90, 0),
        'right': (90, 0),
        'up': (180, 90),
        'down': (180, -90),
        's_curve': (180, 0),
        'loop': (180, 0),
    }
    
    def __init__(self, fov=90, out_width=1920, out_height=1080):
        self.fov = fov
        self.out_width = out_width
        self.out_height = out_height
        
        self.map_x = None
        self.map_y = None
        
        self.current_yaw = None
        self.current_pitch = None
        self.current_erp_size = None
    
    def _build_perspective_map(self, yaw_deg, pitch_deg, erp_width, erp_height):
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        fov_rad = np.deg2rad(self.fov)
        
        aspect = self.out_width / self.out_height
        
        x = np.linspace(-1, 1, self.out_width)
        y = np.linspace(-1, 1, self.out_height)
        xv, yv = np.meshgrid(x, y)
        
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        xv = xv * aspect
        
        xc = xv
        yc = -yv
        zc = -f
        
        norm = np.sqrt(xc**2 + yc**2 + zc**2)
        xc /= norm
        yc /= norm
        zc /= norm
        
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        
        x_rot1 = cos_y * xc + sin_y * zc
        y_rot1 = yc
        z_rot1 = -sin_y * xc + cos_y * zc
        
        x_world = x_rot1
        y_world = cos_p * y_rot1 - sin_p * z_rot1
        z_world = sin_p * y_rot1 + cos_p * z_rot1
        
        longitude = np.arctan2(-x_world, z_world)
        latitude = np.arcsin(np.clip(y_world, -1.0, 1.0))
        
        map_x = (longitude / (2 * np.pi) + 0.5) * erp_width
        map_y = (0.5 - latitude / np.pi) * erp_height
        
        self.map_x = map_x.astype(np.float32)
        self.map_y = map_y.astype(np.float32)
        
        self.current_yaw = yaw_deg
        self.current_pitch = pitch_deg
        self.current_erp_size = (erp_width, erp_height)
    
    def convert_frame(self, erp_frame, yaw_deg, pitch_deg):
        erp_height, erp_width = erp_frame.shape[:2]
        
        need_rebuild = (
            self.map_x is None or 
            self.current_yaw != yaw_deg or 
            self.current_pitch != pitch_deg or
            self.current_erp_size != (erp_width, erp_height)
        )
        
        if need_rebuild:
            self._build_perspective_map(yaw_deg, pitch_deg, erp_width, erp_height)
        
        perspective_frame = cv2.remap(
            erp_frame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )
        
        return perspective_frame
    
    def convert_video(self, input_path, output_path, yaw_deg, pitch_deg, fps=None):
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            return False
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open input video: {input_path}")
            return False
        
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if fps is None:
            fps = input_fps
        
        print(f"Input video info:")
        print(f"  Resolution: {input_width}x{input_height}")
        print(f"  FPS: {input_fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"\nOutput video info:")
        print(f"  Resolution: {self.out_width}x{self.out_height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  FOV: {self.fov}°")
        print(f"  Direction: yaw={yaw_deg}°, pitch={pitch_deg}°")
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (self.out_width, self.out_height)
        )
        
        if not out.isOpened():
            print(f"Error: Cannot create output video: {output_path}")
            cap.release()
            return False
        
        print("\nStarting conversion...")
        with tqdm(total=total_frames, desc="Progress") as pbar:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                perspective_frame = self.convert_frame(frame, yaw_deg, pitch_deg)
                
                out.write(perspective_frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        print(f"\nConversion complete!")
        print(f"Output file: {output_path}")
        print(f"Processed frames: {frame_count}")
        
        return True


def batch_convert_videos(input_pattern: str, output_dir: str = None, 
                         fov: float = 90, width: int = 1920, height: int = 1080, fps: float = None):
    directions = ['forward', 'backward', 'left', 'right', 's_curve', 'loop']
    
    converter = ERPToPerspective(fov=fov, out_width=width, out_height=height)
    
    success_count = 0
    total_count = 0
    
    print("="*70)
    print("Batch conversion mode")
    print(f"Input pattern: {input_pattern}")
    print(f"Direction list: {', '.join(directions)}")
    print(f"Note: s_curve and loop will use forward direction (yaw=0°, pitch=0°)")
    print("="*70)
    
    for direction in directions:
        total_count += 1
        
        input_path = input_pattern.replace('{dir}', direction)
        
        if not os.path.exists(input_path):
            print(f"\n[{total_count}/{len(directions)}] ⚠️  Skip {direction}: File does not exist")
            print(f"    Path: {input_path}")
            continue
        
        if output_dir is None:
            input_dir = os.path.dirname(input_path)
            output_path = os.path.join(input_dir, f"in_01_{direction}_per.mp4")
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"in_01_{direction}_per.mp4")
        
        yaw_deg, pitch_deg = ERPToPerspective.DIRECTIONS.get(direction, (0, 0))
        
        print(f"\n[{total_count}/{len(directions)}] 🎬 Processing {direction}")
        print(f"    Input: {input_path}")
        print(f"    Output: {output_path}")
        
        if direction in ['s_curve', 'loop']:
            print(f"    View: {direction} -> Using forward direction (yaw={yaw_deg}°, pitch={pitch_deg}°)")
        else:
            print(f"    View: {direction} (yaw={yaw_deg}°, pitch={pitch_deg}°)")
        
        try:
            success = converter.convert_video(
                input_path=input_path,
                output_path=output_path,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                fps=fps
            )
            
            if success:
                success_count += 1
                print(f"    ✅ Done")
            else:
                print(f"    ❌ Failed")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print("\n" + "="*70)
    print(f"Batch conversion complete: {success_count}/{total_count} successful")
    print("="*70)
    
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert ERP panoramic video to perspective video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

1. Single video - using preset direction:
   python erp_to_perspective.py -i input.mp4 -o output.mp4 --direction forward

2. Single video - custom angles:
   python erp_to_perspective.py -i input.mp4 -o output.mp4 --yaw 45 --pitch 30

3. Batch mode - auto process multiple directions:
   python erp_to_perspective.py \\
       --batch "/path/to/vis_ours_480p_speed_1_{dir}/in_01/generated.mp4"
   
   Batch mode will replace {dir} with: forward, backward, left, right, s_curve, loop
   Output files named as: in_01_{dir}_per.mp4 (same directory as input)

4. Batch mode - specify output directory:
   python erp_to_perspective.py \\
       --batch "/path/to/vis_ours_480p_speed_1_{dir}/in_01/generated.mp4" \\
       --output_dir /path/to/output

Preset directions:
  forward  : yaw=0°,    pitch=0°   (front, longitude 0 latitude 0)
  backward : yaw=180°,  pitch=0°   (back)
  left     : yaw=90°,   pitch=0°   (left)
  right    : yaw=-90°,  pitch=0°   (right)
  up       : yaw=0°,    pitch=90°  (up)
  down     : yaw=0°,    pitch=-90° (down)
  s_curve  : yaw=0°,    pitch=0°   (use forward direction)
  loop     : yaw=0°,    pitch=0°   (use forward direction)
        """
    )
    
    parser.add_argument('--batch', type=str, default=None,
                        help='Batch mode: input path pattern (with {dir} placeholder)')
    
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Single mode: input ERP panoramic video path')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Single mode: output perspective video path')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Batch mode: output directory (default: same as input)')
    
    parser.add_argument('--fov', type=float, default=90,
                        help='Field of view (degrees), default 90')
    parser.add_argument('--width', type=int, default=1920,
                        help='Output video width, default 1920')
    parser.add_argument('--height', type=int, default=1080,
                        help='Output video height, default 1080')
    parser.add_argument('--fps', type=float, default=None,
                        help='Output video FPS (default: same as input)')
    
    parser.add_argument('--direction', type=str, default=None,
                        choices=list(ERPToPerspective.DIRECTIONS.keys()),
                        help='Single mode: preset direction')
    
    parser.add_argument('--yaw', type=float, default=0,
                        help='Single mode: yaw angle (degrees), 0=forward, positive rotates right')
    parser.add_argument('--pitch', type=float, default=0,
                        help='Single mode: pitch angle (degrees), 0=horizontal, positive is up')
    
    args = parser.parse_args()
    
    if args.batch:
        if '{dir}' not in args.batch:
            print("Error: Batch mode requires {dir} placeholder in input path")
            return 1
        
        success_count = batch_convert_videos(
            input_pattern=args.batch,
            output_dir=args.output_dir,
            fov=args.fov,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        
        return 0 if success_count > 0 else 1
    
    else:
        if not args.input or not args.output:
            print("Error: Single mode requires -i/--input and -o/--output")
            print("Tip: Use --batch for batch processing, or -h for help")
            return 1
        
        if args.direction:
            yaw_deg, pitch_deg = ERPToPerspective.DIRECTIONS[args.direction]
            print(f"Using preset direction: {args.direction} (yaw={yaw_deg}°, pitch={pitch_deg}°)")
        else:
            yaw_deg, pitch_deg = args.yaw, args.pitch
            print(f"Using custom direction: yaw={yaw_deg}°, pitch={pitch_deg}°")
        
        converter = ERPToPerspective(
            fov=args.fov,
            out_width=args.width,
            out_height=args.height
        )
        
        success = converter.convert_video(
            input_path=args.input,
            output_path=args.output,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            fps=args.fps
        )
        
        if success:
            print("\n✓ Conversion successful!")
            return 0
        else:
            print("\n✗ Conversion failed")
            return 1


if __name__ == "__main__":
    sys.exit(main())

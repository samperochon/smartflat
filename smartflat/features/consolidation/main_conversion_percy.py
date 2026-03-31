"""Convert video frame rates from Percy hospital recordings to target fps.

When to run: When ingesting raw recordings from the Percy hospital file system.
Prerequisites: Input directory with MP4 video files from Percy.
Outputs: FPS-normalized videos (25 fps) or copied Tobii videos in the output directory.
Usage: python -m smartflat.features.consolidation.main_conversion_percy
"""

import os
import subprocess
import time
import shutil

def print_timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@print_timing
def convert_video_fps(input_path, output_path, target_fps=25, process=False):
    command = ['ffmpeg', '-i', input_path, '-vf', f"fps={target_fps}", '-c:a', 'copy', output_path]
    if process:
        print(' '.join(command))
        subprocess.run(command, shell=True)  # `shell=True` for Windows compatibility
    else:
        print(' '.join(command))
    return

@print_timing
def process_dataset(input_dir, output_dir, process=False, target_fps=25, n_max=2):
    n_init = 0
    for root, _, files in os.walk(input_dir): 
        print(f'Processing {root}')
        for file in files:
            if file.endswith(".mp4"):
                if n_init >= n_max:
                    break
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                if os.path.exists(output_path):
                    print(f'Path {output_path} already exists')
                    continue
                if 'Tobii' in input_path:
                    print(f'Copying {input_path} to {output_path}')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    try:
                        if process:
                            shutil.copy2(input_path, output_path)  # shutil for cross-platform copying
                        print(f'Copied {input_path} to {output_path}')
                    except Exception as e:
                        print(f'Error copying {input_path} to {output_path}: {e}')
                    continue

                print(f'Converting {input_path} to {output_path}')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                try:
                    convert_video_fps(input_path, output_path, target_fps, process=process)
                    n_init += 1
                except Exception as e:
                    print(f'Error converting {input_path} to {output_path}: {e}')
        if n_init >= n_max:
            break
    return

if __name__ == "__main__":
    import sys

    process = True
    n_max = 12
    target_fps = 25

    input_dir = sys.argv[1]  # Input dataset path
    output_dir = sys.argv[2]  # Output dataset path
    process_dataset(input_dir, output_dir, process=process, target_fps=target_fps, n_max=n_max)

    sys.exit(0)
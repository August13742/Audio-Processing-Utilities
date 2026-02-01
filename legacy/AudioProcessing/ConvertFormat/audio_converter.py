import os
import subprocess
from pathlib import Path


# Supported formats with their ffmpeg codec settings
FORMATS = {
    "ogg": {
        "codec": ["-c:a", "libvorbis"],
        "quality": ["-q:a", "5"],
        "description": "Ogg Vorbis"
    },
    "mp3": {
        "codec": ["-c:a", "libmp3lame"],
        "quality": ["-q:a", "2"],
        "description": "MP3 (LAME)"
    },
    "flac": {
        "codec": ["-c:a", "flac"],
        "quality": ["-compression_level", "8"],
        "description": "FLAC (Lossless)"
    },
    "wav": {
        "codec": ["-c:a", "pcm_s16le"],
        "quality": [],
        "description": "WAV (Uncompressed)"
    },
    "aac": {
        "codec": ["-c:a", "aac"],
        "quality": ["-b:a", "192k"],
        "description": "AAC"
    },
    "m4a": {
        "codec": ["-c:a", "aac"],
        "quality": ["-b:a", "192k"],
        "description": "M4A (AAC)"
    },
    "opus": {
        "codec": ["-c:a", "libopus"],
        "quality": ["-b:a", "128k"],
        "description": "Opus"
    },
    "wma": {
        "codec": ["-c:a", "wmav2"],
        "quality": ["-b:a", "192k"],
        "description": "WMA"
    }
}


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_format_choice() -> str:
    """Display format options and get user choice."""
    print("\n" + "=" * 50)
    print("Available output formats:")
    print("=" * 50)
    
    formats_list = list(FORMATS.keys())
    for i, fmt in enumerate(formats_list, 1):
        print(f"  {i}. {fmt:6} - {FORMATS[fmt]['description']}")
    
    print("=" * 50)
    
    while True:
        choice = input("\nEnter format name or number (e.g., 'mp3' or '2'): ").strip().lower()
        
        # Check if it's a number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(formats_list):
                return formats_list[idx]
            print(f"Invalid number. Please enter 1-{len(formats_list)}.")
        # Check if it's a valid format name
        elif choice in FORMATS:
            return choice
        else:
            print(f"Invalid format. Supported formats: {', '.join(FORMATS.keys())}")


def convert_file(input_file: Path, output_file: Path, format_settings: dict) -> bool:
    """Convert a single file using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-i", str(input_file),
        "-vn",  # No video
        *format_settings["codec"],
        *format_settings["quality"],
        "-y",  # Overwrite output file if exists
        str(output_file)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Error: {e.stderr.splitlines()[-1] if e.stderr else 'Unknown error'}")
        return False


def convert_files(input_folder: Path, output_folder: Path, output_format: str) -> tuple[int, int]:
    """Convert all files in input folder to specified format."""
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all files in input folder (excluding hidden files)
    files = [f for f in input_folder.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    if not files:
        print("\nNo files found in input folder.")
        return 0, 0
    
    format_settings = FORMATS[output_format]
    success_count = 0
    fail_count = 0
    
    print(f"\nConverting {len(files)} file(s) to {output_format.upper()}...")
    print("-" * 50)
    
    for i, file in enumerate(files, 1):
        # Skip if already in target format
        if file.suffix.lower() == f".{output_format}":
            print(f"[{i}/{len(files)}] Skipping (already {output_format}): {file.name}")
            continue
        
        output_file = output_folder / f"{file.stem}.{output_format}"
        print(f"[{i}/{len(files)}] Converting: {file.name}")
        
        if convert_file(file, output_file, format_settings):
            print(f"    -> {output_file.name} ✓")
            success_count += 1
        else:
            fail_count += 1
    
    return success_count, fail_count


def main():
    print("\n" + "=" * 50)
    print("       AUDIO CONVERTER (FFmpeg)")
    print("=" * 50)
    
    # Check ffmpeg availability
    if not check_ffmpeg():
        print("\nError: ffmpeg not found!")
        print("Please install ffmpeg and ensure it's in your PATH.")
        print("Download: https://ffmpeg.org/download.html")
        return
    
    print("\nffmpeg found")
    
    # Setup paths
    script_dir = Path(__file__).parent
    input_folder = script_dir / "input"
    output_folder = script_dir / "output"
    
    # Check input folder
    if not input_folder.exists():
        print(f"\nCreating input folder: {input_folder}")
        input_folder.mkdir(parents=True, exist_ok=True)
        print("Please add files to the input folder and run again.")
        return
    
    # Count files
    file_count = len([f for f in input_folder.iterdir() if f.is_file()])
    print(f"Input folder: {input_folder}")
    print(f"  Found {file_count} file(s)")
    
    if file_count == 0:
        print("\nNo files found. Please add files to the input folder.")
        return
    
    # Get format choice
    output_format = get_format_choice()
    print(f"\nSelected format: {output_format.upper()} ({FORMATS[output_format]['description']})")
    
    # Confirm
    confirm = input("\nProceed with conversion? (y/n): ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("Conversion cancelled.")
        return
    
    # Convert files
    success, failed = convert_files(input_folder, output_folder, output_format)
    
    # Summary
    print("\n" + "=" * 50)
    print("CONVERSION COMPLETE")
    print("=" * 50)
    print(f"  Successful: {success}")
    print(f"  Failed:     {failed}")
    print(f"  Output:     {output_folder}")
    print("=" * 50)


if __name__ == "__main__":
    main()

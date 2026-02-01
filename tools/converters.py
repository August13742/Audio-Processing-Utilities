
import subprocess
import os
from pathlib import Path
from tqdm import tqdm

class AudioConverter:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert(self, input_path: str, output_path: str, format: str = "ogg", bitrate: str = "192k"):
        """
        Converts audio using ffmpeg.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_path),
            "-vn", # No video
            "-acodec", "libvorbis" if format == "ogg" else "aac" if format == "m4a" else "libmp3lame" if format == "mp3" else "pcm_s16le",
            "-ab", bitrate,
            "-ar", "44100",
            "-ac", "2", # Force stereo for compatibility
            str(output_path)
        ]
        
        # WAV usually doesn't need bitrate, but ffmpeg ignores it for pcm usually.
        if format == "wav":
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(input_path), "-vn", str(output_path)]

        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERR] Conversion failed: {e}")
            return False

    def batch_convert(self, input_dir: str, output_dir: str, target_format: str = "ogg"):
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        files = list(in_path.rglob("*"))
        audio_files = [f for f in files if f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]]
        
        print(f"[CONVERT] Processing {len(audio_files)} files...")
        
        for f in tqdm(audio_files):
            rel_path = f.relative_to(in_path)
            dest_file = out_path / rel_path.with_suffix(f".{target_format}")
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.convert(str(f), str(dest_file), format=target_format)

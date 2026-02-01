from pydub import AudioSegment, silence
from tqdm import tqdm
import subprocess
import os
import tempfile
import json
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# === CONFIGURABLE PARAMETERS ===
INPUT_DIR = "input"
OUTPUT_SUFFIX = "_chunks"
SUPPORTED_EXTENSIONS = (".mp4", ".mkv", ".webm")
SILENCE_THRESH = -40      # dBFS
MIN_SILENCE_LEN = 2500    # ms
KEEP_SILENCE = 500        # ms
TIME_RANGE = None    # seconds: (start, end), or set to None for full length
EXPORT_FORMAT = "mp3"
MAX_WORKERS = min(4, multiprocessing.cpu_count())  # Limit concurrent exports

def get_audio_duration(file_path):
    """Get the duration of an audio/video file in seconds using ffmpeg."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
        file_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
        else:
            print(f"Warning: Could not get duration for {file_path}")
            return None
    except Exception as e:
        print(f"Warning: Error getting duration: {e}")
        return None

def ffmpeg_trim(input_path, start, duration):
    """Trim input audio using ffmpeg and return path to temp file."""
    tmp_path = tempfile.mktemp(suffix=".wav")
    
    print(f"Starting ffmpeg trim operation...")
    print(f"   Input: {os.path.basename(input_path)}")
    print(f"   Output: {os.path.basename(tmp_path)}")
    print(f"   Time range: {start}s to {start + duration}s")
    
    # Optimized ffmpeg command with faster preset and threading
    cmd = [
        "ffmpeg", "-y",
        "-threads", str(min(4, multiprocessing.cpu_count())),  # Use multiple threads
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",             # Standard sample rate
        "-ac", "2",                 # Stereo
        "-avoid_negative_ts", "make_zero",
        tmp_path
    ]
    
    print(f"Running ffmpeg command...")
    start_time = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed_time = time.time() - start_time
    
    print(f"   FFmpeg completed in {elapsed_time:.2f} seconds")
    
    if result.returncode != 0:
        print(f"FFmpeg error (return code: {result.returncode}):")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed to trim audio: {result.stderr}")
    
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError(f"FFmpeg did not create output file or file is empty: {tmp_path}")
    
    file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    print(f"Trimmed file created: {file_size_mb:.2f} MB")
    
    return tmp_path

def export_chunk_worker(args):
    """Worker function for parallel chunk export."""
    chunk, output_path, export_format = args
    try:
        chunk.export(output_path, format=export_format, 
                    parameters=["-threads", "1"])  # Single thread per worker to avoid conflicts
        return True
    except Exception as e:
        print(f"Error exporting {output_path}: {e}")
        return False

def load_audio_optimized(file_path):
    """Load audio with optimized parameters for faster processing."""
    # Use lower quality for faster loading and processing
    return AudioSegment.from_file(
        file_path,
        parameters=["-ar", "44100", "-ac", "2"]  #  stereo
    )

def detect_silence_optimized(audio, min_silence_len, silence_thresh, keep_silence):
    """Optimized silence detection with reduced precision for speed."""
    # Use a lower frame rate for faster silence detection
    return silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
        seek_step=100  # Larger seek step for faster processing
    )

def extract_audio_chunks(file_path, time_range=None):
    print(f"\nProcessing: {file_path}")

    # Check if input file exists and is accessible
    if not os.path.exists(file_path):
        print(f"Error: Input file does not exist: {file_path}")
        return
    
    if os.path.getsize(file_path) == 0:
        print(f"Error: Input file is empty: {file_path}")
        return

    # Get file duration and validate time range
    file_duration = get_audio_duration(file_path)
    if file_duration:
        print(f"File duration: {file_duration:.2f} seconds ({file_duration/60:.2f} minutes)")
        
        if time_range:
            start, end = time_range
            if end > file_duration:
                print(f"Warning: Requested end time ({end}s) is longer than file duration ({file_duration:.2f}s)")
                print(f"   Adjusting end time to file duration")
                time_range = (start, file_duration)
                end = file_duration
            
            if start >= file_duration:
                print(f"Error: Start time ({start}s) is beyond file duration ({file_duration:.2f}s)")
                return

    try:
        # Use ffmpeg to trim first
        if time_range:
            start, end = time_range
            duration = end - start
            print(f"Trimming audio from {start}s to {end}s (duration: {duration}s)")
            trimmed_path = ffmpeg_trim(file_path, start, duration)
        else:
            print(f"Using full file (no trimming)")
            trimmed_path = file_path

        # Load trimmed audio with optimization
        print(f"Loading audio from: {os.path.basename(trimmed_path)}")
        with tqdm(desc="Loading audio", unit=" MB") as pbar:
            audio = load_audio_optimized(trimmed_path)
            pbar.update(1)
        
        duration_minutes = len(audio) / 60000
        print(f"Loaded audio: {duration_minutes:.2f} minutes ({len(audio)} ms)")

    except Exception as e:
        print(f"Error loading audio: {e}")
        if time_range and 'trimmed_path' in locals() and os.path.exists(trimmed_path):
            os.remove(trimmed_path)
        return

    # Split audio by silence with optimization
    print(f"Analyzing silence patterns...")
    print(f"   Silence threshold: {SILENCE_THRESH} dBFS")
    print(f"   Minimum silence length: {MIN_SILENCE_LEN} ms")
    
    with tqdm(desc="Detecting silence", unit=" segments") as pbar:
        chunks = detect_silence_optimized(
            audio,
            MIN_SILENCE_LEN,
            SILENCE_THRESH,
            KEEP_SILENCE
        )
        pbar.update(len(chunks))

    # Output path setup
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(INPUT_DIR, f"{base_name}{OUTPUT_SUFFIX}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Detected {len(chunks)} segments. Exporting to {output_dir}...")
    
    # Calculate total duration for progress tracking
    total_duration = sum(len(chunk) for chunk in chunks) / 1000
    print(f"   Total audio to export: {total_duration/60:.2f} minutes")
    
    # Parallel export of chunks
    export_args = []
    for i, chunk in enumerate(chunks):
        out_path = os.path.join(output_dir, f"{base_name}_part_{i+1:03}.{EXPORT_FORMAT}")
        export_args.append((chunk, out_path, EXPORT_FORMAT))
    
    # Use ThreadPoolExecutor for I/O bound export operations
    successful_exports = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(desc="Exporting segments", total=len(chunks), unit=" files") as pbar:
            for result in executor.map(export_chunk_worker, export_args):
                if result:
                    successful_exports += 1
                pbar.update(1)
    
    print(f"Successfully exported {successful_exports}/{len(chunks)} segments")

    # Clean up temp file
    if time_range and trimmed_path != file_path:
        print(f"Cleaning up temporary file...")
        os.remove(trimmed_path)

    print("Done.")

def process_file_wrapper(args):
    """Wrapper for multiprocessing file processing."""
    file_path, time_range = args
    extract_audio_chunks(file_path, time_range)

if __name__ == "__main__":
    print(f"Scanning '{INPUT_DIR}' for audio files...")
    files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not files:
        print("No supported audio files found.")
    else:
        # Process files in parallel if multiple files exist
        if len(files) > 1:
            print(f"Processing {len(files)} files in parallel...")
            file_args = [(file_path, TIME_RANGE) for file_path in files]
            
            with ProcessPoolExecutor(max_workers=min(2, len(files))) as executor:
                executor.map(process_file_wrapper, file_args)
        else:
            # Single file processing
            extract_audio_chunks(files[0], TIME_RANGE)

import os
from tools.karaoke import KaraokePipeline

def test_full_pipeline():
    audio_path = "TestData/betelgeuse.mp3"
    output_dir = "KaraokeTest"
    
    if not os.path.exists(audio_path):
        print(f"Test file {audio_path} not found.")
        return
        
    print(f"--- Testing Full Karaoke Pipeline on {audio_path} ---")
    pipeline = KaraokePipeline(device="cuda")
    result_json = pipeline.process_song(audio_path, output_dir)
    
    if result_json and os.path.exists(result_json):
        print(f"SUCCESS: Result saved to {result_json}")
        # Print a snippet of the JSON
        import json
        with open(result_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Transcribed Lyrics: {data['lyrics_text'][:100]}...")
            print(f"Alignment Entries: {len(data['lyrics_aligned'])}")
    else:
        print("FAILED: No result JSON generated.")

if __name__ == "__main__":
    test_full_pipeline()

import argparse

parser = argparse.ArgumentParser(description="Relative Visual Relation Detection")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--video_file', type=str, default=None)
group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=8)

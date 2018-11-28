from utils import *
from utils.opts import parser


def main():
    global args
    args = parser.parse_args()

    # Obtain video frames
    if args.frame_folder is not None:
        print('Loading frames in %s' % args.frame_folder)
        import glob

        # here make sure after sorting the frame paths have the correct temporal order
        frame_paths = sorted(glob.glob(os.path.join(args.frame_folder, '*.jpg')))
        frames = load_frames(frame_paths)
        print('Succeed to load frames.')
    else:
        print('Extracting frames using ffmpeg...')
        frames, frame_paths = extract_frames(args.video_file, args.test_segments)
        print('Succeed to extract {} frames from {} to {}'.format(args.test_segments, args.video_file, frame_paths))


if __name__ == '__main__':
    main()

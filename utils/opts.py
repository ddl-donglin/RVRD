import argparse

parser = argparse.ArgumentParser(description="Relative Visual Relation Detection")

# Extract frames from a given video
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--video_file', type=str, default=None)
group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=8)

# Object Detection of Image, Faster R-CNN
parser.add_argument('--obj_net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
parser.add_argument('--obj_load_dir', dest='load_dir',
                        help='directory to load models')
parser.add_argument('--obj_image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="frames")
parser.add_argument('--obj_cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
parser.add_argument('--obj_checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
parser.add_argument('--obj_checkepoch', dest='checkepoch',
                    help='checkepoch to load network',
                    default=10, type=int)
parser.add_argument('--obj_checkpoint', dest='checkpoint',
                    help='checkpoint to load network',
                    default=625, type=int)

# ---

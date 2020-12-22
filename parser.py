import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--HEIGHT', type=int, default=288,
                    help='height of image input(default: 288)')
parser.add_argument('--WIDTH', type=int, default=512,
                    help='width of image input(default: 512)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of training epochs(default: 50)')
parser.add_argument('--load_weights', type=str, default="weights/TrackNet",
                    help='path to load pre-trained weights(default: weights/TrackNet)')
parser.add_argument('--sigma', type=float, default=2.5,
                    help='radius of circle generated in heat map(default: 2.5)')
parser.add_argument('--mag', type=float, default=1.0,
                    help='factor to change range of grayscale(default: 1.0)')
parser.add_argument('--tol', type=float, default=5.0,
                    help='''acceptable tolerance of heat map circle center between 
                            ground truth and prediction(default: 5.0)''')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size(default: 2)')
parser.add_argument('--frame_stack', type=int, default=3,
                    help='number of frames to be stacked(default: 3)')
parser.add_argument('--save_weights', type=str, default='weights/TrackNet',
                    help='path for saving trained weights(default: weights/TrackNet)')
parser.add_argument('--match_folder', type=str, default='train_data',
                    help='folder path of images(default: train_data)')
parser.add_argument('--split_ratio', type=float, default=0.9,
                    help='ratio of train-test split(default: train_data/y_data)')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate(default: 1.0)')
parser.add_argument('--min_lr', type=float, default=0.01,
                    help='minimum learning rate(default: 0.01)')
parser.add_argument('--min_delta', type=float, default=0.0,
                    help='minimum delta of loss(default: 0.0)')
parser.add_argument('--patience', type=int, default=3,
                    help='''number of epochs with no improvement after which 
                            learning rate will be reduced.(default: 3)''')
parser.add_argument('--r_factor', type=float, default=0.1,
                    help='lr reduce factor(default: 0.1)')
parser.add_argument('--pre_trained', type=bool, default=False,
                    help='whether to load pre-trained model(default: False)')

# parser for imgLabel
parser.add_argument('--label_video_path', type=str, default='test/test.mp4',
                    help='video path to label')
parser.add_argument('--csv_path', type=str, default='',
                    help='load csv have labeled')

# parser for predict
parser.add_argument('--video_path', type=str, default='test/test.mp4',
                    help='video path to predict')
parser.add_argument('--label_path', type=str, default='test/test.csv',
                    help='load ground truth csv for predict')
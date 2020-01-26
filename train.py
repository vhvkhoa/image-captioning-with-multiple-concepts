import argparse
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.dataset import CocoCaptionDataset

parser = argparse.ArgumentParser(description='Train model.')

"""Model's parameters"""
parser.add_argument('--image_feature_size', type=int, default=196, help='Multiplication of width and height of image feature\'s dimension, e.g 14x14=196 in the original paper.')
parser.add_argument('--image_feature_depth', type=int, default=1024, help='Depth dimension of image feature, e.g 512 if you extract features at conv-5 of VGG-16 model.')
parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='Hidden layer size for LSTM cell.')
parser.add_argument('--time_steps', type=int, default=31, help='Number of time steps to be iterating through.')
parser.add_argument('--embed_dim', type=int, default=512, help='Embedding space size for embedding tokens.')
parser.add_argument('--beam_size', type=int, default=3, help='Beam size for inference phase.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout portion.')
parser.add_argument('--prev2out', action='store_true', default=True, help='Link previous hidden state to output.')
parser.add_argument('--ctx2out', action='store_true', default=True, help='Link context features to output.')
parser.add_argument('--enable_selector', action='store_true', default=True, help='Enable selector to determine how much important the image context is at every time step.')

"""Training parameters"""
parser.add_argument('--device', type=str, default='cuda:0', help='Device to be used for training model.')
parser.add_argument('--optimizer', type=str, default='rmsprop', help='Optimizer used to update model\'s weights.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='Number of examples per mini-batch.')
parser.add_argument('--eval_steps', type=int, default=100, help='Evaluate and save current model every eval_steps steps.')
parser.add_argument('--metric', type=str, default='CIDEr', help='Metric being based on to choose best model, please insert on of these strings: [Bleu_i, METEOR, ROUGE_L, CIDEr] with i is 1 through 4.')
parser.add_argument('--checkpoint', type=str, help='Path to a pretrained model to initiate weights from.') 
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/', help='Path to directory where checkpoints saved every eval_steps.')
parser.add_argument('--log_path', type=str, default='log/', help='Path to directory where logs saved during the training process. You can use tensorboard to visualize logging informations and re-read IFO printed on the console in .log files.')
parser.add_argument('--scene_dir', type=str, default='data/annotations/scene_features/raw/')

def main():
    args = parser.parse_args()
    # load train dataset
    train_data = CocoCaptionDataset(caption_file='./data/train/captions_train2017.json', 
                                    concept_file='./data/train/train_concepts.json',
                                    action_file='./data/train/train_actions.json',
                                    scene_dir=args.scene_dir, split='train')
    val_data = CocoCaptionDataset(caption_file='./data/val/captions_val2017.json',
                                  concept_file='./data/val/val_concepts.json',
                                  action_file='./data/val/val_actions.json',
                                  scene_dir=args.scene_dir, split='val')
    word_to_idx = train_data.get_vocab_dict()
    # load val dataset to print out scores every epoch

    model = CaptionGenerator(feature_dim=[args.image_feature_size, args.image_feature_depth], 
                                    num_tags=23, embed_dim=args.embed_dim,
                                    hidden_dim=args.lstm_hidden_size, prev2out=args.prev2out, len_vocab=len(word_to_idx),
                                    ctx2out=args.ctx2out, enable_selector=args.enable_selector, dropout=args.dropout,
                                    action_dim=args.embed_dim, scene_dim=512).to(device=args.device)

    solver = CaptioningSolver(model, word_to_idx, train_data, val_data, n_time_steps=args.time_steps,
                                    batch_size=args.batch_size, beam_size=args.beam_size, optimizer=args.optimizer, 
                                    learning_rate=args.learning_rate, metric=args.metric,
                                    eval_every=args.eval_steps,
                                    checkpoint=args.checkpoint, checkpoint_dir=args.checkpoint_dir, 
                                    log_path=args.log_path, device=args.device)

    solver.train(num_epochs=args.num_epochs)

if __name__ == "__main__":
    main()

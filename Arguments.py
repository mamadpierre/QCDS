import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the Environment and RL algorithm')

    parser.add_argument(
        '--device', default='cpu', help='cpu or cuda device')

    parser.add_argument(
        '--path', default='Results', help='path')

    parser.add_argument(
        '--policy', default='loss', help='q_accuracy as reward or q_loss as loss')

    parser.add_argument(
        '--small_design', default=False, action='store_true', help='one layer design')

    parser.add_argument(
        '--QuantumPATH', default='TorchSaves', help='QuantumPATH')

    parser.add_argument(
        '--batch_size', type=int, default=64, help='quantum Train batch size')

    parser.add_argument(
        '--test_batch_size', type=int, default=1000, help='quantum Test batch size')

    parser.add_argument(
        '--Cepochs', type=int, default=50, help='number of epochs for training controller')

    parser.add_argument(
        '--Qepochs', type=int, default=50, help='number of epochs for training quantum network')

    parser.add_argument(
        '--cuda', default=False, action='store_true', help='cuda')

    parser.add_argument(
        '--lr', type=float, default=0.01, help='Learning rate')

    parser.add_argument(
        '--Clr', type=float, default=0.1, help='Learning rate')

    parser.add_argument(
        '--entropy_weight', type=float, default=0.01, help='coefficient of entropy loss')

    parser.add_argument(
        '--seed', type=int, default=1, help='Seed')
    parser.add_argument(
        '--n_qubits', type=int, default=4, help='number of qubits')

    parser.add_argument(
        '--n_output', type=int, default=3, help='number of output measurements')
    parser.add_argument(
        '--q_depth', type=int, default=6, help='Depth of the stacked quantum layer')

    args = parser.parse_args()
    return args


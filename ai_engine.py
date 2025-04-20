import torch
import torch.nn as nn
import chess
import numpy as np

# Định nghĩa mô hình AI cho cờ vua
class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 12, 128)  # 8x8x12 = 768 đầu vào
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 4672)         # 4672 nước đi khả dĩ theo mapping

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hàm chuyển FEN sang ma trận 8x8x12 (dùng để làm đầu vào cho mô hình)
def fen_to_input(fen):
    board = chess.Board(fen)
    input_matrix = np.zeros((8, 8, 12))
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        input_matrix[row, col, piece.piece_type - 1] = 1 if piece.color else -1
    return input_matrix

# Tạo danh sách tất cả các nước đi khả dĩ theo định dạng UCI (ví dụ: "a2a3", "a2a4", …)
def generate_all_possible_moves():
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
    moves = []
    for from_file in files:
        for from_rank in ranks:
            for to_file in files:
                for to_rank in ranks:
                    moves.append(from_file + from_rank + to_file + to_rank)
    return moves

all_moves = generate_all_possible_moves()

# Hàm ánh xạ nước đi (đối tượng chess.Move) thành chỉ số (label)
def move_to_index(move):
    try:
        return all_moves.index(move.uci())
    except ValueError:
        return None

# Hàm dự đoán nước đi của AI từ bàn cờ hiện tại
def predict_move(board, model):
    board_input = fen_to_input(board.fen()).flatten()
    board_input_tensor = torch.tensor(board_input, dtype=torch.float32).reshape(1, -1)
    output = model(board_input_tensor)
    return torch.argmax(output, dim=1).item()

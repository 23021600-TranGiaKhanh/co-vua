# ai_engine.py

import torch
import torch.nn as nn
import chess
import numpy as np

# Mô hình học sâu cho AI chơi cờ vua
class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.fc1 = nn.Linear(8 * 8 * 12, 128)  # Ma trận bàn cờ
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 4672)  # 4672 là số nước đi có thể

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Chuyển bàn cờ FEN thành ma trận đầu vào cho AI
def fen_to_input(fen):
    board = chess.Board(fen)
    input_matrix = np.zeros((8, 8, 12))  # 8x8 cho bàn cờ, 12 lớp cho các quân cờ

    # Duyệt qua các quân cờ trên bàn cờ và cập nhật ma trận
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type  # Loại quân cờ
        color = 1 if piece.color else -1  # Màu quân (1 cho trắng, -1 cho đen)

        # Cập nhật ma trận với giá trị của quân cờ
        input_matrix[row, col, piece_type - 1] = color

    return input_matrix

# Dự đoán nước đi tiếp theo từ bàn cờ
def predict_move(board, model):
    board_input = fen_to_input(board.fen())  # Chuyển bàn cờ thành ma trận đầu vào
    board_input_tensor = torch.tensor(board_input, dtype=torch.float32).reshape(1, -1)
    output = model(board_input_tensor)
    predicted_move = torch.argmax(output, dim=1).item()  # Dự đoán nước đi
    return predicted_move

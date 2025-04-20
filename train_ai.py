import os
import chess.pgn
import numpy as np
import torch
import pickle
import time
import torch.optim as optim
import torch.nn as nn

from ai_engine import ChessAI, fen_to_input, move_to_index
from config import TRAINING_PGN_PATH, WEIGHTS_PATH, PICKLE_PATH


class ChessDataset:
    def __init__(self, file_path, batch_size=100):
        self.batch_size = batch_size
        self.current_index = 0

        # Kiểm tra nếu có file pickle đã lưu dữ liệu
        if os.path.exists(PICKLE_PATH):
            print("Tải dữ liệu đã xử lý từ pickle...")
            try:
                with open(PICKLE_PATH, 'rb') as f:
                    self.data, self.labels = pickle.load(f)
                    print(f"Tải thành công {len(self.data)} mẫu dữ liệu từ pickle.")
            except EOFError:
                print("Lỗi khi tải dữ liệu từ pickle, tạo lại dữ liệu từ file PGN.")
                self.data, self.labels = self.load_data(file_path)
                self.save_data_to_pickle()
        else:
            # Nếu không có, thì load và xử lý lại dữ liệu từ file PGN
            print("Đang xử lý dữ liệu từ file PGN...")
            self.data, self.labels = self.load_data(file_path)
            # Lưu dữ liệu đã xử lý vào pickle để sử dụng cho lần sau
            self.save_data_to_pickle()

    def load_data(self, file_path):
        """
        Đọc dữ liệu từ file PGN và xử lý theo batch (100 ván cờ mỗi lần).
        """
        X_data = []  # Danh sách chứa các vector đầu vào (FEN)
        y_data = []  # Danh sách chứa các nhãn (move)
        game_count = 0
        start_time = time.time()

        with open(file_path, encoding="utf-8") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)  # Đọc game từ file PGN
                if game is None:  # Kết thúc khi không còn game nào
                    break
                board = game.board()  # Lấy bàn cờ bắt đầu từ ván cờ

                # Duyệt qua các nước đi trong game
                for move in game.mainline_moves():
                    try:
                        # Chuyển trạng thái bàn cờ (FEN) thành vector đầu vào (flatten)
                        board_input = fen_to_input(board.fen()).flatten()
                        # Ánh xạ nước đi thành nhãn label (index của move)
                        label = move_to_index(move)
                        if label is not None:
                            X_data.append(board_input)  # Thêm vector vào dữ liệu đầu vào
                            y_data.append(label)  # Thêm nhãn vào dữ liệu đầu ra
                    except Exception as e:
                        print(f"Lỗi trong quá trình xử lý move: {move}, lỗi: {e}")
                        # Lưu dữ liệu đã xử lý và dừng tiếp tục
                        print("Lưu dữ liệu đã xử lý trước đó...")
                        self.save_data_to_pickle(X_data, y_data)
                        return np.array(X_data), np.array(y_data)

                    board.push(move)  # Cập nhật trạng thái bàn cờ sau mỗi nước đi

                game_count += 1
                # In thông tin về quá trình xử lý mỗi 1000 ván cờ
                if game_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Đã xử lý {game_count} game, thời gian: {elapsed:.2f}s")

        return np.array(X_data), np.array(y_data)  # Trả về dữ liệu và nhãn dưới dạng mảng NumPy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def next_batch(self):
        if self.current_index + self.batch_size > len(self.data):
            return None, None  # Đã hết dữ liệu
        batch_data = self.data[self.current_index:self.current_index + self.batch_size]
        batch_labels = self.labels[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return batch_data, batch_labels

    def save_data_to_pickle(self, data=None, labels=None):
        """Lưu dữ liệu đã xử lý vào file pickle."""
        if data is None or labels is None:
            data = self.data
            labels = self.labels
        try:
            with open(PICKLE_PATH, 'wb') as f:
                pickle.dump((data, labels), f)
            print(f"Dữ liệu đã được lưu vào {PICKLE_PATH}")
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu vào pickle: {e}")


# Lưu trọng số mô hình sau mỗi epoch
def save_model_checkpoint(model, optimizer, loss, accuracy, weights_path=WEIGHTS_PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, weights_path)
    
    print(f"Trọng số mô hình đã được lưu tại: {weights_path}")


# Tiến hành huấn luyện mô hình
if __name__ == "__main__":
    pgn_file_path = TRAINING_PGN_PATH
    print("Bắt đầu xử lý file PGN từ:", pgn_file_path)

    dataset = ChessDataset(pgn_file_path)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ChessAI()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("Bắt đầu huấn luyện mô hình...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        batch_data, batch_labels = dataset.next_batch()
        if batch_data is None:
            break

        batch_data_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(batch_data_tensor)
        loss = criterion(outputs, batch_labels_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == batch_labels_tensor).sum().item()
        total_samples += batch_labels_tensor.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_preds / total_samples
        print(f"Epoch {epoch+1}/{epochs} - Loss trung bình: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        save_model_checkpoint(model, optimizer, avg_loss, accuracy)

    print("Huấn luyện hoàn thành.")

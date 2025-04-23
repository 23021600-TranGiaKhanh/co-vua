import chess.engine
import os

class EngineHandler:
    def __init__(self, engine_path, weights_path, threads=1):
        """
        engine_path: đường dẫn tới UCI binary (ví dụ lc0.exe hoặc maia.exe)
        weights_path: đường dẫn tới file .pb.gz chứa mô hình đã huấn luyện
        threads: số luồng CPU/GPU cho engine
        """
        # Chuẩn hóa đường dẫn
        self.engine_path = os.path.normpath(engine_path)
        self.weights_path = os.path.normpath(weights_path)
        
        # Khởi UCI engine, truyền CLI args cho weights và threads
        cmd = [
            self.engine_path,
            f"--threads={threads}",
            f"--weights={self.weights_path}"
        ]
        self.engine = chess.engine.SimpleEngine.popen_uci(cmd)

    def get_best_move(self, board, thinking_time=0.2, nodes=None):
        """
        Trả về nước đi tốt nhất cho board hiện tại.
        - thinking_time: thời gian suy tính (giây)
        - nodes: số nút MCTS (nếu engine hỗ trợ)
        """
        try:
            limit = chess.engine.Limit(time=thinking_time, nodes=nodes)
            result = self.engine.play(board, limit)
            return result.move
        except Exception as e:
            print("AI error:", e)
            return None

    def quit(self):
        """Đóng process engine"""
        self.engine.quit()
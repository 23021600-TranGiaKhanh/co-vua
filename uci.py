import chess.engine
import os


class EngineHandler:
    def __init__(self, engine_path, threads=1):
        # Chuẩn hóa đường dẫn
        self.engine_path = os.path.normpath(engine_path)

        # Khởi UCI engine, chỉ truyền threads
        cmd = [
            self.engine_path,
            f"--threads={threads}"
        ]
        self.engine = chess.engine.SimpleEngine.popen_uci(cmd)

    def get_best_move(self, board, thinking_time=0.5, nodes=None):
        
        try:
            limit = chess.engine.Limit(time=thinking_time, nodes=nodes)
            result = self.engine.play(board, limit)
            return result.move
        except Exception as e:
            print("AI error:", e)
            return None

    def quit(self):
        self.engine.quit()
# ai.py

import chess
import time
from collections import defaultdict
from chess.polyglot import zobrist_hash

# Giá trị cơ bản (centipawns)
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000
}

# Piece-square tables (placeholder giá trị 0)
PST = {t: [0]*64 for t in PIECE_VALUES}

MATE_VALUE = 10**9

class TTEntry:
    def __init__(self, depth, value, flag, best_move):
        self.depth = depth
        self.value = value
        self.flag  = flag
        self.best_move = best_move

class TimeLimitedEngine:
    def __init__(self, time_limit: float = 30.0, max_depth: int = 8):
        self.time_limit = time_limit
        self.max_depth  = max_depth
        # Persistent tables
        self.tt      = {}
        self.killers = defaultdict(lambda: [None, None])
        self.history = defaultdict(int)

    def reset_search(self):
        self.start_time = time.time()

    def evaluate(self, board: chess.Board) -> int:
        score = 0
        # Material + PST
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                base = PIECE_VALUES[p.piece_type]
                idx = sq if p.color == chess.WHITE else 63 - sq
                score_piece = base + PST[p.piece_type][idx]
                score += score_piece if p.color == chess.WHITE else -score_piece
        # Mobility
        moves = board.legal_moves
        mob = sum(1 for _ in moves)
        score += mob * 5 if board.turn else -mob * 5
        return score

    def is_time_up(self) -> bool:
        return (time.time() - self.start_time) >= self.time_limit

    def get_best_move(self, board: chess.Board) -> chess.Move | None:
        self.reset_search()
        best_move = None
        prev_score = 0
        for depth in range(1, self.max_depth + 1):
            if self.is_time_up():
                break
            b = board.copy()
            if depth == 1:
                alpha, beta = -MATE_VALUE, MATE_VALUE
            else:
                delta = 50
                alpha = prev_score - delta
                beta  = prev_score + delta
            try:
                score, mv = self._alphabeta(b, depth, alpha, beta, True, depth)
                if score <= alpha:
                    score, mv = self._alphabeta(b, depth, -MATE_VALUE, beta, True, depth)
                elif score >= beta:
                    score, mv = self._alphabeta(b, depth, alpha, MATE_VALUE, True, depth)
            except TimeoutError:
                break
            if mv is not None:
                best_move = mv
                prev_score = score
        return best_move

    def _alphabeta(self, board, depth, alpha, beta, maximizing, root_depth):
        if self.is_time_up():
            raise TimeoutError()
        # Terminal
        if board.is_checkmate():
            return ((-MATE_VALUE, None) if maximizing else (MATE_VALUE, None))
        if board.is_stalemate() or board.is_insufficient_material() or \
           board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0, None
        if depth == 0:
            return self._quiescence(board, alpha, beta), None

        key = zobrist_hash(board)
        if key in self.tt:
            entry = self.tt[key]
            if entry.depth >= depth:
                if entry.flag == 0:
                    return entry.value, entry.best_move
                elif entry.flag < 0:
                    alpha = max(alpha, entry.value)
                else:
                    beta  = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value, entry.best_move

        # Move ordering helpers
        def cap_score(m):
            if board.is_en_passant(m):
                victim_val = PIECE_VALUES[chess.PAWN]
            else:
                vt = board.piece_type_at(m.to_square)
                victim_val = PIECE_VALUES[vt] if vt else 0
            at = board.piece_type_at(m.from_square)
            attacker_val = PIECE_VALUES[at] if at else 0
            return victim_val*100 - attacker_val

        moves = list(board.legal_moves)
        # TT move
        if key in self.tt and self.tt[key].best_move in moves:
            mv_tt = self.tt[key].best_move
            moves.remove(mv_tt);
            moves.insert(0, mv_tt)
        # Killers
        kms = self.killers[root_depth - depth]
        for km in kms:
            if km in moves:
                moves.remove(km);
                moves.insert(1, km)
        # MVV-LVA + history
        rest = moves[2:]
        caps = [m for m in rest if board.is_capture(m)]
        caps.sort(key=cap_score, reverse=True)
        others = [m for m in rest if not board.is_capture(m)]
        others.sort(key=lambda m: self.history[m], reverse=True)
        moves = moves[:2] + caps + others

        best_move = None
        value = -MATE_VALUE if maximizing else MATE_VALUE
        flag = 1
        for mv in moves:
            if self.is_time_up():
                raise TimeoutError()
            board.push(mv)
            score, _ = self._alphabeta(board, depth-1, alpha, beta, not maximizing, root_depth)
            board.pop()
            if maximizing:
                if score > value:
                    value, best_move = score, mv
                alpha = max(alpha, value)
            else:
                if score < value:
                    value, best_move = score, mv
                beta = min(beta, value)
            if best_move and ((maximizing and value >= beta) or (not maximizing and value <= alpha)):
                lst = self.killers[root_depth - depth]
                if best_move not in lst:
                    lst[1] = lst[0]
                    lst[0] = best_move
                self.history[best_move] += depth * depth
            if alpha >= beta:
                flag = -1 if maximizing else 1
                break

        self.tt[key] = TTEntry(depth, value, flag, best_move)
        return value, best_move

    def _quiescence(self, board, alpha, beta):
        if self.is_time_up():
            raise TimeoutError()
        stand = self.evaluate(board)
        if stand >= beta:
            return beta
        alpha = max(alpha, stand)
        for mv in board.legal_moves:
            if board.is_capture(mv):
                board.push(mv)
                score = -self._quiescence(board, -beta, -alpha)
                board.pop()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha
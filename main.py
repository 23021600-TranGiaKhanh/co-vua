import pygame
import sys
import os
import chess

# Khởi tạo Pygame và cài đặt kích thước cửa sổ
pygame.init()
TILE_SIZE = 80
TOP_MARGIN = 50  # Khoảng cách từ trên xuống
BOARD_SIZE = TILE_SIZE * 8  # Kích thước bàn cờ
WINDOW_WIDTH = BOARD_SIZE  # Chiều rộng của cửa sổ (tương ứng với kích thước bàn cờ)
WINDOW_HEIGHT = BOARD_SIZE + TOP_MARGIN  # Chiều cao cửa sổ (thêm không gian cho thanh công cụ trên)

# Tạo cửa sổ game với kích thước đã định
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Cờ Vua")  # Tiêu đề của cửa sổ

# Định nghĩa các màu sắc để sử dụng trong game
WHITE = (240, 217, 181)  # Màu trắng cho ô và quân trắng
BLACK = (100, 100, 100)  # Màu đen cho ô và quân đen
HIGHLIGHT = (0, 255, 0)  # Màu highlight (khi chọn quân cờ hợp lệ)
MENU_BG = (180, 180, 180)  # Màu nền cho menu và thanh công cụ
RED = (255, 0, 0)  # Màu đỏ cho việc highlight quân vua bị chiếu
font = pygame.font.SysFont(None, 36)  # Phông chữ sử dụng cho các thông báo

# Biến trạng thái để theo dõi các thông tin trong game
selected_square = None  # Lưu tọa độ (row, col) của quân được chọn
undone_moves = []  # Lưu trữ các nước đi đã undo (để redo)
menu_active = False  # Trạng thái của menu (hiện/ẩn)
game_over = False  # Trạng thái game (chưa kết thúc/đã kết thúc)
board = chess.Board()

# Hàm load ảnh quân cờ
def load_piece_images(path):
    images = {}
    piece_names = {
        "p": "tot_den.png", "n": "ma_den.png", "b": "tinh_den.png", "r": "xe_den.png",
        "q": "hau_den.png", "k": "vua_den.png",
        "P": "tot_trang.png", "N": "ma_trang.png", "B": "tinh_trang.png", "R": "xe_trang.png",
        "Q": "hau_trang.png", "K": "vua_trang.png"
    }
    for piece, filename in piece_names.items():
        img_fullpath = os.path.join(path, filename)
        images[piece] = pygame.transform.scale(
            pygame.image.load(img_fullpath),
            (TILE_SIZE, TILE_SIZE)
        )
    return images

# Load ảnh quân cờ và các biểu tượng (hamburger, undo, redo, exit)
pieces_img = load_piece_images("img")
icons = {
    "hamburger": pygame.transform.scale(pygame.image.load(os.path.join("img", "hamburger_button.png")), (40, 30)),
    "undo": pygame.transform.scale(pygame.image.load(os.path.join("img", "undo.png")), (40, 30)),
    "redo": pygame.transform.scale(pygame.image.load(os.path.join("img", "redo.png")), (40, 30)),
    "exit": pygame.transform.scale(pygame.image.load(os.path.join("img", "exit.png")), (40, 30))
}

# Load hình ảnh chiến thắng và hòa cờ (stalemate)
victory_images = {
    "white": pygame.transform.scale(pygame.image.load(os.path.join("img", "trang_thang.png")), (300, 200)),
    "black": pygame.transform.scale(pygame.image.load(os.path.join("img", "den_thang.png")), (300, 200)),
    "stalemate": pygame.transform.scale(pygame.image.load(os.path.join("img", "stalemate.png")), (300, 200))
}

# Các chức năng vẽ bàn cờ và quân cờ
def draw_board(screen, selected_square, tile_size, white, black, highlight, offset=(0,0)):
    for row in range(8):
        for col in range(8):
            color = white if (row + col) % 2 == 0 else black
            pygame.draw.rect(screen, color, (offset[0] + col * tile_size, offset[1] + row * tile_size, tile_size, tile_size))
    if selected_square:
        pygame.draw.rect(screen, highlight, (offset[0] + selected_square[1] * tile_size, offset[1] + selected_square[0] * tile_size, tile_size, tile_size), 3)

def draw_pieces(screen, board, pieces_img, offset=(0,0)):
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            col = chess.square_file(sq)
            row = 7 - chess.square_rank(sq)
            screen.blit(pieces_img[piece.symbol()], (offset[0] + col * TILE_SIZE, offset[1] + row * TILE_SIZE))

def get_square_under_mouse(pos, offset=(0,0)):
    x, y = pos[0] - offset[0], pos[1] - offset[1]
    return y // TILE_SIZE, x // TILE_SIZE

def square_name_from_pos(row, col):
    return chess.square_name(chess.square(col, 7 - row))

def highlight_possible_moves(screen, board, selected_square, color, offset=(0,0)):
    if selected_square is not None:
        from_sq = chess.parse_square(square_name_from_pos(*selected_square))
        for move in board.legal_moves:
            if move.from_square == from_sq:
                to_sq = move.to_square
                col = chess.square_file(to_sq)
                row = 7 - chess.square_rank(to_sq)
                pygame.draw.rect(screen, color, (offset[0] + col * TILE_SIZE, offset[1] + row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)

# Các chức năng vẽ thanh công cụ và menu
def draw_top_bar(screen):
    pygame.draw.rect(screen, MENU_BG, (0, 0, WINDOW_WIDTH, TOP_MARGIN))  # Vẽ thanh công cụ
    rects = []  # Danh sách chứa các hình chữ nhật của các icon
    for i, key in enumerate(icons):
        rect = icons[key].get_rect(topleft=(10 + 50 * i, (TOP_MARGIN - 30) // 2))  # Vị trí của từng icon
        screen.blit(icons[key], rect)  # Hiển thị icon
        rects.append(rect)  # Thêm vào danh sách rects
    turn_text = "Turn: White" if board.turn else "Turn: Black"
    turn_surface = font.render(turn_text, True, (0, 0, 0))  # Vẽ văn bản lượt chơi
    screen.blit(turn_surface, (WINDOW_WIDTH - turn_surface.get_width() - 10, (TOP_MARGIN - turn_surface.get_height()) // 2))
    return rects  # Trả về danh sách các hình chữ nhật

def draw_menu(screen):
    overlay = pygame.Surface((WINDOW_WIDTH, BOARD_SIZE))
    overlay.set_alpha(200)
    overlay.fill(MENU_BG)
    screen.blit(overlay, (0, TOP_MARGIN))
    buttons = {
        "resume": pygame.Rect(WINDOW_WIDTH // 2 - 100, TOP_MARGIN + 150, 200, 50),
        "newgame": pygame.Rect(WINDOW_WIDTH // 2 - 100, TOP_MARGIN + 210, 200, 50)
    }
    for key, rect in buttons.items():
        pygame.draw.rect(screen, (200, 200, 200), rect)
        label = font.render(key.capitalize(), True, (0, 0, 0))
        screen.blit(label, (rect.x + 50, rect.y + 10))
    return buttons  # Trả về các nút bấm

def highlight_king_in_check():
    if board.is_check():
        king_sq = board.king(board.turn)
        if king_sq is not None:
            col = chess.square_file(king_sq)
            row = 7 - chess.square_rank(king_sq)
            pygame.draw.rect(screen, RED, (col * TILE_SIZE, TOP_MARGIN + row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)

# Hiển thị thông báo chiến thắng hoặc hòa cờ và yêu cầu nhấn phím để chơi lại
def draw_victory_overlay():
    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner is True:
            win_img = victory_images["white"]
        elif outcome.winner is False:
            win_img = victory_images["black"]
        elif outcome.termination == chess.TERMINATION_STALEMATE:
            win_img = victory_images["stalemate"]
        else:
            win_img = None
        if win_img:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))  # Tạo overlay mờ
            overlay.set_alpha(150)  # Làm mờ overlay
            overlay.fill((0, 0, 0))  # Đặt màu nền cho overlay
            screen.blit(overlay, (0, 0))  # Vẽ overlay lên màn hình
            img_rect = win_img.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 30))  # Vị trí của hình ảnh chiến thắng
            screen.blit(win_img, img_rect)  # Vẽ hình ảnh chiến thắng lên màn hình
            msg = "Press any key to play again"
            msg_surf = font.render(msg, True, (255, 255, 255))  # Tạo văn bản thông báo
            msg_rect = msg_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + img_rect.height // 2))  # Vị trí của thông báo
            screen.blit(msg_surf, msg_rect)  # Vẽ thông báo lên màn hình
            pygame.display.flip()  # Cập nhật màn hình

# Hàm để chọn quân và di chuyển quân cờ
def select_or_move_piece(row, col):
    global selected_square  # Biến lưu tọa độ quân cờ đã chọn
    clicked_sq_name = square_name_from_pos(row, col)
    clicked_piece = board.piece_at(chess.parse_square(clicked_sq_name))
    
    # Nếu chưa chọn quân cờ
    if selected_square is None:
        # Kiểm tra nếu ô có quân cờ và quân cờ đó thuộc quyền sở hữu của người chơi hiện tại
        if clicked_piece and ((board.turn and clicked_piece.color == chess.WHITE) or (not board.turn and clicked_piece.color == chess.BLACK)):
            selected_square = (row, col)  # Chọn quân cờ
    else:
        # Nếu đã chọn quân cờ, kiểm tra di chuyển đến ô
        move_uci = square_name_from_pos(*selected_square) + clicked_sq_name
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                selected_square = None  # Bỏ chọn quân cờ sau khi di chuyển
            else:
                selected_square = (row, col)  # Nếu chọn lại quân cờ khác
        except chess.InvalidMoveError:
            print("Invalid move UCI:", move_uci)

# Vòng lặp chính của trò chơi
running = True
while running:
    if board.is_game_over() and not game_over:
        game_over = True
    if game_over:
        draw_board(screen, None, TILE_SIZE, WHITE, BLACK, HIGHLIGHT, offset=(0, TOP_MARGIN))
        draw_pieces(screen, board, pieces_img, offset=(0, TOP_MARGIN))
        draw_victory_overlay()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                board.reset()
                undone_moves.clear()
                selected_square = None
                game_over = False
        continue

    icon_rects = draw_top_bar(screen)

    if menu_active:
        draw_board(screen, None, TILE_SIZE, WHITE, BLACK, HIGHLIGHT, offset=(0, TOP_MARGIN))
        draw_pieces(screen, board, pieces_img, offset=(0, TOP_MARGIN))
        buttons = draw_menu(screen)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                for key, rect in buttons.items():
                    if rect.collidepoint(pos):
                        if key == "resume":
                            menu_active = False
                        elif key == "newgame":
                            board.reset()
                            undone_moves.clear()
                            selected_square = None
                            game_over = False
                            menu_active = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    menu_active = False
        continue

    draw_board(screen, selected_square, TILE_SIZE, WHITE, BLACK, HIGHLIGHT, offset=(0, TOP_MARGIN))
    highlight_possible_moves(screen, board, selected_square, HIGHLIGHT, offset=(0, TOP_MARGIN))
    draw_pieces(screen, board, pieces_img, offset=(0, TOP_MARGIN))
    highlight_king_in_check()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = pygame.mouse.get_pos()
            if icon_rects[0].collidepoint(pos):
                menu_active = not menu_active
            elif icon_rects[1].collidepoint(pos) and board.move_stack:
                undone_moves.append(board.pop())
            elif icon_rects[2].collidepoint(pos) and undone_moves:
                board.push(undone_moves.pop())
            elif icon_rects[3].collidepoint(pos):
                running = False
            elif pos[1] > TOP_MARGIN:
                row, col = get_square_under_mouse(pos, offset=(0, TOP_MARGIN))
                select_or_move_piece(row, col)  # Gọi hàm chọn hoặc di chuyển quân cờ

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                menu_active = not menu_active
            elif event.key == pygame.K_z and board.move_stack:
                undone_moves.append(board.pop())
            elif event.key == pygame.K_x and undone_moves:
                board.push(undone_moves.pop())

pygame.quit()  # Dừng Pygame
sys.exit()  # Thoát chương trình

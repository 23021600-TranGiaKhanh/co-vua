import pygame
import sys
import os
import chess

# Khởi tạo Pygame và cài đặt kích thước cửa sổ
pygame.init()
TILE_SIZE = 80
TOP_MARGIN = 50  # Khoảng cách từ trên xuống (dành cho thanh bar)
BOARD_SIZE = TILE_SIZE * 8
WINDOW_WIDTH = BOARD_SIZE
WINDOW_HEIGHT = BOARD_SIZE + TOP_MARGIN

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Chess Game")

# Định nghĩa màu sắc và font
WHITE = (240, 217, 181)
BLACK = (100, 100, 100)
HIGHLIGHT = (0, 255, 0)
MENU_BG = (180, 180, 180)
RED = (255, 0, 0)
font = pygame.font.SysFont(None, 36)

# Khởi tạo board
board = chess.Board()

#LOAD ẢNH
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

pieces_img = load_piece_images("assets/img")

# Các icon trên thanh bar (loại bỏ nút Exit)
icons = {
    "hamburger": pygame.transform.scale(
        pygame.image.load(os.path.join("assets/img", "hamburger_button.png")), (40, 30)
    ),
    "undo": pygame.transform.scale(
        pygame.image.load(os.path.join("assets/img", "undo.png")), (40, 30)
    ),
    "redo": pygame.transform.scale(
        pygame.image.load(os.path.join("assets/img", "redo.png")), (40, 30)
    )
}

# Load hình ảnh chiến thắng/hòa (nếu cần)
victory_images = {
    "white": pygame.transform.scale(
        pygame.image.load(os.path.join("assets/img", "trang_thang.png")), (300, 200)
    ),
    "black": pygame.transform.scale(
        pygame.image.load(os.path.join("assets/img", "den_thang.png")), (300, 200)
    ),
    "stalemate": pygame.transform.scale(
        pygame.image.load(os.path.join("assets/img", "stalemate.png")), (300, 200)
    )
}

checkmate_sound = False
def play_sound_checkmate():
    pygame.mixer.music.load("assets/sound/sound_checkmate.ogg")  # Đổi đường dẫn nếu khác
    pygame.mixer.music.set_volume(0.5)  # Âm lượng từ 0.0 đến 1.0
    pygame.mixer.music.play()

eat_capture = False
def play_sound_capture():
    pygame.mixer.music.load("assets/sound/sound_capture.ogg")  # Đổi đường dẫn nếu khác
    pygame.mixer.music.set_volume(0.5)  # Âm lượng từ 0.0 đến 1.0
    pygame.mixer.music.play()

move_sound = False
def play_sound_move():
    pygame.mixer.music.load("assets/sound/sound_move.ogg")  # Đổi đường dẫn nếu khác
    pygame.mixer.music.set_volume(0.5)  # Âm lượng từ 0.0 đến 1.0
    pygame.mixer.music.play()
    
finish_sound = False
def play_sound_finish():
    pygame.mixer.music.load("assets/sound/sound_finish.ogg")  # Đổi đường dẫn nếu khác
    pygame.mixer.music.set_volume(0.5)  # Âm lượng từ 0.0 đến 1.0
    pygame.mixer.music.play()    
    
menu_music_playing = False
def play_menu_music():
    pygame.mixer.music.load("assets/sound/sound_game.ogg")  # Đổi đường dẫn nếu khác
    pygame.mixer.music.set_volume(0.5)  # Âm lượng từ 0.0 đến 1.0
    pygame.mixer.music.play(-1)  # -1 để phát lặp lại vô hạn

#START MENU (CHỌN CHẾ ĐỘ)
menu_background = pygame.transform.scale(
    pygame.image.load(os.path.join("assets/img", "menu2.png")), (WINDOW_WIDTH, WINDOW_HEIGHT)
)
one_player_img = pygame.image.load(os.path.join("assets/img", "1p.png"))
two_player_img = pygame.image.load(os.path.join("assets/img", "2p.png"))
one_player_img = pygame.transform.scale(one_player_img, (150, 50))
two_player_img = pygame.transform.scale(two_player_img, (150, 50))

one_player_rect = one_player_img.get_rect(center=(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2))
two_player_rect = two_player_img.get_rect(center=(WINDOW_WIDTH // 2 + 100, WINDOW_HEIGHT // 2))

def draw_start_menu():
    screen.blit(menu_background, (0, 0))
    screen.blit(one_player_img, one_player_rect)
    screen.blit(two_player_img, two_player_rect)
    title = font.render("Select game mode", True, (0, 0, 0))
    title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
    screen.blit(title, title_rect)
    pygame.display.flip()

#  HÀM VẼ TRONG GAME 
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

def draw_top_bar(screen):
    # Vẽ thanh bar chứa các icon: hamburger, undo, redo
    pygame.draw.rect(screen, MENU_BG, (0, 0, WINDOW_WIDTH, TOP_MARGIN))
    rects = []
    for i, key in enumerate(icons):
        rect = icons[key].get_rect(topleft=(10 + 50 * i, (TOP_MARGIN - 30) // 2))
        screen.blit(icons[key], rect)
        rects.append(rect)
    turn_text = "Turn: White" if board.turn else "Turn: Black"
    turn_surface = font.render(turn_text, True, (0, 0, 0))
    screen.blit(turn_surface, (WINDOW_WIDTH - turn_surface.get_width() - 10, (TOP_MARGIN - turn_surface.get_height()) // 2))
    return rects

def draw_pause_menu(screen):
    # Pause menu có các nút sắp xếp dọc: Resume, Newgame, Home, Exit
    overlay = pygame.Surface((WINDOW_WIDTH, BOARD_SIZE))
    overlay.set_alpha(200)
    overlay.fill(MENU_BG)
    screen.blit(overlay, (0, TOP_MARGIN))
    
    button_width = 200
    button_height = 50
    gap = 20
    total_height = button_height * 4 + gap * 3
    start_y = TOP_MARGIN + (BOARD_SIZE - total_height) // 2
    start_x = (WINDOW_WIDTH - button_width) // 2
    
    buttons = {
        "resume": pygame.Rect(start_x, start_y, button_width, button_height),
        "newgame": pygame.Rect(start_x, start_y + button_height + gap, button_width, button_height),
        "home": pygame.Rect(start_x, start_y + 2 * (button_height + gap), button_width, button_height),
        "exit": pygame.Rect(start_x, start_y + 3 * (button_height + gap), button_width, button_height)
    }
    
    for key, rect in buttons.items():
        pygame.draw.rect(screen, (200, 200, 200), rect)
        label = font.render(key.capitalize(), True, (0, 0, 0))
        label_rect = label.get_rect(center=rect.center)
        screen.blit(label, label_rect)
    return buttons

def draw_confirmation_dialog(action):
    # Vẽ hộp xác nhận với overlay
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((50, 50, 50))
    screen.blit(overlay, (0, 0))

    box_width, box_height = 500, 200
    box_rect = pygame.Rect(
        (WINDOW_WIDTH - box_width) // 2, 
        (WINDOW_HEIGHT - box_height) // 2, 
        box_width, box_height
    )
    pygame.draw.rect(screen, (200, 200, 200), box_rect)

    
    if action == "exit":
        prompt = "Are you sure you want to exit?"
    elif action == "newgame":
        prompt = "Are you sure you want a new game?"
    elif action == "home":
        prompt = "Are you sure you want to go home?"
    else:
        prompt = "Xác nhận?"
    prompt_surf = font.render(prompt, True, (0, 0, 0))
    prompt_rect = prompt_surf.get_rect(center=(WINDOW_WIDTH // 2, box_rect.y + 40))
    screen.blit(prompt_surf, prompt_rect)
    
    # Vẽ nút Yes và No
    btn_width, btn_height = 100, 40
    yes_rect = pygame.Rect(box_rect.x + 30, box_rect.y + box_height - 60, btn_width, btn_height)
    no_rect = pygame.Rect(box_rect.x + box_width - 130, box_rect.y + box_height - 60, btn_width, btn_height)
    
    pygame.draw.rect(screen, (150, 150, 150), yes_rect)
    pygame.draw.rect(screen, (150, 150, 150), no_rect)
    yes_surf = font.render("Yes", True, (0, 0, 0))
    no_surf = font.render("No", True, (0, 0, 0))
    screen.blit(yes_surf, yes_surf.get_rect(center=yes_rect.center))
    screen.blit(no_surf, no_surf.get_rect(center=no_rect.center))
    
    return yes_rect, no_rect

def highlight_king_in_check():
    if board.is_check():
        king_sq = board.king(board.turn)
        if king_sq is not None:
            col = chess.square_file(king_sq)
            row = 7 - chess.square_rank(king_sq)
            pygame.draw.rect(screen, RED, (col * TILE_SIZE, TOP_MARGIN + row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 3)

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
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(150)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            img_rect = win_img.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 30))
            screen.blit(win_img, img_rect)
            msg = "Press any key to play again"
            msg_surf = font.render(msg, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + img_rect.height // 2))
            screen.blit(msg_surf, msg_rect)
            pygame.display.flip()

def select_or_move_piece(row, col):
    global selected_square
    clicked_sq_name = square_name_from_pos(row, col)
    clicked_piece = board.piece_at(chess.parse_square(clicked_sq_name))

    if selected_square is None:
        if clicked_piece and ((board.turn and clicked_piece.color == chess.WHITE) or (not board.turn and clicked_piece.color == chess.BLACK)):
            selected_square = (row, col)
        else:
            print("Không có quân cờ ở ô này!")
    else:
        move_uci = square_name_from_pos(*selected_square) + clicked_sq_name
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                is_capture = board.is_capture(move)
                board.push(move)

                if board.is_check():
                    play_sound_checkmate()
                elif is_capture:
                    play_sound_capture()
                else:
                    play_sound_move()

                selected_square = None
            else:
                print("Nước đi không hợp lệ!")
                if clicked_piece and ((board.turn and clicked_piece.color == chess.WHITE) or (not board.turn and clicked_piece.color == chess.BLACK)):
                    selected_square = (row, col)
        except chess.InvalidMoveError:
            print("Invalid move UCI:", move_uci)


#  MAIN LOOP (STATE MACHINE) 
state = "start_menu"  # Các trạng thái: "start_menu" và "game"
running = True

# Biến xác nhận (confirm): None hoặc action cần xác nhận ("exit", "newgame", "home")
confirm_action = None

while running:
    #  START MENU 
    if state == "start_menu":
        if not menu_music_playing:
            play_menu_music()
            menu_music_playing = True

        draw_start_menu()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                if one_player_rect.collidepoint(pos):
                    print("Chế độ 1 người chơi chưa được xử lý.")
                elif two_player_rect.collidepoint(pos):
                    pygame.mixer.music.stop()
                    menu_music_playing = False
                    board.reset()
                    selected_square = None
                    undone_moves = []
                    state = "game"
        pygame.display.flip()
    
    #  GAME LOOP 
    elif state == "game":
        pause_menu_active = False
        game_over = False
        while state == "game" and running:
            # Nếu chưa kích hoạt hộp xác nhận thì xử lý bình thường
            if confirm_action is None:
                if board.is_game_over() and not game_over:
                    game_over = True
                    outcome = board.outcome()
                    if outcome is not None:
                        if outcome.winner is True:
                            play_sound_finish()
                        elif outcome.winner is False:
                            play_sound_finish()    
                if game_over:
                    draw_board(screen, None, TILE_SIZE, WHITE, BLACK, HIGHLIGHT, offset=(0, TOP_MARGIN))
                    draw_pieces(screen, board, pieces_img, offset=(0, TOP_MARGIN))
                    draw_victory_overlay()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            state = ""
                        elif event.type == pygame.KEYDOWN:
                            board.reset()
                            undone_moves.clear()
                            selected_square = None
                            game_over = False
                    continue

                icon_rects = draw_top_bar(screen)
                draw_board(screen, selected_square, TILE_SIZE, WHITE, BLACK, HIGHLIGHT, offset=(0, TOP_MARGIN))
                highlight_possible_moves(screen, board, selected_square, HIGHLIGHT, offset=(0, TOP_MARGIN))
                draw_pieces(screen, board, pieces_img, offset=(0, TOP_MARGIN))
                highlight_king_in_check()

                if pause_menu_active:
                    pause_buttons = draw_pause_menu(screen)
                
                pygame.display.flip()

            # Nếu có hộp xác nhận đang hiển thị, vẽ nó lên (trên cùng)
            else:
                yes_rect, no_rect = draw_confirmation_dialog(confirm_action)
                pygame.display.flip()
            
            # Xử lý sự kiện
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    state = ""
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = pygame.mouse.get_pos()
                    
                    # Nếu đang hiển thị hộp xác nhận
                    if confirm_action is not None:
                        # Kiểm tra click vào nút Yes/No trong hộp xác nhận
                        if yes_rect.collidepoint(pos):
                            if confirm_action == "exit":
                                running = False
                                state = ""
                            elif confirm_action == "newgame":
                                board.reset()
                                undone_moves.clear()
                                selected_square = None
                                pause_menu_active = False  # Tắt pause menu khi bắt đầu ván mới
                                confirm_action = None
                            elif confirm_action == "home":
                                state = "start_menu"
                                pause_menu_active = False
                                confirm_action = None
                        elif no_rect.collidepoint(pos):
                            confirm_action = None
                        continue

                    # Nếu chưa kích hoạt hộp xác nhận
                    if pause_menu_active:
                        for key, rect in pause_buttons.items():
                            if rect.collidepoint(pos):
                                if key == "resume":
                                    pause_menu_active = False
                                elif key == "newgame":
                                    confirm_action = "newgame"
                                elif key == "home":
                                    confirm_action = "home"
                                elif key == "exit":
                                    confirm_action = "exit"
                        continue

                    if pos[1] < TOP_MARGIN:
                        if icon_rects[0].collidepoint(pos):
                            pause_menu_active = not pause_menu_active
                        elif icon_rects[1].collidepoint(pos) and board.move_stack:
                            undone_moves.append(board.pop())
                        elif icon_rects[2].collidepoint(pos) and undone_moves:
                            board.push(undone_moves.pop())
                    elif pos[1] > TOP_MARGIN and not pause_menu_active:
                        row, col = get_square_under_mouse(pos, offset=(0, TOP_MARGIN))
                        select_or_move_piece(row, col)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pause_menu_active = not pause_menu_active
                    elif event.key == pygame.K_z and board.move_stack:
                        undone_moves.append(board.pop())
                    elif event.key == pygame.K_x and undone_moves:
                        board.push(undone_moves.pop())

pygame.quit()
sys.exit()
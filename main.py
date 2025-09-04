# main.py (Fixed: Advance mode, safe AI moves, highlights, last-move, undo/redo, improved voice)
import pygame
import chess
import engine
import os
import time
import threading
from queue import Queue
import numpy as np
import speech_recognition as sr
import random
import re
from typing import Optional, List

# --- ChessAI helper class (levels + adaptive Advance mode) ---
class ChessAI:
    def __init__(self):
        # Note: include both "Advance" (internal) and "Advanced" (menu) synonyms will be normalized
        self.levels = {
            "Easy": {"depth": 1, "delay": (1, 5)},
            "Normal": {"depth": 2, "delay": (10, 20)},
            "Hard": {"depth": 3, "delay": (30, 60)},
            "Moderate": {"depth": 3, "delay": (30, 60)},  # alias to Hard
            "Advance": {"adaptive": True},                 # internal key for adaptive mode
        }
        self.current_level = "Easy"
        self.ai_win_ratio = None  # 0..100 or None

    def normalize_level(self, level: str) -> str:
        """Accept menu labels like 'Advanced' and return internal key (e.g., 'Advance')."""
        if not isinstance(level, str):
            return "Easy"
        lvl = level.strip()
        # Accept both forms: 'Advance' or 'Advanced'
        if lvl.lower().startswith("adv"):
            return "Advance"
        # normalize other common names
        if lvl.lower() == "easy":
            return "Easy"
        if lvl.lower() == "normal":
            return "Normal"
        if lvl.lower() in ("hard", "moderate"):
            return "Hard" if lvl.lower() == "hard" else "Moderate"
        # fallback
        return lvl

    def set_level(self, level: str):
        norm = self.normalize_level(level)
        if norm in self.levels:
            # if user selected "Moderate", keep that; if it maps to Hard we already set levels
            self.current_level = norm
            print(f"[ChessAI] Level set to '{self.current_level}'")
        else:
            print(f"[ChessAI] Invalid level '{level}', defaulting to Easy.")
            self.current_level = "Easy"

    def get_ai_depth_and_delay(self, board: chess.Board, win_ratio_pct: float | None = None, in_check: bool = False):
        """Return (depth, delay_tuple). Depth mapping:
           Easy=1, Normal=2, Hard=3, Advance adaptive:
              - if in_check -> depth 3
              - if win_ratio > 45% -> depth 1
              - if 28..45 -> depth 2
              - if <28 -> depth 3
        """
        if self.current_level != "Advance":
            spec = self.levels.get(self.current_level, self.levels["Normal"])
            # enforce mapping requested by user: Easy=1, Normal=2, Hard=3
            depth = spec.get("depth", 2)
            return depth, spec.get("delay", (1, 5))

        # Advance adaptive logic (win_ratio_pct expected 0..100)
        if in_check:
            return 3, (90, 120)
        if win_ratio_pct is None:
            return 2, (30, 60)
        # now use thresholds in percents
        if win_ratio_pct > 45.0:
            return 1, (1, 5)
        elif 28.0 <= win_ratio_pct <= 45.0:
            return 2, (30, 60)
        else:
            return 3, (90, 120)


# --- Constants ---
BOARD_WIDTH, BOARD_HEIGHT = 600, 600
PANEL_WIDTH = 300
WIDTH, HEIGHT = BOARD_WIDTH + PANEL_WIDTH, BOARD_HEIGHT
DIMENSION = 8
SQ_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 30
IMAGES = {}

def load_images():
    pieces = ['wP', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bP', 'bR', 'bN', 'bB', 'bK', 'bQ']
    for piece in pieces:
        path = os.path.join('images', piece + '.png')
        IMAGES[piece] = pygame.transform.scale(pygame.image.load(path), (SQ_SIZE, SQ_SIZE))

def check_for_mate_in_one(board: chess.Board) -> bool:
    for move in board.legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return True
    return False

# --- Pygame Setup & UI Elements ---
pygame.init()
pygame.display.set_caption("Chess AI")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Fonts (robust fallback)
try:
    FONT = pygame.font.SysFont("Helvetica", 28, True, False)
    SMALL_FONT = pygame.font.SysFont("Helvetica", 20, True, False)
    COORD_FONT = pygame.font.SysFont("Helvetica", 16, True, False)
except Exception:
    FONT = pygame.font.Font(None, 28)
    SMALL_FONT = pygame.font.Font(None, 20)
    COORD_FONT = pygame.font.Font(None, 16)

MAIN_MENU_BUTTONS = {
    "VS AI": pygame.Rect(WIDTH // 2 - 100, 200, 200, 50),
    "Pass and Play": pygame.Rect(WIDTH // 2 - 100, 270, 200, 50),
}
DIFFICULTY_BUTTONS = {
    "Easy": pygame.Rect(WIDTH // 2 - 100, 200, 200, 50),
    "Normal": pygame.Rect(WIDTH // 2 - 100, 270, 200, 50),
    "Moderate": pygame.Rect(WIDTH // 2 - 100, 340, 200, 50),
    "Advanced": pygame.Rect(WIDTH // 2 - 100, 410, 200, 50),
    "Back": pygame.Rect(WIDTH // 2 - 100, 480, 200, 50),
}
NEW_GAME_BUTTON = pygame.Rect(BOARD_WIDTH + 50, HEIGHT - 70, PANEL_WIDTH - 100, 50)
UNDO_BUTTON = pygame.Rect(BOARD_WIDTH + 20, 450, 120, 40)
REDO_BUTTON = pygame.Rect(BOARD_WIDTH + 160, 450, 120, 40)
VOICE_BUTTON = pygame.Rect(BOARD_WIDTH + 20, 500, 260, 40)

# --- Voice recognition helpers (kept from your last version) ---
LETTER_MAP = {
    "a": "a", "alpha": "a", "apple": "a", "ay": "a",
    "b": "b", "bravo": "b", "bombay": "b", "bhai": "b", "bee": "b",
    "c": "c", "charlie": "c", "chennai": "c", "see": "c",
    "d": "d", "delta": "d", "delhi": "d", "diya": "d", "dee": "d", "dog": "d",
    "e": "e", "echo": "e", "elephant": "e",
    "f": "f", "foxtrot": "f", "fish": "f", "eff": "f",
    "g": "g", "golf": "g", "goa": "g", "ganesh": "g", "ganga": "g", "jivan": "g", "jeevan": "g", "jee": "g",
    "h": "h", "hotel": "h", "hari": "h", "house": "h", "aitch": "h",
}
NUMBER_MAP = {
    "one": "1", "1": "1", "first": "1",
    "two": "2", "to": "2", "too": "2", "2": "2",
    "three": "3", "tree": "3", "3": "3",
    "four": "4", "for": "4", "4": "4",
    "five": "5", "5": "5",
    "six": "6", "sex": "6", "sick": "6", "6": "6",
    "seven": "7", "7": "7",
    "eight": "8", "ate": "8", "8": "8"
}
STOPWORDS = {"move", "from", "to", "the", "please", "pawn", "knight", "bishop", "rook", "queen", "king", "sir", "ji"}

def parse_voice_command(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    t = re.sub(r'[^a-z0-9\s\-]', ' ', text.lower())
    t = re.sub(r'\s+', ' ', t).strip()
    m = re.search(r'([a-h][1-8])\s*[-]?\s*([a-h][1-8])', t)
    if m:
        return (m.group(1) + m.group(2)).lower()

    tokens = t.split()
    parsed = []
    for tok in tokens:
        if tok in STOPWORDS:
            continue
        if re.fullmatch(r'[a-h][1-8]', tok):
            parsed.append(tok[0]); parsed.append(tok[1]); continue
        if tok in LETTER_MAP:
            parsed.append(LETTER_MAP[tok]); continue
        if tok in NUMBER_MAP:
            parsed.append(NUMBER_MAP[tok]); continue
        compact = re.findall(r'([a-h]|[1-8])', tok)
        for c in compact: parsed.append(c)

    if len(parsed) >= 4:
        # build first two pairs (letter+digit)
        sequence = []
        i = 0
        while i < len(parsed) and len(sequence) < 4:
            val = parsed[i]
            if val in 'abcdefgh' and i + 1 < len(parsed) and parsed[i + 1] in '12345678':
                sequence.append(val); sequence.append(parsed[i+1]); i += 2
            else:
                i += 1
        if len(sequence) == 4:
            return (sequence[0] + sequence[1] + sequence[2] + sequence[3]).lower()

    pairs = re.findall(r'[a-h][1-8]', t)
    if len(pairs) >= 2:
        return (pairs[0] + pairs[1]).lower()
    return None

def listen_for_voice_command(voice_queue: Queue, recognizer: sr.Recognizer):
    with sr.Microphone() as source:
        print("Listening for a move...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, phrase_time_limit=4)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            move_uci = parse_voice_command(text)
            if move_uci:
                print(f"Parsed move: {move_uci}")
                voice_queue.put(move_uci)
            else:
                cleaned = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
                fallback = parse_voice_command(cleaned)
                if fallback:
                    print(f"Fallback parsed: {fallback}")
                    voice_queue.put(fallback)
                else:
                    print("Could not parse a valid chess move from speech.")
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

# --- New helper: try applying a UCI move robustly (promotion fallback) ---
def try_apply_uci_move(board: chess.Board, uci: str) -> Optional[chess.Move]:
    """
    Try to convert a uci string to a legal move on the given board.
    Handles 4-char UCI, 5-char with promotion, and tries promotion alternatives if necessary.
    Returns the chess.Move if valid (but does NOT push it), otherwise None.
    """
    if not uci or not isinstance(uci, str):
        return None
    uci = uci.strip().lower()
    # direct attempt
    try:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
    except Exception:
        pass

    # If uci is longer (like ended with q,r,b,n) or malformed, try to salvage:
    # Try base 4 char and add promotions if pawn promotion possible
    base = uci[:4]
    if re.fullmatch(r'[a-h][1-8][a-h][1-8]', base):
        # attempt promotions
        promotions = ['q', 'r', 'b', 'n']
        for p in promotions:
            candidate = base + p
            try:
                move = chess.Move.from_uci(candidate)
                if move in board.legal_moves:
                    return move
            except Exception:
                continue
        # Also, if engine returned something like 'd2b1q' but base d2b1 is not a promotion (illegal),
        # then there's no salvage; return None.
    # Finally, try to find a legal move that matches from & to squares ignoring trailing garbage
    matches = re.findall(r'([a-h][1-8])', uci)
    if len(matches) >= 2:
        from_sq = matches[0]
        to_sq = matches[1]
        try:
            candidate = chess.Move.from_uci(from_sq + to_sq)
            if candidate in board.legal_moves:
                return candidate
        except Exception:
            pass
        # try promotions for candidate base
        for p in ['q', 'r', 'b', 'n']:
            try:
                candidate = chess.Move.from_uci(from_sq + to_sq + p)
                if candidate in board.legal_moves:
                    return candidate
            except Exception:
                pass
    return None

# --- Drawing helpers: highlights & last move ---
def square_to_screen_pos(square: int):
    """Convert python-chess square (0..63) to top-left pixel (x,y)."""
    file = chess.square_file(square)  # 0..7
    rank = chess.square_rank(square)  # 0..7 (0 = rank1)
    x = file * SQ_SIZE
    y = (7 - rank) * SQ_SIZE
    return x, y

def draw_highlights(screen, board: chess.Board, selected_square: Optional[int], last_move: Optional[chess.Move]):
    # highlight last move (if any) with translucent rectangle
    if last_move:
        for sq in (last_move.from_square, last_move.to_square):
            x, y = square_to_screen_pos(sq)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill((255, 215, 0, 80))  # gold translucent
            screen.blit(s, (x, y))

    # highlight legal destinations for selected square
    if selected_square is not None:
        # draw source square marker
        sx, sy = square_to_screen_pos(selected_square)
        ssrc = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        ssrc.fill((50, 150, 250, 80))
        screen.blit(ssrc, (sx, sy))

        # get legal moves from selected_square
        moves = [m for m in board.legal_moves if m.from_square == selected_square]
        for m in moves:
            tx, ty = square_to_screen_pos(m.to_square)
            # draw a small filled circle at center of target
            cx = tx + SQ_SIZE // 2
            cy = ty + SQ_SIZE // 2
            pygame.draw.circle(screen, (30, 200, 50), (cx, cy), SQ_SIZE // 8)
            # draw faint line from source center to target center
            pygame.draw.line(screen, (30,200,50), (sx + SQ_SIZE//2, sy + SQ_SIZE//2), (cx, cy), 2)

# --- Drawing board & pieces (uses highlights) ---
def draw_board_and_pieces(screen, board: chess.Board, selected_square: Optional[int], last_move: Optional[chess.Move]):
    colors = [pygame.Color("#EBECD0"), pygame.Color("#779556")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            pygame.draw.rect(screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            # pieces drawn later after highlights to allow overlays
    # draw last move & selected move highlights below pieces
    draw_highlights(screen, board, selected_square, last_move)
    # draw pieces
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board.piece_at(chess.square(c, 7 - r))
            if piece is not None:
                piece_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
                screen.blit(IMAGES[piece_key], pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
    # draw board coordinates (files and ranks)
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for i, f in enumerate(files):
        text = COORD_FONT.render(f, True, pygame.Color("white"))
        screen.blit(text, (i * SQ_SIZE + (SQ_SIZE - text.get_width()) // 2, BOARD_HEIGHT - text.get_height() + 2))
        text2 = COORD_FONT.render(f, True, pygame.Color("white"))
        screen.blit(text2, (i * SQ_SIZE + (SQ_SIZE - text2.get_width()) // 2, 2))
    for rank in range(8, 0, -1):
        i = 8 - rank
        text = COORD_FONT.render(str(rank), True, pygame.Color("white"))
        screen.blit(text, (2, i * SQ_SIZE + (SQ_SIZE - text.get_height()) // 2))
        screen.blit(text, (BOARD_WIDTH - text.get_width() - 2, i * SQ_SIZE + (SQ_SIZE - text.get_height()) // 2))

# --- Info panel (unchanged except receives chess_ai) ---
def draw_info_panel(screen, font, board: chess.Board, white_time, black_time, game_over, ai_is_thinking,
                    turn_start_time, current_depth, mate_in_one, voice_on, game_mode, chess_ai: ChessAI, ai_difficulty_label):
    panel_rect = pygame.Rect(BOARD_WIDTH, 0, PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, pygame.Color("#262522"), panel_rect)
    y_offset = 20
    level_display = f"Level: {chess_ai.current_level}"
    level_text = SMALL_FONT.render(level_display, True, pygame.Color("white"))
    screen.blit(level_text, (BOARD_WIDTH + 20, y_offset))
    y_offset += 40
    if game_mode == "Pass and Play":
        status_text = "White's Turn" if board.turn else "Black's Turn"
    else:
        status_text = "Your Turn (White)" if board.turn else f"AI's Turn (Depth: {current_depth})"
        if ai_is_thinking:
            status_text = f"AI is Thinking... (Depth: {current_depth})"
    if game_over:
        result = board.result()
        if result == "1-0":
            status_text = "Game Over - White Wins!"
        elif result == "0-1":
            status_text = "Game Over - Black Wins!"
        else:
            status_text = "Game Over - Draw!"
    text_obj = FONT.render(status_text, True, pygame.Color('white'))
    screen.blit(text_obj, (BOARD_WIDTH + 20, y_offset))
    y_offset += 60
    elapsed_time = 0 if game_over else time.time() - turn_start_time
    display_white_time = white_time + (elapsed_time if board.turn == chess.WHITE else 0)
    white_time_str = time.strftime('%M:%S', time.gmtime(display_white_time))
    white_text = SMALL_FONT.render(f"White Time: {white_time_str}", True, pygame.Color('white'))
    screen.blit(white_text, (BOARD_WIDTH + 20, y_offset))
    y_offset += 26
    display_black_time = black_time + (elapsed_time if board.turn == chess.BLACK else 0)
    black_time_str = time.strftime('%M:%S', time.gmtime(display_black_time))
    black_text = SMALL_FONT.render(f"Black Time: {black_time_str}", True, pygame.Color('white'))
    screen.blit(black_text, (BOARD_WIDTH + 20, y_offset))
    y_offset += 34
    if (game_mode == "VS AI"):
        try:
            raw_prediction = engine.MODEL.predict(np.expand_dims(engine.board_to_input_array(board), axis=0), verbose=0)[0][0]
            white_win_chance = 50 * (raw_prediction + 1)
            white_win_chance = max(0.0, min(100.0, float(white_win_chance)))
            black_win_chance = 100.0 - white_win_chance
            win_chance_text = SMALL_FONT.render(f"White Win: {white_win_chance:.1f}%", True, pygame.Color('white'))
            screen.blit(win_chance_text, (BOARD_WIDTH + 20, y_offset)); y_offset += 22
            win_chance_text_black = SMALL_FONT.render(f"Black Win: {black_win_chance:.1f}%", True, pygame.Color('white'))
            screen.blit(win_chance_text_black, (BOARD_WIDTH + 20, y_offset)); y_offset += 28
            chess_ai.ai_win_ratio = white_win_chance
        except Exception:
            info_text = SMALL_FONT.render("Win prob: (model unavailable)", True, pygame.Color('gray'))
            screen.blit(info_text, (BOARD_WIDTH + 20, y_offset)); y_offset += 28
            chess_ai.ai_win_ratio = None
    if mate_in_one and not game_over:
        mate_text = SMALL_FONT.render("Checkmate in 1!", True, pygame.Color('yellow'))
        screen.blit(mate_text, (BOARD_WIDTH + 20, y_offset)); y_offset += 28
    pygame.draw.rect(screen, pygame.Color("dimgray"), UNDO_BUTTON)
    screen.blit(SMALL_FONT.render("Undo", True, pygame.Color("white")), (UNDO_BUTTON.x + 30, UNDO_BUTTON.y + 8))
    pygame.draw.rect(screen, pygame.Color("dimgray"), REDO_BUTTON)
    screen.blit(SMALL_FONT.render("Redo", True, pygame.Color("white")), (REDO_BUTTON.x + 30, REDO_BUTTON.y + 8))
    voice_color = pygame.Color("darkgreen") if voice_on else pygame.Color("darkred")
    pygame.draw.rect(screen, voice_color, VOICE_BUTTON)
    screen.blit(SMALL_FONT.render(f"Voice: {'ON' if voice_on else 'OFF'}", True, pygame.Color("white")),
                (VOICE_BUTTON.x + 12, VOICE_BUTTON.y + 8))
    if game_over:
        pygame.draw.rect(screen, pygame.Color("darkblue"), NEW_GAME_BUTTON)
        btn_text = FONT.render("Main Menu", True, pygame.Color("white"))
        screen.blit(btn_text, (NEW_GAME_BUTTON.x + 20, NEW_GAME_BUTTON.y + 10))

def draw_menu(screen, title, buttons):
    screen.fill(pygame.Color("#262522"))
    title_font = pygame.font.SysFont("Helvetica", 50, True, False)
    title_text = title_font.render(title, True, pygame.Color("white"))
    screen.blit(title_text, ((WIDTH - title_text.get_width()) // 2, 100))
    for name, rect in buttons.items():
        pygame.draw.rect(screen, pygame.Color("darkgreen"), rect)
        btn_text = FONT.render(name, True, pygame.Color("white"))
        screen.blit(btn_text, (rect.x + (rect.width - btn_text.get_width()) // 2, rect.y + 10))

def difficulty_to_depth(label: str) -> int:
    # mapping (kept for compatibility; Advance handled by ChessAI.get_ai_depth_and_delay)
    return {"Easy":1,"Normal":2,"Moderate":3,"Hard":3,"Advance":2,"Advanced":2}.get(label,2)

def handle_post_move_updates(board, side_was_white, times, turn_start_time_ref):
    white_time, black_time = times
    time_spent = time.time() - turn_start_time_ref[0]
    if side_was_white:
        white_time += time_spent
    else:
        black_time += time_spent
    mate_in_one = check_for_mate_in_one(board) if not board.is_game_over() else False
    turn_start_time_ref[0] = time.time()
    return white_time, black_time, mate_in_one, board.is_game_over()

# --- Main loop ---
def main():
    load_images()
    chess_ai = ChessAI()

    game_state, game_mode, ai_difficulty = 'MAIN_MENU', None, None
    board, redo_stack = chess.Board(), []
    white_time, black_time = 0.0, 0.0
    turn_start_time_ref = [time.time()]
    selected_square, game_over, mate_in_one = None, False, False
    last_move: Optional[chess.Move] = None
    ai_move_queue, voice_move_queue = Queue(), Queue()
    ai_is_thinking, voice_command_on = False, False
    current_search_depth = 0
    recognizer = sr.Recognizer()

    running = True
    while running:
        if game_state == 'MAIN_MENU':
            draw_menu(screen, "Chess AI", MAIN_MENU_BUTTONS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    location = pygame.mouse.get_pos()
                    if MAIN_MENU_BUTTONS["VS AI"].collidepoint(location):
                        game_state = 'DIFFICULTY_MENU'
                    elif MAIN_MENU_BUTTONS["Pass and Play"].collidepoint(location):
                        game_mode, game_state = 'Pass and Play', 'PLAYING'
                        board, redo_stack = chess.Board(), []
                        white_time, black_time = 0.0, 0.0
                        turn_start_time_ref[0] = time.time()
                        selected_square, game_over, mate_in_one = None, False, False
                        chess_ai.set_level("Easy")

        elif game_state == 'DIFFICULTY_MENU':
            draw_menu(screen, "Select AI Difficulty", DIFFICULTY_BUTTONS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    location = pygame.mouse.get_pos()
                    if DIFFICULTY_BUTTONS["Back"].collidepoint(location):
                        game_state = 'MAIN_MENU'
                    else:
                        for name, rect in DIFFICULTY_BUTTONS.items():
                            if rect.collidepoint(location) and name != "Back":
                                # normalize name through ChessAI.set_level
                                chess_ai.set_level(name)
                                ai_difficulty = name
                                game_mode, game_state = 'VS AI', 'PLAYING'
                                board, redo_stack = chess.Board(), []
                                white_time, black_time = 0.0, 0.0
                                turn_start_time_ref[0] = time.time()
                                selected_square, game_over, mate_in_one = None, False, False

        elif game_state == 'PLAYING':
            is_human_turn = (game_mode == 'Pass and Play') or (game_mode == 'VS AI' and board.turn == chess.WHITE)

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    location = pygame.mouse.get_pos()
                    if game_over and NEW_GAME_BUTTON.collidepoint(location):
                        game_state, game_over, selected_square = 'MAIN_MENU', False, None
                        continue
                    if not game_over and UNDO_BUTTON.collidepoint(location) and board.move_stack:
                        last = board.pop()
                        redo_stack.append(last)
                        turn_start_time_ref[0] = time.time()
                        mate_in_one = check_for_mate_in_one(board) if not board.is_game_over() else False
                        game_over = board.is_game_over()
                        last_move = board.peek() if board.move_stack else None
                    elif not game_over and REDO_BUTTON.collidepoint(location) and redo_stack:
                        move = redo_stack.pop()
                        board.push(move)
                        turn_start_time_ref[0] = time.time()
                        mate_in_one = check_for_mate_in_one(board) if not board.is_game_over() else False
                        game_over = board.is_game_over()
                        last_move = move
                    elif not game_over and VOICE_BUTTON.collidepoint(location):
                        voice_command_on = not voice_command_on
                        if voice_command_on:
                            threading.Thread(target=listen_for_voice_command, args=(voice_move_queue, recognizer), daemon=True).start()
                    elif is_human_turn and not game_over and location[0] <= BOARD_WIDTH:
                        col, row = location[0] // SQ_SIZE, location[1] // SQ_SIZE
                        clicked_square = chess.square(col, 7 - row)
                        if selected_square is None:
                            piece = board.piece_at(clicked_square)
                            if piece is not None and piece.color == board.turn:
                                selected_square = clicked_square
                        else:
                            move = chess.Move(selected_square, clicked_square)
                            piece_at_sel = board.piece_at(selected_square)
                            if piece_at_sel and piece_at_sel.piece_type == chess.PAWN and chess.square_rank(clicked_square) in (0, 7):
                                move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)
                            if move in board.legal_moves:
                                side_was_white = board.turn == chess.WHITE
                                board.push(move)
                                last_move = move
                                redo_stack.clear()
                                white_time, black_time, mate_in_one, game_over = handle_post_move_updates(board, side_was_white, (white_time, black_time), turn_start_time_ref)
                            else:
                                print("Illegal human move attempted:", move)
                            selected_square = None

            # Voice move (human)
            if not voice_move_queue.empty() and is_human_turn and not game_over:
                move_uci = voice_move_queue.get()
                applied = None
                try:
                    applied = try_apply_uci_move(board, move_uci)
                    if applied:
                        side_was_white = board.turn == chess.WHITE
                        board.push(applied)
                        last_move = applied
                        redo_stack.clear()
                        white_time, black_time, mate_in_one, game_over = handle_post_move_updates(board, side_was_white, (white_time, black_time), turn_start_time_ref)
                    else:
                        print(f"Invalid voice move: {move_uci}")
                except Exception as ex:
                    print(f"Could not parse/apply voice move '{move_uci}': {ex}")
                if voice_command_on:
                    threading.Thread(target=listen_for_voice_command, args=(voice_move_queue, recognizer), daemon=True).start()

            # AI move
            if game_mode == 'VS AI' and not is_human_turn and not game_over and not ai_is_thinking:
                win_ratio_pct = chess_ai.ai_win_ratio
                in_check = board.is_check()
                depth, _ = chess_ai.get_ai_depth_and_delay(board, win_ratio_pct, in_check)
                current_search_depth = depth
                ai_is_thinking = True
                threading.Thread(target=engine.find_best_move, args=(board.copy(), current_search_depth, ai_move_queue), daemon=True).start()

            if not ai_move_queue.empty():
                ai_move = ai_move_queue.get()
                ai_is_thinking = False
                if ai_move is not None and not game_over:
                    # Safety check: ensure move is legal on current board
                    if ai_move in board.legal_moves:
                        side_was_white = board.turn == chess.WHITE
                        board.push(ai_move)
                        last_move = ai_move
                        white_time, black_time, mate_in_one, game_over = handle_post_move_updates(board, side_was_white, (white_time, black_time), turn_start_time_ref)
                    else:
                        # If engine returned a UCI string instead of Move object, try to apply robustly
                        if isinstance(ai_move, str):
                            candidate = try_apply_uci_move(board, ai_move)
                            if candidate:
                                side_was_white = board.turn == chess.WHITE
                                board.push(candidate)
                                last_move = candidate
                                white_time, black_time, mate_in_one, game_over = handle_post_move_updates(board, side_was_white, (white_time, black_time), turn_start_time_ref)
                                continue
                        print("Engine suggested illegal move:", ai_move)
                        # fallback to a safe legal move (choose best available or random)
                        legal = list(board.legal_moves)
                        if legal:
                            fallback = random.choice(legal)
                            print("Falling back to legal move:", fallback)
                            side_was_white = board.turn == chess.WHITE
                            board.push(fallback)
                            last_move = fallback
                            white_time, black_time, mate_in_one, game_over = handle_post_move_updates(board, side_was_white, (white_time, black_time), turn_start_time_ref)
                        else:
                            print("No legal moves available for fallback.")

            if not game_over and board.is_game_over():
                game_over = True

            # Draw board with highlights
            draw_board_and_pieces(screen, board, selected_square, last_move)
            draw_info_panel(screen, FONT, board, white_time, black_time, game_over, ai_is_thinking, turn_start_time_ref[0], current_search_depth, mate_in_one, voice_command_on, game_mode, chess_ai, ai_difficulty)

        pygame.display.flip()
        clock.tick(MAX_FPS)

    pygame.quit()

if __name__ == "__main__":
    main()

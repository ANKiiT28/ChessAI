# engine.py (Final Version with Performance Fix)
import chess
import numpy as np
import tensorflow as tf

try:
    MODEL = tf.keras.models.load_model('chess_model_2018_02_fast.h5')
    print("Successfully loaded the trained model: chess_model_2018_02_fast.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL = None

def board_to_input_array(board):
    """Converts a chess.Board object into a numerical representation (8x8x12)."""
    input_array = np.zeros((8, 8, 12), dtype=np.int8)
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = chess.square_rank(square), chess.square_file(square)
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            input_array[rank, file, channel] = 1
    return input_array

if MODEL is not None:
    print("Warming up the model...")
    dummy_board = chess.Board()
    dummy_input = np.expand_dims(board_to_input_array(dummy_board), axis=0)
    MODEL.predict(dummy_input, verbose=0)
    print("Model is ready.")

def evaluate_board(board):
    """
    Evaluates the board using our trained neural network.
    Returns a score from the perspective of the current player.
    """
    if MODEL is None: return 0 
        
    if board.is_checkmate():
        return -9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    input_array = np.expand_dims(board_to_input_array(board), axis=0)
    prediction = MODEL.predict(input_array, verbose=0)[0][0]
    
    score = prediction if board.turn == chess.WHITE else -prediction
    return score

def minimax_alpha_beta(board, depth, alpha, beta, is_maximizing_player):
    """Minimax algorithm with Alpha-Beta Pruning."""
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if is_maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax_alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax_alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

# --- BUG FIX WAS HERE: This function now correctly uses Alpha-Beta Pruning ---
def find_best_move(board, depth, return_queue):
    """This function finds the best move for the AI."""
    best_move = None
    is_maximizing = board.turn == chess.WHITE
    alpha = -float('inf')
    beta = float('inf')

    if is_maximizing:
        best_value = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax_alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            if board_value > best_value:
                best_value = board_value
                best_move = move
            alpha = max(alpha, best_value)
    else: # Minimizing player
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax_alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            if board_value < best_value:
                best_value = board_value
                best_move = move
            beta = min(beta, best_value)
            
    return_queue.put(best_move)
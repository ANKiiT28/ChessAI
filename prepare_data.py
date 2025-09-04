# prepare_data.py
import chess
import chess.pgn
import numpy as np

def board_to_input_array(board):
    """
    Converts a chess.Board object into a numerical representation (8x8x12)
    for the neural network.
    """
    # 8x8 board, 12 channels (wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK)
    input_array = np.zeros((8, 8, 12), dtype=np.int8)
    
    piece_to_channel = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4,
        (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.KING, chess.BLACK): 11,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_to_channel[(piece.piece_type, piece.color)]
            input_array[rank, file, channel] = 1
            
    return input_array

def process_pgn_file(pgn_filename, num_samples_limit=10000):
    """
    Reads a PGN file and processes its games to create training data.
    """
    X = []  # Board positions (features)
    y = []  # Game results (labels)
    
    result_map = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}
    
    with open(pgn_filename) as pgn:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break # End of file
            
            result_str = game.headers.get('Result')
            if result_str not in result_map:
                continue # Skip games with unknown results

            result = result_map[result_str]
            board = game.board()
            
            # Process each move in the game
            for move in game.mainline_moves():
                board.push(move)
                # Add the board state and the final game result to our dataset
                X.append(board_to_input_array(board))
                y.append(result)
                
                if len(X) >= num_samples_limit:
                    print(f"Reached sample limit of {num_samples_limit}.")
                    return np.array(X), np.array(y)

            game_count += 1
            print(f"Processed game #{game_count}. Total samples: {len(X)}")

    return np.array(X), np.array(y)

if __name__ == '__main__':
    # IMPORTANT: Replace this with the actual name of your PGN file
    pgn_file = 'lichess_db_standard_rated_2025-08.pgn'
    
    print(f"Starting data preparation from '{pgn_file}'...")
    try:
        X_train, y_train = process_pgn_file(pgn_file)
        
        print("\nData preparation complete.")
        print(f"Shape of X_train (board positions): {X_train.shape}")
        print(f"Shape of y_train (game results): {y_train.shape}")
        
        # Optionally, save the processed data to a file for later use
        np.savez_compressed('chess_training_data.npz', X=X_train, y=y_train)
        print("Saved processed data to 'chess_training_data.npz'")
        
    except FileNotFoundError:
        print(f"\nERROR: PGN file not found: '{pgn_file}'")
        print("Please make sure you have downloaded and decompressed the file")
        print("and that the filename matches the one in this script.")
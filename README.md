This repository hosts a Chess AI project, developed primarily using Jupyter Notebook and Python. The project aims to create an artificial intelligence capable of playing chess.
Features
Chess Engine: Contains the core logic for the AI's chess moves and game management.
Pre-trained Model: Includes a pre-trained neural network model (chess_model_2018_02_fast.h5) for move prediction or evaluation.
Data Preparation: Scripts to prepare and process data for training the chess AI.
Interactive Development: The main development was conducted within a Jupyter Notebook, allowing for an interactive and exploratory approach to building the AI.
Files and Directories
images/: Directory for any images used within the project (e.g., for visualization in the notebook).
*[1] Chess_AI_Project.ipynb: The main Jupyter Notebook containing the project's code, explanations, and analysis.
*[1] chess_model_2018_02_fast.h5: A pre-trained model file, likely a Keras or TensorFlow model, used by the AI to make decisions.
*[1] engine.py: Python script containing the game engine logic, handling board state, valid moves, and other chess rules.
*[1] main.py: The primary executable script to run the Chess AI.
*[1] prepare_data.py: Python script responsible for data loading, preprocessing, and feature engineering for the AI model.
*[1] requirements.txt: Lists the Python dependencies required to run the project.
[1]## Technologies Used
Jupyter Notebook: For interactive development and analysis.
*[1] Python: The primary programming language.
*[1] TensorFlow/Keras (likely): Given the .h5 model file, it's probable that TensorFlow or Keras was used for building and training the neural network.
Getting Started
Prerequisites
Ensure you have Python installed. You can install the required packages using pip:
code
Bash
pip install -r requirements.txt
Running the AI
To run the chess AI, execute the main.py script:
code
Bash
python main.py
For an interactive experience and to understand the development process, open the Jupyter Notebook:
code
Bash
jupyter notebook Chess_AI_Project.ipynb
Contribution
Currently, there are no specific guidelines for contributions.
License
(No license information found in the repository. Please add appropriate license information here.)
Contact
For any inquiries, please contact the repository owner, ANKiiT28.

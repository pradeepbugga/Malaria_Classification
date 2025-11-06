#analyze_history.py
#visualize training history from json file
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.visualize import plot_history

log_dir = '/home/pb929/Projects/Malaria_Classification/models/cnn_rgb_3l/logs'
history_path = os.path.join(log_dir, 'history.json')
plot_history(history_path, save_path=os.path.join(log_dir, 'history.png'))

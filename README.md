
# Traffic Light Management System

This project implements a reinforcement learning-based system to optimize traffic light management using SUMO (Simulation of Urban MObility). The system is designed to train a traffic model on a specific network and test its performance, providing insights into reducing waiting time and improving traffic flow.

## Features

- Train a PPO-based model to optimize traffic light schedules.
- Test the trained model on different networks.
- Automatically generate route files using `randomTrips.py`.
- Visualize traffic performance metrics such as waiting time.

## Installation

1. Install SUMO:
   - Download and install SUMO from [SUMO Official Site](https://sumo.dlr.de/docs/Downloads.php).
   - Set the `SUMO_HOME` environment variable to point to the SUMO installation directory.

2. Clone this repository:
   ```bash
   git clone https://github.com/YourGitHub/Dynamic-Traffic-light-management-system.git
   cd Dynamic-Traffic-light-management-system
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

### Training a Model

To train a traffic management model on a specific network:
```bash
python train.py --train true --net_file "maps/city1.net.xml" --steps 500
```

### Testing a Model

To test a pre-trained model on a specific network:
```bash
python train.py --train false --net_file "maps/city1.net.xml" --model_name "models/traffic_model_city3.net.pth" --epochs 1 --steps 500
```

### Generate Route Files

To generate route files for a network:
```bash
python <SUMO_HOME>/tools/randomTrips.py -n maps/city1.net.xml -r maps/city1.rou.xml
```

### Run SUMO GUI

To visualize the performance in SUMO GUI, ensure you specify `--train false` and `--net_file` with the desired network.

## File Structure

- `maps/`: Contains `.net.xml` files (network definitions) and `.rou.xml` files (route files).
- `models/`: Directory where trained models are saved.
- `train.py`: Main script for training and testing the traffic light management system.
- `randomTrips.py`: Script to generate random trips for simulation.
- `requirements.txt`: Python dependencies.

## Credits

This project draws data and inspiration from [Maunish-dave's GitHub Repository](https://github.com/Maunish-dave/Dynamic-Traffic-light-management-system/tree/main).


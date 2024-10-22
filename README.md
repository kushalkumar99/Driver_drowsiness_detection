# Driver Drowsiness Detection System

This project is designed to detect driver drowsiness in real-time to prevent accidents. The system uses computer vision techniques through OpenCV to monitor the driver's eye movements. If drowsiness is detected, the system triggers an alert. The project consists of a backend server written in Python (`app.py`) and a frontend user interface (`index.html`).

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Driver Drowsiness Detection System is built using OpenCV for real-time eye-tracking and Flask as the web framework for the backend. The frontend interface is served using an HTML file inside the `templates` folder. The system monitors the user's eyes and alerts them if drowsiness is detected.

## Technologies Used

- **Python**: Backend programming language.
- **Flask**: Web framework to handle backend services.
- **OpenCV**: Computer vision library for eye detection and tracking.
- **dlib**: For facial landmark detection.
- **HTML**: Frontend technologies for the user interface.
- **Tesseract**: Optical Character Recognition (optional).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/driver-drowsiness-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd driver-drowsiness-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python app.py
   ```

## Usage

1. Open your browser and navigate to `http://localhost:5000`.
2. The frontend page will load, and you can start the camera by clicking the **Start** button.
3. The system will continuously monitor the driver’s eyes. If drowsiness is detected, an alert sound will play.

## How It Works

1. **Face Detection**: Using OpenCV, the system detects the face and eyes of the driver in real-time.
2. **Eye Aspect Ratio (EAR)**: The eye region is extracted, and the EAR is calculated. If the EAR falls below a certain threshold for a consecutive number of frames, the system identifies drowsiness.
3. **Alert System**: Once drowsiness is detected, the system plays an alarm to alert the driver.

## Screenshots

_Add screenshots or GIFs demonstrating the functionality._

## Contributing

Feel free to fork this repository and make changes. Contributions are always welcome.

# Person Detection

A Python-based application utilizing deep learning for real-time person detection in images and videos. Built with Flask for the web interface and PyTorch for model inference, this project demonstrates the deployment of a trained YOLOv5 model (`survei2.pt`) for detecting individuals in various media formats.

## Features

* **Real-Time Detection**: Processes images and videos to identify and highlight persons.
* **Web Interface**: User-friendly interface built with Flask, allowing easy interaction and visualization.
* **Pre-trained Model**: Utilizes a custom-trained YOLOv5 model (`survei2.pt`) for accurate detections.

## Project Structure

```
Person-Detection/
├── app.py                 # Main Flask application
├── sample.py              # Sample script for testing
├── test.py                # Script for running tests
├── survei2.pt             # Pre-trained YOLOv5 model
├── static/                # Static files (CSS, JS, images)
└── templates/             # HTML templates for Flask
```



## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/nithishsivakumar1551/Person-Detection.git
   cd Person-Detection
   ```



2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```



*Note: Ensure that `requirements.txt` is present and contains all necessary dependencies.*

## Usage

1. **Run the Flask Application**

   ```bash
   python app.py
   ```



The application will start on `http://127.0.0.1:5000/`.

2. **Access the Web Interface**

   Open your web browser and navigate to `http://127.0.0.1:5000/` to use the person detection interface.

3. **Testing with Sample Scripts**

   * **Sample Detection**

     ```bash
     python sample.py
     ```

   * **Run Tests**

     ```bash
     python test.py
     ```

## Model Details

* **Model**: YOLOv5
* **Weights File**: `survei2.pt`
* **Framework**: PyTorch

*Note: Ensure that the `survei2.pt` file is present in the root directory. If not, place the trained model file accordingly.*

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

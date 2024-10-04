# Student prioritization dashboard
## Project overview
This project aims to develop a tool for educational advisors to prioritize students who need academic support. The dashboard visualizes students' current performance and the complexity of providing personalized help, based on various factors.

## Getting started
### Prerequisites

- Python 3.x
- Libraries listed in `requirements.txt`

### Clone the project repository

```bash
git clone https://github.com/hajar-hajji/school_help_dashboard.git
cd school_help_dashboard
```

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Running the dashboard

### Launch the Streamlit dashboard

```bash
streamlit run main.py
```

### Access the dashboard

Once executed, the dashboard will be accessible in your browser at the following address: [http://localhost:8501](http://localhost:8501)

## Usage with Docker

Make sure Docker is installed and configured (on Windows, enable WSL 2 integration in Docker Desktop).

### Build the Docker image

```bash
docker build -t my_streamlit_dashboard .
```

### Run the Docker container

```bash
docker run -p 8501:8501 my_streamlit_dashboard
```

### Access the application 
Access the application via [http://localhost:8501](http://localhost:8501)
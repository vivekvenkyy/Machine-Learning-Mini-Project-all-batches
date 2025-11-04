# ADR Regression Analysis Tool

This is a Streamlit-based web application that allows users to upload their datasets and train an Automatic Relevance Determination (ARD) regression model. The app includes dataset validation, model training, and prediction capabilities.

## Features

- Dataset validation and preprocessing
- Interactive model training with ARD regression
- Feature relevance analysis
- Prediction generation for new data
- Downloadable prediction results

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload your dataset (CSV format) and follow the interactive instructions in the web interface

## Dataset Requirements

- Must be in CSV format
- Must contain at least 10 rows
- Should have numeric features for analysis
- Should not have missing values

## Deployment

To deploy this application to a production environment:

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. You can deploy this application to various platforms that support Python web applications, such as:
   - Streamlit Cloud
   - Heroku
   - Google Cloud Platform
   - AWS Elastic Beanstalk

For Streamlit Cloud deployment:
1. Push your code to a GitHub repository
2. Visit https://share.streamlit.io/
3. Connect your GitHub account
4. Select your repository and the `app.py` file
5. Deploy!

## Project Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── src/
│   ├── data_validation.py  # Dataset validation utilities
│   └── model.py            # ARD model implementation
└── README.md
```

## Contributing

Feel free to submit issues and enhancement requests!
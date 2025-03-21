# FACTORY HACK KOREA 2025 - Forbao

## 📌 Project Overview
This project was developed for an **Industrial AI Hackathon** aimed at solving **manufacturing site issues**.  
We built an **AI-powered quality prediction and monitoring system** for **milling machine processes**, allowing real-time defect prediction and explanation of influential factors.

## 🚀 Main Objectives
- Predict defects occurring mid-process
- Enable real-time data monitoring and quality forecasting
- Identify key factors influencing defect occurrence and provide interpretable insights

## 🔍 Problem Definition
- **Automotive ball lamp manufacturing process** experiences a surge in defect rates when machining specific holes
- Process interruptions and manual inspections lower productivity
- **A defect prediction and root cause analysis system** is needed to improve quality and efficiency

## 📊 Data Overview
- **7 Key Variables**: Feed rate (ActF), spindle speed, spindle load, servo motor load, servo motor current, tool usage count (TotalCount), and other sensor data
- Data collected in **0.1-second and 0.5-second intervals**, aligned at **0.5-second resolution** for processing
- Exploratory Data Analysis (EDA) conducted to investigate defect distributions and key influencing factors

## 🏗️ Research Process
1. **Problem Definition**  
2. **EDA (Exploratory Data Analysis)**  
   - Identifying key defect-prone process phases  
   - Analyzing defect patterns for different defect types (roundness, groove depth, positioning, etc.)  
3. **Modeling (InceptionTime)**  
   - Deep learning-based **multivariate time-series classification**  
   - **Standardized time-series length** using padding and cropping  
   - Applied **InceptionTime** architecture to efficiently detect temporal patterns and enhance training stability  
4. **Quality Monitoring System Development**  
   - Interpretable AI model for defect prediction  
   - **Real-time data dashboard for process monitoring**  

## 🤖 Model Details
- **Model Architecture**: Time-series classification using **InceptionTime**
- **Input Variables**: Key sensor data (spindle load, servo motor current, feed rate, etc.)
- **Output**: Predicts defect types (OK/NG classification)
- **Feature Engineering**:  
  - Extracted key statistical features (Mean, Coefficient of Variation, Flat Spots, Lumpiness, Spike, etc.)
  - Generated **35 new engineered features** to improve prediction accuracy

## 🏆 Research Results
- **Improved defect prediction accuracy**  
  - Successfully trained models to classify defects in **groove depth, positioning, roundness, groove diameter, and inner diameter**  
  - Utilized complete time-series data for better feature representation and accuracy  
- **Developed an interpretable AI model**  
  - Provided insights into which variables influence defects  
  - Enabled domain experts to **understand and interpret AI-driven predictions**  
- **Built a real-time quality monitoring system**  
  - **Visualized defect probabilities** for immediate response  
  - Integrated with a deep-learning model for **real-time defect detection and dashboard monitoring**  

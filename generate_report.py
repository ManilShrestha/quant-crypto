from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import os

def create_report():
    # Create the PDF document
    doc = SimpleDocTemplate(
        "crypto_ml_stat_arb_report.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        alignment=TA_JUSTIFY
    ))
    
    # Content
    story = []
    
    # Title
    story.append(Paragraph("Cryptocurrency Machine Learning Statistical Arbitrage", styles['CustomTitle']))
    story.append(Spacer(1, 12))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['CustomHeading']))
    story.append(Paragraph(
        "This project implements a sophisticated trading framework that combines machine learning and statistical arbitrage "
        "strategies for cryptocurrency trading. The system is designed to analyze and trade the top 12 cryptocurrencies "
        "by trading volume, utilizing both daily and intraday data. The framework incorporates advanced feature engineering, "
        "machine learning models, and comprehensive backtesting capabilities to evaluate trading strategies.",
        styles['CustomBody']
    ))
    story.append(Spacer(1, 12))
    
    # Project Overview
    story.append(Paragraph("Project Overview", styles['CustomHeading']))
    story.append(Paragraph(
        "The project is structured as a comprehensive framework with several key components:",
        styles['CustomBody']
    ))
    
    # Components Table
    components = [
        ["Component", "Description"],
        ["Data Collection", "Fetches historical data from multiple exchanges using CCXT"],
        ["Feature Engineering", "Computes technical indicators and statistical features"],
        ["Machine Learning", "Implements XGBoost and LSTM models for prediction"],
        ["Trading Strategies", "Combines ML predictions with statistical arbitrage"],
        ["Backtesting", "Evaluates strategy performance with realistic constraints"],
        ["Performance Analysis", "Calculates key metrics and generates visualizations"]
    ]
    
    t = Table(components, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Technical Implementation
    story.append(Paragraph("Technical Implementation", styles['CustomHeading']))
    story.append(Paragraph(
        "The system is built using Python and incorporates several key technical components:",
        styles['CustomBody']
    ))
    
    # Technical Details
    tech_details = [
        ["Component", "Implementation Details"],
        ["Data Collection", "Uses CCXT library for exchange connectivity\nSupports multiple timeframes\nImplements rate limiting and error handling"],
        ["Feature Engineering", "Technical indicators using pandas_ta\nStatistical features for arbitrage\nFeature selection and normalization"],
        ["Machine Learning", "XGBoost for classification\nLSTM for time series prediction\nCross-validation and hyperparameter tuning"],
        ["Trading Logic", "Signal generation from ML predictions\nPosition sizing and risk management\nTransaction cost modeling"],
        ["Backtesting", "Realistic slippage and fees\nMultiple performance metrics\nPortfolio optimization"]
    ]
    
    t = Table(tech_details, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Performance Metrics
    story.append(Paragraph("Performance Metrics", styles['CustomHeading']))
    story.append(Paragraph(
        "The system evaluates trading strategies using several key performance metrics:",
        styles['CustomBody']
    ))
    
    metrics = [
        ["Metric", "Description"],
        ["Total Return", "Overall percentage return of the strategy"],
        ["Annual Return", "Annualized return rate"],
        ["Sharpe Ratio", "Risk-adjusted return measure"],
        ["Maximum Drawdown", "Largest peak-to-trough decline"],
        ["Win Rate", "Percentage of profitable trades"],
        ["Profit Factor", "Ratio of gross profits to gross losses"]
    ]
    
    t = Table(metrics, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Future Improvements
    story.append(Paragraph("Future Improvements", styles['CustomHeading']))
    story.append(Paragraph(
        "Several areas for future development and improvement have been identified:",
        styles['CustomBody']
    ))
    
    improvements = [
        "Implementation of deep learning models for better pattern recognition",
        "Addition of sentiment analysis from social media and news sources",
        "Integration with more cryptocurrency exchanges",
        "Development of real-time trading capabilities",
        "Enhanced risk management and portfolio optimization",
        "Implementation of market regime detection",
        "Addition of more sophisticated statistical arbitrage strategies"
    ]
    
    for improvement in improvements:
        story.append(Paragraph(f"• {improvement}", styles['CustomBody']))
    
    # Build the PDF
    doc.build(story)

if __name__ == "__main__":
    create_report() 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(
        self,
        backtest_results: str = "backtest_results.json",
        pairs_file: str = "data/processed/selected_pairs.json",
        output_dir: str = "reports"
    ):
        """Initialize the report generator.
        
        Args:
            backtest_results (str): Path to backtest results JSON file
            pairs_file (str): Path to selected pairs JSON file
            output_dir (str): Directory to save reports and plots
        """
        self.backtest_results = backtest_results
        self.pairs_file = pairs_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        with open(backtest_results, 'r') as f:
            self.results = json.load(f)
        with open(pairs_file, 'r') as f:
            self.pairs = json.load(f)
            
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_equity_curve(self):
        """Plot equity curve."""
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df.index, equity_df['capital'], label='Portfolio Value')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_file = os.path.join(self.output_dir, 'equity_curve.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
        
    def plot_drawdown(self):
        """Plot drawdown chart."""
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['capital'] - equity_df['peak']) / equity_df['peak']
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(equity_df.index, equity_df['drawdown'], 0, color='red', alpha=0.3)
        plt.plot(equity_df.index, equity_df['drawdown'], color='red')
        plt.title('Portfolio Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        
        # Save plot
        output_file = os.path.join(self.output_dir, 'drawdown.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
        
    def plot_monthly_returns(self):
        """Plot monthly returns heatmap."""
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        # Calculate monthly returns
        monthly_returns = equity_df['capital'].resample('M').last().pct_change()
        monthly_returns = monthly_returns.to_frame()
        monthly_returns.columns = ['returns']
        
        # Create heatmap data
        returns_matrix = monthly_returns['returns'].values.reshape(-1, 12)
        years = monthly_returns.index.year.unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(returns_matrix, 
                   xticklabels=months,
                   yticklabels=years,
                   cmap='RdYlGn',
                   center=0,
                   annot=True,
                   fmt='.1%')
        plt.title('Monthly Returns Heatmap')
        
        # Save plot
        output_file = os.path.join(self.output_dir, 'monthly_returns.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
        
    def plot_pair_correlation(self):
        """Plot correlation matrix for selected pairs."""
        # Create correlation matrix
        symbols = set()
        for pair in self.pairs:
            symbols.add(pair['symbol1'])
            symbols.add(pair['symbol2'])
            
        corr_matrix = pd.DataFrame(index=symbols, columns=symbols)
        for pair in self.pairs:
            corr_matrix.loc[pair['symbol1'], pair['symbol2']] = pair['correlation']
            corr_matrix.loc[pair['symbol2'], pair['symbol1']] = pair['correlation']
            
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix,
                   cmap='coolwarm',
                   center=0,
                   annot=True,
                   fmt='.2f')
        plt.title('Pair Correlation Matrix')
        
        # Save plot
        output_file = os.path.join(self.output_dir, 'pair_correlation.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
        
    def generate_pdf_report(self):
        """Generate PDF report with all plots and metrics."""
        # Create PDF
        output_file = os.path.join(self.output_dir, 'pairs_trading_report.pdf')
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph('Pairs Trading Strategy Report', title_style))
        story.append(Spacer(1, 12))
        
        # Date
        date_style = ParagraphStyle(
            'CustomDate',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.gray
        )
        story.append(Paragraph(
            f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            date_style
        ))
        story.append(Spacer(1, 30))
        
        # Performance Metrics
        story.append(Paragraph('Performance Metrics', styles['Heading2']))
        story.append(Spacer(1, 12))
        
        metrics = self.results['metrics']
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{metrics['total_return']:.2%}"],
            ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
            ['Max Drawdown', f"{metrics['max_drawdown']:.2%}"],
            ['Win Rate', f"{metrics['win_rate']:.2%}"],
            ['Profit Factor', f"{metrics['profit_factor']:.2f}"],
            ['Total Trades', str(metrics['total_trades'])]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
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
        story.append(metrics_table)
        story.append(Spacer(1, 30))
        
        # Generate and add plots
        story.append(Paragraph('Equity Curve', styles['Heading2']))
        equity_plot = self.plot_equity_curve()
        story.append(Image(equity_plot, width=6*inch, height=3*inch))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph('Portfolio Drawdown', styles['Heading2']))
        drawdown_plot = self.plot_drawdown()
        story.append(Image(drawdown_plot, width=6*inch, height=3*inch))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph('Monthly Returns', styles['Heading2']))
        monthly_returns_plot = self.plot_monthly_returns()
        story.append(Image(monthly_returns_plot, width=6*inch, height=4*inch))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph('Pair Correlations', styles['Heading2']))
        correlation_plot = self.plot_pair_correlation()
        story.append(Image(correlation_plot, width=6*inch, height=5*inch))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Generated PDF report: {output_file}")

def main():
    # Initialize report generator
    generator = ReportGenerator()
    
    # Generate report
    generator.generate_pdf_report()
    logger.info("Report generation completed")

if __name__ == "__main__":
    main() 
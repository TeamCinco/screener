import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class QuantRankingSystem:
    def __init__(self, core_features_path, distribution_path):
        """
        Initialize with your data paths
        """
        self.core_features_path = core_features_path
        self.distribution_path = distribution_path
        self.scaler = StandardScaler()
        
    def get_sector_and_mcap_data(self, symbols):
        """
        Get real sector and market cap data from Yahoo Finance
        """
        print(f"🔍 Fetching sector/market cap data for {len(symbols)} symbols...")
        
        sector_data = []
        
        # Process in batches to avoid overwhelming Yahoo Finance
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            for symbol in batch:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    sector_data.append({
                        'symbol': symbol,
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'enterprise_value': info.get('enterpriseValue', 0),
                        'shares_outstanding': info.get('sharesOutstanding', 0),
                        'float_shares': info.get('floatShares', 0),
                        'beta': info.get('beta', 1.0),
                        'pe_ratio': info.get('trailingPE', None),
                        'pb_ratio': info.get('priceToBook', None)
                    })
                    
                except Exception as e:
                    print(f"⚠️  Could not fetch data for {symbol}: {e}")
                    # Default values for failed lookups
                    sector_data.append({
                        'symbol': symbol,
                        'sector': 'Unknown',
                        'industry': 'Unknown', 
                        'market_cap': 1e9,  # Default to 1B instead of 0
                        'enterprise_value': 0,
                        'shares_outstanding': 0,
                        'float_shares': 0,
                        'beta': 1.0,
                        'pe_ratio': None,
                        'pb_ratio': None
                    })
        
        return pd.DataFrame(sector_data)
    
    def load_and_merge_data(self):
        """
        Load and merge your core features with distribution analysis and Yahoo Finance data
        """
        print("Loading core features...")
        # Load all core feature files
        import glob
        core_files = glob.glob(f"{self.core_features_path}/*.csv")
        
        core_dfs = []
        for file in core_files:
            df = pd.read_csv(file)
            core_dfs.append(df)
        
        self.core_data = pd.concat(core_dfs, ignore_index=True)
        
        print("Loading distribution analysis...")
        self.dist_data = pd.read_csv(self.distribution_path)
        
        # Get unique symbols for Yahoo Finance lookup
        symbols = self.core_data['symbol'].unique().tolist()
        
        # Get sector and market cap data
        self.yahoo_data = self.get_sector_and_mcap_data(symbols)
        
        # Merge everything together - FIXED ORDER
        print("Merging datasets...")
        self.merged_data = self.core_data.merge(
            self.dist_data, on='symbol', how='left'
        ).merge(
            self.yahoo_data, on='symbol', how='left'
        )
        
        # Ensure market_cap is properly filled
        self.merged_data['market_cap'] = self.merged_data['market_cap'].fillna(1e9)
        
        print(f"✅ Loaded data for {len(self.merged_data['symbol'].unique())} symbols")
        print(f"✅ Market cap column exists: {'market_cap' in self.merged_data.columns}")
        
        return self.merged_data
    
    def create_sector_buckets(self):
        """
        Create sector + market cap buckets using REAL data
        """
        # Clean sector data
        self.merged_data['sector_clean'] = self.merged_data['sector'].fillna('Unknown')
        self.merged_data['sector_clean'] = self.merged_data['sector_clean'].replace('Unknown', 'Other')
        
        # Create market cap quintiles (make sure market_cap exists!)
        if 'market_cap' not in self.merged_data.columns:
            print("❌ ERROR: market_cap column missing!")
            return self.merged_data
            
        # Define market cap buckets (in billions)
        def categorize_mcap(mcap):
            mcap_b = mcap / 1e9  # Convert to billions
            if mcap_b < 2:
                return 'Small'
            elif mcap_b < 10:
                return 'Mid'
            elif mcap_b < 50:
                return 'Large'
            else:
                return 'Mega'
        
        self.merged_data['cap_bucket'] = self.merged_data['market_cap'].apply(categorize_mcap)
        
        # Create combined bucket for peer group comparison
        self.merged_data['bucket'] = self.merged_data['sector_clean'] + '_' + self.merged_data['cap_bucket']
        
        # Print distributions using the merged data
        latest_data = self.merged_data.groupby('symbol').tail(1)
        
        print("\n📊 SECTOR DISTRIBUTION:")
        sector_counts = latest_data['sector_clean'].value_counts()
        print(sector_counts.head(10))
        
        print("\n💰 MARKET CAP DISTRIBUTION:")
        mcap_counts = latest_data['cap_bucket'].value_counts()
        print(mcap_counts)
        
        return self.merged_data
    
    def detect_regime(self, window=20):
        """
        Enhanced regime detection using cross-asset signals
        """
        # Calculate rolling volatility for each symbol
        self.merged_data['regime_vol'] = self.merged_data.groupby('symbol')['real_vol_20d'].rolling(window).mean().reset_index(drop=True)
        
        # Also look at correlation regimes (when correlations spike, it's usually stress)
        self.merged_data['regime_corr'] = self.merged_data.groupby('symbol')['corr_20d'].rolling(window).mean().reset_index(drop=True)
        
        # Combined regime signal
        vol_75th = self.merged_data['regime_vol'].quantile(0.75)
        vol_25th = self.merged_data['regime_vol'].quantile(0.25)
        corr_75th = self.merged_data['regime_corr'].quantile(0.75)
        
        def classify_regime(row):
            vol = row['regime_vol']
            corr = row['regime_corr']
            
            if pd.isna(vol) or pd.isna(corr):
                return 'normal'
                
            if vol > vol_75th or corr > corr_75th:
                return 'stress'  # High vol OR high correlation = stress
            elif vol < vol_25th and corr < corr_75th:
                return 'calm'    # Low vol AND normal correlation = calm
            else:
                return 'normal'  # Everything else
        
        self.merged_data['regime'] = self.merged_data[['regime_vol', 'regime_corr']].apply(classify_regime, axis=1)
        
        print("\n🎯 REGIME DISTRIBUTION:")
        regime_counts = self.merged_data['regime'].value_counts()
        print(regime_counts)
        
        return self.merged_data
    
    def build_composite_scores(self):
        """
        Build enhanced factor scores - FIXED to handle market_cap properly
        """
        # Get latest data for each symbol for scoring
        latest_data = self.merged_data.groupby('symbol').tail(1).copy()
        
        # VERIFY market_cap exists
        if 'market_cap' not in latest_data.columns:
            print("❌ CRITICAL ERROR: market_cap missing in latest_data!")
            print("Available columns:", latest_data.columns.tolist())
            return latest_data
        
        print(f"✅ Market cap range: {latest_data['market_cap'].min():,.0f} to {latest_data['market_cap'].max():,.0f}")
        
        # A) VOLATILITY REGIME SCORE (enhanced with beta)
        latest_data['vol_score'] = (
            -latest_data['vol_ratio_5_10'].fillna(1.0) +           
            -latest_data['vol_ratio_10_20'].fillna(1.0) +          
            -latest_data['real_vol_20d'].fillna(0.2) +             
            -(latest_data['beta'].fillna(1.0) - 1.0)   
        )
        
        # B) DISTRIBUTION ANOMALY SCORE
        latest_data['dist_score'] = (
            latest_data['close_return_shapiro_p'].fillna(0.5) +
            -latest_data['close_return_kurtosis'].fillna(3) +
            -np.abs(latest_data['close_return_skew'].fillna(0))
        )
        
        # C) VALUE/QUALITY SCORE
        latest_data['value_score'] = 0
        # Lower PE is better (but avoid negatives)
        pe_score = latest_data['pe_ratio'].fillna(20)
        pe_score = np.where(pe_score > 0, -np.log(pe_score + 1), 0)
        latest_data['value_score'] += pe_score
        
        # Lower PB is better
        pb_score = latest_data['pb_ratio'].fillna(2)
        pb_score = np.where(pb_score > 0, -np.log(pb_score + 1), 0)
        latest_data['value_score'] += pb_score
        
        # D) MOMENTUM SCORE
        latest_data['momentum_score'] = (
            latest_data['rel_strength_20d'].fillna(0) +
            latest_data['rel_strength_60d'].fillna(0) +
            (latest_data['RSI_14'].fillna(50) - 50) / 50
        )
        
        # E) MEAN REVERSION SCORE
        latest_data['mean_reversion_score'] = (
            -latest_data['autocorr_ret_20d'].fillna(0) +
            (50 - latest_data['RSI_14'].fillna(50)) / 50 +
            -latest_data['BB_pctB_20'].fillna(0.5)
        )
        
        # F) LIQUIDITY SCORE - FIXED
        # Ensure market_cap is positive and take log
        mcap_safe = np.maximum(latest_data['market_cap'], 1e6)  # Minimum 1M market cap
        mcap_score = np.log(mcap_safe) / 10  # Normalize log market cap
        
        latest_data['liquidity_score'] = (
            -latest_data['roll_spread_pct_20d'].fillna(0.01) +
            -latest_data['price_impact_20d'].fillna(0.01) +
            mcap_score  # Larger companies = more liquid
        )
        
        print("✅ All composite scores calculated successfully!")
        return latest_data
    
    def normalize_within_buckets(self, data):
        """
        Z-score normalize each factor within sector/size buckets
        """
        score_cols = ['vol_score', 'dist_score', 'value_score', 'momentum_score', 
                     'mean_reversion_score', 'liquidity_score']
        
        for col in score_cols:
            data[f'{col}_norm'] = data.groupby('bucket')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        
        return data
    
    def apply_regime_weighting(self, data):
        """
        Enhanced regime-conditional weighting
        """
        def get_regime_weights(regime):
            if regime == 'calm':
                return {
                    'vol_weight': 0.15,
                    'dist_weight': 0.10,
                    'value_weight': 0.25,      
                    'momentum_weight': 0.20,
                    'mean_rev_weight': 0.20,   
                    'liquidity_weight': 0.10
                }
            elif regime == 'stress':
                return {
                    'vol_weight': 0.30,        
                    'dist_weight': 0.25,       
                    'value_weight': 0.05,      
                    'momentum_weight': 0.25,   
                    'mean_rev_weight': 0.05,   
                    'liquidity_weight': 0.10   
                }
            else:  # normal
                return {
                    'vol_weight': 0.20,
                    'dist_weight': 0.15,
                    'value_weight': 0.20,
                    'momentum_weight': 0.20,
                    'mean_rev_weight': 0.15,
                    'liquidity_weight': 0.10
                }
        
        # Apply regime-specific weights
        regime_weights = data['regime'].apply(get_regime_weights)
        
        data['final_score'] = (
            data['vol_score_norm'] * regime_weights.apply(lambda x: x['vol_weight']) +
            data['dist_score_norm'] * regime_weights.apply(lambda x: x['dist_weight']) +
            data['value_score_norm'] * regime_weights.apply(lambda x: x['value_weight']) +
            data['momentum_score_norm'] * regime_weights.apply(lambda x: x['momentum_weight']) +
            data['mean_reversion_score_norm'] * regime_weights.apply(lambda x: x['mean_rev_weight']) +
            data['liquidity_score_norm'] * regime_weights.apply(lambda x: x['liquidity_weight'])
        )
        
        return data
    
    def create_rankings(self, data):
        """
        Create final rankings within each bucket
        """
        # Rank within buckets (1 = best, higher = worse)
        data['bucket_rank'] = data.groupby('bucket')['final_score'].rank(
            ascending=False, method='dense'
        )
        
        # Overall percentile rank
        data['overall_rank'] = data['final_score'].rank(pct=True, ascending=False)
        
        # Create buy/sell signals
        data['signal'] = 'HOLD'
        
        # More sophisticated signal generation
        def assign_signals(group):
            n = len(group)
            if n >= 5:  # Lower threshold for smaller buckets
                top_n = max(1, n // 5)  # Top 20%
                bottom_n = max(1, n // 5)  # Bottom 20%
                
                group = group.sort_values('final_score', ascending=False)
                group.iloc[:top_n, group.columns.get_loc('signal')] = 'BUY'
                group.iloc[-bottom_n:, group.columns.get_loc('signal')] = 'AVOID'
            
            return group
        
        data = data.groupby('bucket').apply(assign_signals).reset_index(drop=True)
        
        return data
    
    def run_full_analysis(self):
        """
        Run the complete ranking system
        """
        print("🚀 Starting Enhanced Quant Ranking System...")
        
        # Load and merge data
        self.load_and_merge_data()
        
        # Create sector buckets
        print("📊 Creating sector/size buckets...")
        self.create_sector_buckets()
        
        # Detect regimes
        print("🎯 Detecting volatility regimes...")
        self.detect_regime()
        
        # Build composite scores
        print("⚡ Building composite factor scores...")
        scored_data = self.build_composite_scores()
        
        # Normalize within buckets
        print("📈 Normalizing within peer groups...")
        scored_data = self.normalize_within_buckets(scored_data)
        
        # Apply regime weighting
        print("🎲 Applying regime-conditional weighting...")
        scored_data = self.apply_regime_weighting(scored_data)
        
        # Create final rankings
        print("🏆 Creating final rankings...")
        final_rankings = self.create_rankings(scored_data)
        
        return final_rankings
    
    def get_trading_signals(self, top_n=20):
        """
        Get the actual trading signals with enhanced output
        """
        results = self.run_full_analysis()
        
        # Filter for actionable signals
        buy_signals = results[results['signal'] == 'BUY'].nlargest(top_n, 'final_score')
        avoid_signals = results[results['signal'] == 'AVOID'].nsmallest(top_n, 'final_score')
        
        print(f"\n🔥 TOP {top_n} BUY SIGNALS:")
        display_cols = ['symbol', 'sector_clean', 'cap_bucket', 'regime', 'final_score', 'overall_rank']
        if len(buy_signals) > 0:
            print(buy_signals[display_cols].to_string())
        else:
            print("No buy signals generated")
        
        print(f"\n❌ TOP {top_n} AVOID SIGNALS:")
        if len(avoid_signals) > 0:
            print(avoid_signals[display_cols].to_string())
        else:
            print("No avoid signals generated")
        
        return buy_signals, avoid_signals, results

# USAGE
if __name__ == "__main__":
    ranker = QuantRankingSystem(
        core_features_path="/Users/jazzhashzzz/Desktop/Cinco-Quant/02_feature_eng/output/core_features_20250514",
        distribution_path="/Users/jazzhashzzz/Desktop/Cinco-Quant/stock_distribution_analysis_full.csv"
    )
    
    buy_signals, avoid_signals, all_results = ranker.get_trading_signals(top_n=25)
    all_results.to_csv("enhanced_quant_rankings.csv", index=False)
    print("\n✅ Results saved to enhanced_quant_rankings.csv")
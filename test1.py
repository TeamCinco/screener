import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class EnhancedQuantRankingSystem:
    def __init__(self, core_features_path, distribution_path):
        """
        Enhanced Quant Ranking System with all improvements
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
                        'pb_ratio': info.get('priceToBook', None),
                        'revenue_growth': info.get('revenueGrowth', None),
                        'profit_margin': info.get('profitMargins', None),
                        'debt_to_equity': info.get('debtToEquity', None)
                    })
                    
                except Exception as e:
                    print(f"⚠️  Could not fetch data for {symbol}: {e}")
                    sector_data.append({
                        'symbol': symbol,
                        'sector': 'Unknown',
                        'industry': 'Unknown', 
                        'market_cap': 1e9,
                        'enterprise_value': 0,
                        'shares_outstanding': 0,
                        'float_shares': 0,
                        'beta': 1.0,
                        'pe_ratio': None,
                        'pb_ratio': None,
                        'revenue_growth': None,
                        'profit_margin': None,
                        'debt_to_equity': None
                    })
        
        return pd.DataFrame(sector_data)
    
    def load_and_merge_data(self):
        """
        Load and merge all datasets
        """
        print("Loading core features...")
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
        
        # Merge everything together
        print("Merging datasets...")
        self.merged_data = self.core_data.merge(
            self.dist_data, on='symbol', how='left'
        ).merge(
            self.yahoo_data, on='symbol', how='left'
        )
        
        # Ensure market_cap is properly filled
        self.merged_data['market_cap'] = self.merged_data['market_cap'].fillna(1e9)
        
        print(f"✅ Loaded data for {len(self.merged_data['symbol'].unique())} symbols")
        return self.merged_data
    
    def create_enhanced_sector_buckets(self):
        """
        Enhanced sector bucketing with minimum viable bucket sizes
        """
        # Clean sector data
        self.merged_data['sector_clean'] = self.merged_data['sector'].fillna('Unknown')
        self.merged_data['sector_clean'] = self.merged_data['sector_clean'].replace('Unknown', 'Other')
        
        # Enhanced market cap buckets
        def categorize_mcap(mcap):
            mcap_b = mcap / 1e9
            if mcap_b < 1:
                return 'Micro'
            elif mcap_b < 5:
                return 'Small'
            elif mcap_b < 25:
                return 'Mid'
            elif mcap_b < 100:
                return 'Large'
            else:
                return 'Mega'
        
        self.merged_data['cap_bucket'] = self.merged_data['market_cap'].apply(categorize_mcap)
        
        # Create initial buckets
        self.merged_data['initial_bucket'] = self.merged_data['sector_clean'] + '_' + self.merged_data['cap_bucket']
        
        # Check bucket sizes and consolidate small ones
        latest_data = self.merged_data.groupby('symbol').tail(1)
        bucket_sizes = latest_data['initial_bucket'].value_counts()
        
        # Consolidate buckets with <5 stocks
        small_buckets = bucket_sizes[bucket_sizes < 5].index.tolist()
        
        def consolidate_bucket(bucket):
            if bucket in small_buckets:
                sector = bucket.split('_')[0]
                return f"{sector}_Mixed"  # Combine all small cap buckets within sector
            return bucket
        
        self.merged_data['bucket'] = self.merged_data['initial_bucket'].apply(consolidate_bucket)
        
        # Final bucket size check
        final_bucket_sizes = self.merged_data.groupby('symbol').tail(1)['bucket'].value_counts()
        
        print("\n📊 ENHANCED SECTOR DISTRIBUTION:")
        sector_counts = latest_data['sector_clean'].value_counts()
        print(sector_counts.head(10))
        
        print("\n💰 ENHANCED MARKET CAP DISTRIBUTION:")
        mcap_counts = latest_data['cap_bucket'].value_counts()
        print(mcap_counts)
        
        print("\n🏗️ FINAL BUCKET SIZES:")
        print(final_bucket_sizes.head(15))
        
        return self.merged_data
    
    def improved_regime_detection(self, window=20):
        """
        Enhanced multi-dimensional regime detection
        """
        # Calculate multiple regime indicators
        self.merged_data['regime_vol'] = self.merged_data.groupby('symbol')['real_vol_20d'].rolling(window).mean().reset_index(drop=True)
        self.merged_data['regime_corr'] = self.merged_data.groupby('symbol')['corr_20d'].rolling(window).mean().reset_index(drop=True)
        self.merged_data['regime_momentum'] = self.merged_data.groupby('symbol')['rel_strength_20d'].rolling(window).mean().reset_index(drop=True)
        
        # More balanced thresholds
        vol_33rd = self.merged_data['regime_vol'].quantile(0.33)
        vol_67th = self.merged_data['regime_vol'].quantile(0.67)
        corr_67th = self.merged_data['regime_corr'].quantile(0.67)
        mom_33rd = self.merged_data['regime_momentum'].quantile(0.33)
        mom_67th = self.merged_data['regime_momentum'].quantile(0.67)
        
        def classify_regime_enhanced(row):
            vol = row['regime_vol']
            corr = row['regime_corr']
            mom = row['regime_momentum']
            
            if pd.isna(vol) or pd.isna(corr) or pd.isna(mom):
                return 'normal'
            
            # Multi-dimensional regime classification
            if vol > vol_67th and corr > corr_67th:
                return 'crisis'
            elif vol > vol_67th and mom > mom_67th:
                return 'volatile_trending'
            elif vol > vol_67th and mom < mom_33rd:
                return 'volatile_reverting'
            elif vol < vol_33rd and mom > mom_67th:
                return 'calm_trending'
            elif vol < vol_33rd and mom < mom_33rd:
                return 'calm_reverting'
            else:
                return 'normal'
        
        self.merged_data['regime'] = self.merged_data[['regime_vol', 'regime_corr', 'regime_momentum']].apply(classify_regime_enhanced, axis=1)
        
        print("\n🎯 ENHANCED REGIME DISTRIBUTION:")
        regime_counts = self.merged_data['regime'].value_counts()
        print(regime_counts)
        
        return self.merged_data
    
    def build_enhanced_composite_scores(self):
        """
        Enhanced factor scoring with cross-sectional elements
        """
        # Get latest data for scoring
        latest_data = self.merged_data.groupby('symbol').tail(1).copy()
        
        # A) ENHANCED VOLATILITY SCORE
        latest_data['vol_score'] = (
            -latest_data['vol_ratio_5_10'].fillna(1.0) +
            -latest_data['vol_ratio_10_20'].fillna(1.0) +
            -latest_data['real_vol_20d'].fillna(0.2) +
            -(latest_data['beta'].fillna(1.0) - 1.0)
        )
        
        # B) ENHANCED DISTRIBUTION ANOMALY SCORE
        latest_data['dist_score'] = (
            latest_data['close_return_shapiro_p'].fillna(0.5) +
            -latest_data['close_return_kurtosis'].fillna(3) +
            -np.abs(latest_data['close_return_skew'].fillna(0)) +
            -latest_data['close_return_jb_p'].fillna(0.5)  # Jarque-Bera test
        )
        
        # C) ENHANCED VALUE/QUALITY SCORE
        latest_data['value_score'] = 0
        
        # PE ratio score (lower is better)
        pe_score = latest_data['pe_ratio'].fillna(20)
        pe_score = np.where(pe_score > 0, -np.log(pe_score + 1), 0)
        latest_data['value_score'] += pe_score
        
        # PB ratio score (lower is better)
        pb_score = latest_data['pb_ratio'].fillna(2)
        pb_score = np.where(pb_score > 0, -np.log(pb_score + 1), 0)
        latest_data['value_score'] += pb_score
        
        # Profitability score (higher is better)
        profit_score = latest_data['profit_margin'].fillna(0.1)
        profit_score = np.where(profit_score > 0, np.log(profit_score + 0.01), -1)
        latest_data['value_score'] += profit_score
        
        # D) ENHANCED MOMENTUM SCORE (Multi-timeframe)
        latest_data['momentum_score'] = (
            latest_data['rel_strength_20d'].fillna(0) * 0.4 +   # Short-term
            latest_data['rel_strength_60d'].fillna(0) * 0.4 +   # Medium-term
            latest_data['rel_strength_120d'].fillna(0) * 0.2 +  # Long-term
            (latest_data['RSI_14'].fillna(50) - 50) / 50 * 0.3   # Technical momentum
        )
        
        # E) ENHANCED MEAN REVERSION SCORE
        latest_data['mean_reversion_score'] = (
            -latest_data['autocorr_ret_20d'].fillna(0) +
            (50 - latest_data['RSI_14'].fillna(50)) / 50 +
            -latest_data['BB_pctB_20'].fillna(0.5) +
            -(latest_data['momentum_score'] * 0.5)  # Inverse of momentum
        )
        
        # F) ENHANCED LIQUIDITY SCORE
        mcap_safe = np.maximum(latest_data['market_cap'], 1e6)
        mcap_score = np.log(mcap_safe) / 15
        
        latest_data['liquidity_score'] = (
            -latest_data['roll_spread_pct_20d'].fillna(0.01) +
            -latest_data['price_impact_20d'].fillna(0.01) +
            mcap_score +
            np.log(latest_data['Volume'].fillna(100000) + 1000) / 10  # Volume component
        )
        
        # G) NEW: CROSS-SECTIONAL SCORES
        # Rank within sector
        latest_data['sector_momentum_rank'] = latest_data.groupby('sector_clean')['momentum_score'].rank(pct=True)
        latest_data['sector_value_rank'] = latest_data.groupby('sector_clean')['value_score'].rank(pct=True, ascending=False)
        latest_data['sector_vol_rank'] = latest_data.groupby('sector_clean')['vol_score'].rank(pct=True, ascending=False)
        
        # Cross-sectional score
        latest_data['cross_sectional_score'] = (
            latest_data['sector_momentum_rank'] * 0.4 +
            latest_data['sector_value_rank'] * 0.4 +
            latest_data['sector_vol_rank'] * 0.2
        )
        
        # H) NEW: TIME-SERIES MOMENTUM
        # Calculate returns over different periods (using relative strength as proxy)
        latest_data['ts_momentum_score'] = (
            latest_data['rel_strength_20d'].fillna(0) * 0.3 +    # 1-month
            latest_data['rel_strength_60d'].fillna(0) * 0.5 +    # 3-month  
            latest_data['rel_strength_120d'].fillna(0) * 0.2     # 6-month
        )
        
        print("✅ All enhanced composite scores calculated!")
        return latest_data
    
    def analyze_factor_correlations(self, data):
        """
        Analyze correlations between factor scores
        """
        factor_cols = ['vol_score', 'dist_score', 'value_score', 'momentum_score', 
                      'mean_reversion_score', 'liquidity_score', 'cross_sectional_score', 'ts_momentum_score']
        
        # Only use columns that exist
        existing_cols = [col for col in factor_cols if col in data.columns]
        
        if len(existing_cols) > 1:
            corr_matrix = data[existing_cols].corr()
            print("\n🔍 FACTOR CORRELATION MATRIX:")
            print(corr_matrix.round(3))
            
            # Flag highly correlated factors
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr_pairs:
                print("\n⚠️  HIGH CORRELATIONS DETECTED:")
                for col1, col2, corr_val in high_corr_pairs:
                    print(f"   {col1} <-> {col2}: {corr_val:.3f}")
            else:
                print("\n✅ No problematic correlations found")
        
        return data
    
    def normalize_within_buckets_enhanced(self, data):
        """
        Enhanced normalization with outlier handling
        """
        score_cols = ['vol_score', 'dist_score', 'value_score', 'momentum_score', 
                     'mean_reversion_score', 'liquidity_score', 'cross_sectional_score', 'ts_momentum_score']
        
        # Only normalize existing columns
        existing_cols = [col for col in score_cols if col in data.columns]
        
        for col in existing_cols:
            # Winsorize outliers before normalization (clip to 5th and 95th percentiles)
            data[f'{col}_winsorized'] = data.groupby('bucket')[col].transform(
                lambda x: np.clip(x, x.quantile(0.05), x.quantile(0.95))
            )
            
            # Z-score normalize within buckets
            data[f'{col}_norm'] = data.groupby('bucket')[f'{col}_winsorized'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
        
        return data
    
    def apply_adaptive_regime_weighting(self, data):
        """
        Adaptive weighting based on regime AND market conditions
        """
        def get_adaptive_regime_weights(regime, market_conditions):
            market_trend = market_conditions.get('trend', 0)
            market_vol = market_conditions.get('vol', 0)
            
            base_weights = {
                'crisis': {
                    'vol_weight': 0.35, 'dist_weight': 0.25, 'value_weight': 0.05,
                    'momentum_weight': 0.15, 'mean_rev_weight': 0.05, 'liquidity_weight': 0.10,
                    'cross_sectional_weight': 0.05, 'ts_momentum_weight': 0.05
                },
                'volatile_trending': {
                    'vol_weight': 0.25, 'dist_weight': 0.15, 'value_weight': 0.10,
                    'momentum_weight': 0.30, 'mean_rev_weight': 0.05, 'liquidity_weight': 0.10,
                    'cross_sectional_weight': 0.15, 'ts_momentum_weight': 0.15
                },
                'volatile_reverting': {
                    'vol_weight': 0.30, 'dist_weight': 0.20, 'value_weight': 0.15,
                    'momentum_weight': 0.05, 'mean_rev_weight': 0.20, 'liquidity_weight': 0.10,
                    'cross_sectional_weight': 0.10, 'ts_momentum_weight': 0.05
                },
                'calm_trending': {
                    'vol_weight': 0.10, 'dist_weight': 0.10, 'value_weight': 0.20,
                    'momentum_weight': 0.35, 'mean_rev_weight': 0.05, 'liquidity_weight': 0.10,
                    'cross_sectional_weight': 0.20, 'ts_momentum_weight': 0.20
                },
                'calm_reverting': {
                    'vol_weight': 0.15, 'dist_weight': 0.10, 'value_weight': 0.30,
                    'momentum_weight': 0.10, 'mean_rev_weight': 0.25, 'liquidity_weight': 0.10,
                    'cross_sectional_weight': 0.15, 'ts_momentum_weight': 0.05
                },
                'normal': {
                    'vol_weight': 0.20, 'dist_weight': 0.15, 'value_weight': 0.20,
                    'momentum_weight': 0.20, 'mean_rev_weight': 0.15, 'liquidity_weight': 0.10,
                    'cross_sectional_weight': 0.15, 'ts_momentum_weight': 0.10
                }
            }
            
            return base_weights.get(regime, base_weights['normal'])
        
        # Calculate market conditions
        market_conditions = {
            'trend': data['momentum_score_norm'].mean(),
            'vol': data['vol_score_norm'].mean()
        }
        
        # Apply adaptive weights
        regime_weights = data['regime'].apply(lambda x: get_adaptive_regime_weights(x, market_conditions))
        
        # Build final score with all available factors
        factor_components = []
        
        if 'vol_score_norm' in data.columns:
            factor_components.append(data['vol_score_norm'] * regime_weights.apply(lambda x: x['vol_weight']))
        if 'dist_score_norm' in data.columns:
            factor_components.append(data['dist_score_norm'] * regime_weights.apply(lambda x: x['dist_weight']))
        if 'value_score_norm' in data.columns:
            factor_components.append(data['value_score_norm'] * regime_weights.apply(lambda x: x['value_weight']))
        if 'momentum_score_norm' in data.columns:
            factor_components.append(data['momentum_score_norm'] * regime_weights.apply(lambda x: x['momentum_weight']))
        if 'mean_reversion_score_norm' in data.columns:
            factor_components.append(data['mean_reversion_score_norm'] * regime_weights.apply(lambda x: x['mean_rev_weight']))
        if 'liquidity_score_norm' in data.columns:
            factor_components.append(data['liquidity_score_norm'] * regime_weights.apply(lambda x: x['liquidity_weight']))
        if 'cross_sectional_score_norm' in data.columns:
            factor_components.append(data['cross_sectional_score_norm'] * regime_weights.apply(lambda x: x['cross_sectional_weight']))
        if 'ts_momentum_score_norm' in data.columns:
            factor_components.append(data['ts_momentum_score_norm'] * regime_weights.apply(lambda x: x['ts_momentum_weight']))
        
        # Sum all components
        data['final_score'] = sum(factor_components)
        
        print(f"✅ Applied adaptive weighting using {len(factor_components)} factors")
        return data
    
    def create_enhanced_rankings(self, data):
        """
        Enhanced ranking with multiple signal types
        """
        # Rank within buckets
        data['bucket_rank'] = data.groupby('bucket')['final_score'].rank(ascending=False, method='dense')
        data['overall_rank'] = data['final_score'].rank(pct=True, ascending=False)
        
        # Create multiple signal types
        data['signal'] = 'HOLD'
        data['strength'] = 'NEUTRAL'
        
        def assign_enhanced_signals(group):
            n = len(group)
            if n >= 3:  # Minimum 3 stocks for signals
                top_20pct = max(1, int(n * 0.2))
                bottom_20pct = max(1, int(n * 0.2))
                
                group = group.sort_values('final_score', ascending=False)
                
                # Strong signals for top/bottom 20%
                group.iloc[:top_20pct, group.columns.get_loc('signal')] = 'BUY'
                group.iloc[:max(1, top_20pct//2), group.columns.get_loc('strength')] = 'STRONG'
                
                group.iloc[-bottom_20pct:, group.columns.get_loc('signal')] = 'AVOID'
                group.iloc[-max(1, bottom_20pct//2):, group.columns.get_loc('strength')] = 'STRONG'
            
            return group
        
        data = data.groupby('bucket').apply(assign_enhanced_signals).reset_index(drop=True)
        
        return data
    
    def run_enhanced_analysis(self):
        """
        Run the complete enhanced analysis
        """
        print("🚀 Starting ENHANCED Quant Ranking System...")
        
        # Load and merge data
        self.load_and_merge_data()
        
        # Enhanced sector buckets
        print("📊 Creating enhanced sector/size buckets...")
        self.create_enhanced_sector_buckets()
        
        # Enhanced regime detection
        print("🎯 Detecting enhanced volatility regimes...")
        self.improved_regime_detection()
        
        # Build enhanced composite scores
        print("⚡ Building enhanced composite factor scores...")
        scored_data = self.build_enhanced_composite_scores()
        
        # Analyze factor correlations
        print("🔍 Analyzing factor correlations...")
        scored_data = self.analyze_factor_correlations(scored_data)
        
        # Enhanced normalization
        print("📈 Enhanced normalization within peer groups...")
        scored_data = self.normalize_within_buckets_enhanced(scored_data)
        
        # Adaptive regime weighting
        print("🎲 Applying adaptive regime-conditional weighting...")
        scored_data = self.apply_adaptive_regime_weighting(scored_data)
        
        # Enhanced rankings
        print("🏆 Creating enhanced rankings...")
        final_rankings = self.create_enhanced_rankings(scored_data)
        
        return final_rankings
    
    def get_enhanced_trading_signals(self, top_n=25):
        """
        Get enhanced trading signals with detailed analysis
        """
        results = self.run_enhanced_analysis()
        
        # Filter for different signal strengths
        strong_buys = results[(results['signal'] == 'BUY') & (results['strength'] == 'STRONG')].nlargest(top_n, 'final_score')
        regular_buys = results[(results['signal'] == 'BUY') & (results['strength'] == 'NEUTRAL')].nlargest(top_n, 'final_score')
        
        strong_avoids = results[(results['signal'] == 'AVOID') & (results['strength'] == 'STRONG')].nsmallest(top_n, 'final_score')
        regular_avoids = results[(results['signal'] == 'AVOID') & (results['strength'] == 'NEUTRAL')].nsmallest(top_n, 'final_score')
        
        # Display results
        display_cols = ['symbol', 'sector_clean', 'cap_bucket', 'regime', 'final_score', 'overall_rank', 'bucket_rank']
        
        print(f"\n🔥 TOP {len(strong_buys)} STRONG BUY SIGNALS:")
        if len(strong_buys) > 0:
            print(strong_buys[display_cols].to_string())
        
        print(f"\n💚 TOP {len(regular_buys)} REGULAR BUY SIGNALS:")
        if len(regular_buys) > 0:
            print(regular_buys[display_cols].to_string())
        
        print(f"\n🚨 TOP {len(strong_avoids)} STRONG AVOID SIGNALS:")
        if len(strong_avoids) > 0:
            print(strong_avoids[display_cols].to_string())
        
        print(f"\n⚠️ TOP {len(regular_avoids)} REGULAR AVOID SIGNALS:")
        if len(regular_avoids) > 0:
            print(regular_avoids[display_cols].to_string())
        
        # Summary statistics
        print(f"\n📊 SIGNAL SUMMARY:")
        signal_summary = results['signal'].value_counts()
        print(signal_summary)
        
        regime_summary = results['regime'].value_counts()
        print(f"\n🎯 REGIME SUMMARY:")
        print(regime_summary)
        
        return strong_buys, regular_buys, strong_avoids, regular_avoids, results
    
    def generate_portfolio_allocation(self, results, max_positions=20, max_sector_weight=0.3):
        """
        Generate optimal portfolio allocation from signals
        """
        # Get all buy signals
        buy_signals = results[results['signal'] == 'BUY'].copy()
        
        if len(buy_signals) == 0:
            print("❌ No buy signals generated")
            return None
        
        # Sort by final score
        buy_signals = buy_signals.sort_values('final_score', ascending=False)
        
        # Apply portfolio constraints
        portfolio = []
        sector_weights = {}
        
        for _, stock in buy_signals.iterrows():
            sector = stock['sector_clean']
            
            # Check sector concentration
            current_sector_weight = sector_weights.get(sector, 0)
            if current_sector_weight >= max_sector_weight:
                continue
            
            # Add to portfolio
            portfolio.append(stock)
            sector_weights[sector] = current_sector_weight + (1/max_positions)
            
            if len(portfolio) >= max_positions:
                break
        
        portfolio_df = pd.DataFrame(portfolio)
        
        print(f"\n💼 PORTFOLIO ALLOCATION ({len(portfolio)} positions):")
        print(portfolio_df[['symbol', 'sector_clean', 'final_score', 'regime']].to_string())
        
        print(f"\n🏭 SECTOR ALLOCATION:")
        sector_allocation = portfolio_df['sector_clean'].value_counts()
        print(sector_allocation)
        
        return portfolio_df

# USAGE
if __name__ == "__main__":
    # Initialize enhanced system
    enhanced_ranker = EnhancedQuantRankingSystem(
        core_features_path="/Users/jazzhashzzz/Desktop/Cinco-Quant/02_feature_eng/output/core_features_20250514",
        distribution_path="/Users/jazzhashzzz/Desktop/Cinco-Quant/stock_distribution_analysis_full.csv"
    )
    
    # Get enhanced trading signals
    strong_buys, regular_buys, strong_avoids, regular_avoids, all_results = enhanced_ranker.get_enhanced_trading_signals(top_n=20)
    
    # Generate portfolio allocation
    portfolio = enhanced_ranker.generate_portfolio_allocation(all_results, max_positions=20, max_sector_weight=0.25)
    
    # Save results
    all_results.to_csv("enhanced_quant_rankings_v2.csv", index=False)
    if portfolio is not None:
        portfolio.to_csv("recommended_portfolio.csv", index=False)
    
    print("\n✅ Enhanced results saved!")
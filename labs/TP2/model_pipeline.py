"""
Machine Learning Pipeline for Delivery Time Analysis
Modular implementation of delivery time clustering and analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
import joblib

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DeliveryTimePipeline:
    """
    Modular pipeline for delivery time analysis and clustering
    """
    
    def __init__(self):
        self.df = None
        self.agg_data = None
        self.kmeans_model = None
        self.scaler = None
        self.cluster_features = None
        
    def prepare_data(self, file_path):
        """
        Load and preprocess the delivery data
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Cleaned and processed DataFrame
        """
        print("=== STEP 1: DATA PREPARATION ===")
        
        # Load dataset
        self.df = pd.read_csv(file_path)
        print(f"Original data shape: {self.df.shape}")
        
        # Convert timestamps
        self.df["created_at2"] = pd.to_datetime(self.df["created_at"], utc=True)
        self.df["actual_delivery_time2"] = pd.to_datetime(self.df["actual_delivery_time"], utc=True)
        self.df["created_at"] = pd.to_datetime(self.df["created_at"], utc=True).dt.tz_convert("Asia/Kolkata")
        self.df["actual_delivery_time"] = pd.to_datetime(self.df["actual_delivery_time"], utc=True).dt.tz_convert("Asia/Kolkata")

        # Timezone adjustment
        self.df['created_at_shifted'] = self.df['created_at2'] - pd.Timedelta(hours=7)
        self.df['actual_delivery_time_shifted'] = self.df['actual_delivery_time2'] - pd.Timedelta(hours=7)

        # Compute delivery time
        self.df["delivery_time"] = (self.df["actual_delivery_time"] - self.df["created_at"]).dt.total_seconds() / 60
        self.df['delivery_duration'] = (self.df['actual_delivery_time_shifted'] - self.df['created_at_shifted']).dt.total_seconds() / 60
        
        # Data cleaning
        self._clean_data()
        
        # Feature engineering
        self._engineer_features()
        
        # Aggregate for clustering
        self._aggregate_for_clustering()
        
        print("✓ Data preparation completed successfully")
        return self.df
    
    def _clean_data(self):
        """Clean the dataset by removing invalid entries"""
        # Drop rows where categorical dimensions are missing
        self.df = self.df.dropna(subset=["store_primary_category", "order_protocol", "market_id"])

        # Remove invalid delivery times
        self.df = self.df[(self.df["delivery_time"] > 0) & (self.df["delivery_time"] < 500)]

        # Fix partner/order anomalies
        for col in ["total_onshift_partners", "total_busy_partners", "total_outstanding_orders"]:
            self.df = self.df[self.df[col] >= 0]
            
        print(f"After cleaning: {len(self.df)} rows remaining")
    
    def _engineer_features(self):
        """Create features for analysis and modeling"""
        # Time features
        self.df['hour'] = self.df['created_at_shifted'].dt.hour
        self.df['weekday'] = self.df['created_at_shifted'].dt.weekday
        self.df['is_weekend'] = self.df['weekday'].isin([5,6]).astype(int)
        
        # Load ratio calculation
        if 'total_onshift_partners' in self.df.columns and 'total_busy_partners' in self.df.columns:
            self.df['total_onshift_partners'] = self.df['total_onshift_partners'].replace(0, np.nan)
            self.df['load_ratio'] = (self.df['total_busy_partners'] / self.df['total_onshift_partners']).fillna(0).clip(0, 2)
        else:
            self.df['load_ratio'] = 0.0
            
        # Time bin for aggregation
        self.df['time_bin'] = (self.df['created_at_shifted'].dt.hour * 60 + self.df['created_at_shifted'].dt.minute) // 15
        self.df['date'] = self.df['created_at_shifted'].dt.date
        
        print("✓ Feature engineering completed")
    
    def _aggregate_for_clustering(self):
        """Aggregate data into time bins for clustering analysis"""
        print("=== STEP 2: DATA AGGREGATION FOR CLUSTERING ===")
        
        # Aggregate by time bins
        self.agg_data = self.df.groupby(['date','time_bin']).agg(
            orders_count=('load_ratio', 'count'),
            avg_onshift=('total_onshift_partners', 'mean'),
            avg_busy=('total_busy_partners', 'mean'),
            mean_load_ratio=('load_ratio', 'mean'),
            avg_outstanding=('total_outstanding_orders', 'mean')
        ).reset_index()

        # Create business features
        self.agg_data['utilization_rate'] = self.agg_data['avg_busy'] / self.agg_data['avg_onshift'].replace(0, np.nan)
        self.agg_data['utilization_rate'] = self.agg_data['utilization_rate'].fillna(0).clip(0, 1)
        
        self.agg_data['demand_supply_ratio'] = self.agg_data['orders_count'] / self.agg_data['avg_onshift'].replace(0, 1)
        self.agg_data['demand_supply_ratio'] = self.agg_data['demand_supply_ratio'].clip(0, 10)
        
        self.agg_data['order_intensity'] = self.agg_data['orders_count'] / 0.25
        self.agg_data['supply_gap'] = self.agg_data['orders_count'] - self.agg_data['avg_onshift']
        self.agg_data['driver_efficiency'] = self.agg_data['orders_count'] / self.agg_data['avg_busy'].replace(0, 1)
        
        # Remove low activity periods
        initial_count = len(self.agg_data)
        self.agg_data = self.agg_data[self.agg_data['orders_count'] >= 1]
        print(f"Removed {initial_count - len(self.agg_data)} low-activity time bins")
        
        print(f"Aggregated data shape: {self.agg_data.shape}")
    
    def prepare_clustering_features(self):
        """
        Prepare features for clustering model
        
        Returns:
            tuple: (X_scaled, feature_names)
        """
        print("=== STEP 3: PREPARING CLUSTERING FEATURES ===")
        
        # Select features for clustering
        self.cluster_features = [
            'orders_count',           # Demand volume
            'avg_onshift',           # Supply capacity
            'utilization_rate',      # System utilization
            'demand_supply_ratio',   # Core business metric
            'driver_efficiency'      # Operational efficiency
        ]
        
        # Ensure all features exist
        self.cluster_features = [c for c in self.cluster_features if c in self.agg_data.columns]
        print(f"Selected features: {self.cluster_features}")
        
        # Prepare feature matrix
        X = self.agg_data[self.cluster_features].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Remove extreme outliers
        for feature in self.cluster_features:
            Q1 = X[feature].quantile(0.01)
            Q3 = X[feature].quantile(0.99)
            X[feature] = X[feature].clip(Q1, Q3)
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Final feature matrix shape: {X_scaled.shape}")
        return X_scaled, self.cluster_features
    
    def train_model(self, X_scaled, k_range=None):
        """
        Train KMeans clustering model
        
        Args:
            X_scaled (array): Scaled feature matrix
            k_range (range): Range of k values to test
            
        Returns:
            KMeans: Trained KMeans model
        """
        print("=== STEP 4: MODEL TRAINING ===")
        
        if k_range is None:
            k_range = range(2, 6)
            
        results = []
        
        # Find optimal k
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
            labels = kmeans.fit_predict(X_scaled)

            sil_score = silhouette_score(X_scaled, labels)
            ch_score = calinski_harabasz_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)

            results.append({
                'k': k,
                'silhouette': sil_score,
                'calinski_harabasz': ch_score,
                'davies_bouldin': db_score,
                'inertia': kmeans.inertia_
            })
        
        results_df = pd.DataFrame(results)
        print("Cluster evaluation results:")
        print(results_df)
        
        # Select best k based on silhouette score
        best_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
        print(f"Selected optimal k: {best_k}")
        
        # Train final model
        self.kmeans_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        self.agg_data['cluster'] = cluster_labels
        
        print("✓ Model training completed successfully")
        return self.kmeans_model
    
    def evaluate_model(self, X_scaled):
        """
        Evaluate the clustering model and provide business insights
        
        Args:
            X_scaled (array): Scaled feature matrix
            
        Returns:
            dict: Evaluation metrics and insights
        """
        print("=== STEP 5: MODEL EVALUATION ===")
        
        if self.kmeans_model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Calculate metrics
        silhouette_avg = silhouette_score(X_scaled, self.agg_data['cluster'])
        ch_score = calinski_harabasz_score(X_scaled, self.agg_data['cluster'])
        db_score = davies_bouldin_score(X_scaled, self.agg_data['cluster'])
        
        # Cluster profiling
        cluster_profiles = self.agg_data.groupby('cluster').agg({
            'orders_count': ['mean', 'std', 'count'],
            'avg_onshift': ['mean', 'std'],
            'utilization_rate': ['mean', 'std'],
            'demand_supply_ratio': ['mean', 'std']
        }).round(3)
        
        # Business categorization
        self._categorize_clusters()
        
        # Generate insights
        insights = self._generate_business_insights()
        
        evaluation_results = {
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': ch_score,
            'davies_bouldin_score': db_score,
            'cluster_profiles': cluster_profiles,
            'insights': insights
        }
        
        print(f"✓ Model evaluation completed")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        return evaluation_results
    
    def _categorize_clusters(self):
        """Categorize clusters into business segments (Rush/Moderate/Non-rush)"""
        cluster_stats = self.agg_data.groupby('cluster').agg({
            'orders_count': 'mean',
            'avg_onshift': 'mean',
            'utilization_rate': 'mean',
            'demand_supply_ratio': 'mean'
        }).round(2)
        
        rush_clusters = []
        moderate_clusters = []
        non_rush_clusters = []

        for cluster_id in cluster_stats.index:
            util_rate = cluster_stats.loc[cluster_id, 'utilization_rate']
            orders = cluster_stats.loc[cluster_id, 'orders_count']
            demand_ratio = cluster_stats.loc[cluster_id, 'demand_supply_ratio']

            if util_rate > 0.75 and orders > 80:
                rush_clusters.append(cluster_id)
            elif util_rate > 0.5 or orders > 30:
                moderate_clusters.append(cluster_id)
            else:
                non_rush_clusters.append(cluster_id)

        def categorize_rush(row):
            if row['cluster'] in rush_clusters:
                return 'Rush'
            elif row['cluster'] in moderate_clusters:
                return 'Moderate'
            else:
                return 'Non-rush'

        self.agg_data['rush_category'] = self.agg_data.apply(categorize_rush, axis=1)
        
        print("✓ Cluster categorization completed")
    
    def _generate_business_insights(self):
        """Generate business insights from clustering results"""
        print("=== STEP 6: BUSINESS INSIGHTS GENERATION ===")
        
        insights = {}
        
        # Basic statistics
        total_orders = self.agg_data['orders_count'].sum()
        category_distribution = self.agg_data['rush_category'].value_counts(normalize=True) * 100
        
        # Orders per category
        orders_by_category = self.agg_data.groupby('rush_category')['orders_count'].sum()
        orders_percentage = (orders_by_category / total_orders * 100).round(1)
        
        # Temporal patterns
        self.agg_data['hour_of_day'] = (self.agg_data['time_bin'] * 15) // 60
        category_by_hour = self.agg_data.groupby(['hour_of_day', 'rush_category']).size().unstack(fill_value=0)
        category_by_hour_pct = category_by_hour.div(category_by_hour.sum(axis=1), axis=0) * 100
        
        insights['category_distribution'] = category_distribution
        insights['orders_by_category'] = orders_percentage
        insights['hourly_patterns'] = category_by_hour_pct
        insights['total_time_bins'] = len(self.agg_data)
        insights['total_orders'] = total_orders
        
        # Print summary
        print("\n📊 BUSINESS INSIGHTS SUMMARY:")
        print(f"Category Distribution: {category_distribution.to_dict()}")
        print(f"Orders by Category: {orders_percentage.to_dict()}")
        
        return insights
    
    def save_model(self, model_path='kmeans_model.joblib', scaler_path='scaler.joblib'):
        """
        Save the trained model and scaler
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler
        """
        if self.kmeans_model is None:
            raise ValueError("No trained model to save")
            
        joblib.dump(self.kmeans_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save cluster information
        cluster_info = {
            'cluster_features': self.cluster_features,
            'agg_data': self.agg_data
        }
        joblib.dump(cluster_info, 'cluster_info.joblib')
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='kmeans_model.joblib', scaler_path='scaler.joblib'):
        """
        Load a saved model and scaler
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
            
        Returns:
            tuple: (model, scaler)
        """
        self.kmeans_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load cluster information
        cluster_info = joblib.load('cluster_info.joblib')
        self.cluster_features = cluster_info['cluster_features']
        self.agg_data = cluster_info['agg_data']
        
        print(f"✓ Model loaded from {model_path}")
        return self.kmeans_model, self.scaler
    
    def print_detailed_analysis(self):
        """Print detailed text-based analysis of results"""
        print("\n" + "="*60)
        print("DETAILED CLUSTER ANALYSIS REPORT")
        print("="*60)
        
        # Cluster statistics
        cluster_stats = self.agg_data.groupby(['cluster', 'rush_category']).agg({
            'orders_count': ['mean', 'count'],
            'avg_onshift': 'mean',
            'utilization_rate': 'mean',
            'demand_supply_ratio': 'mean'
        }).round(2)
        
        print("\nCLUSTER STATISTICS:")
        print(cluster_stats)
        
        print("\nPEAK RUSH HOURS:")
        rush_hours = self.agg_data[self.agg_data['rush_category'] == 'Rush']
        if len(rush_hours) > 0:
            peak_hours = rush_hours.groupby('hour_of_day').size().sort_values(ascending=False).head(5)
            for hour, count in peak_hours.items():
                print(f"  Hour {hour}:00 - {count} rush periods")
        

        self._print_business_recommendations()
    
    def _print_business_recommendations(self):
        """Print business recommendations based on clustering results"""
        print("\n" + "="*60)
        print("BUSINESS RECOMMENDATIONS")
        print("="*60)
        
        rush_data = self.agg_data[self.agg_data['rush_category'] == 'Rush']
        moderate_data = self.agg_data[self.agg_data['rush_category'] == 'Moderate']
        non_rush_data = self.agg_data[self.agg_data['rush_category'] == 'Non-rush']
        
        if len(rush_data) > 0:
            avg_rush_orders = rush_data['orders_count'].mean()
            avg_rush_utilization = rush_data['utilization_rate'].mean()
            
            print(f"\n🚨 RUSH PERIODS RECOMMENDATIONS:")
            print(f"   • Average orders: {avg_rush_orders:.1f}")
            print(f"   • Utilization rate: {avg_rush_utilization:.1%}")
            print(f"   • Action: Increase driver allocation by 40-60%")
            
        if len(moderate_data) > 0:
            print(f"\n⚠️  MODERATE PERIODS RECOMMENDATIONS:")
            print(f"   • Maintain standard staffing levels")
            print(f"   • Good opportunity for driver breaks and maintenance")
            
        if len(non_rush_data) > 0:
            print(f"\n✅ NON-RUSH PERIODS RECOMMENDATIONS:")
            print(f"   • Reduce staffing to minimum levels")
            print(f"   • Focus on training and administrative tasks")


def create_delivery_pipeline():

    return DeliveryTimePipeline()

def prepare_data(file_path):

    pipeline = DeliveryTimePipeline()
    return pipeline.prepare_data(file_path)

def train_model(X_scaled, k_range=None):
    pipeline = DeliveryTimePipeline()
    return pipeline.train_model(X_scaled, k_range)

def evaluate_model(model, X_scaled, agg_data):
    pipeline = DeliveryTimePipeline()
    pipeline.kmeans_model = model
    pipeline.agg_data = agg_data
    return pipeline.evaluate_model(X_scaled)

def save_model(model, scaler, model_path='kmeans_model.joblib', scaler_path='scaler.joblib'):
    pipeline = DeliveryTimePipeline()
    pipeline.kmeans_model = model
    pipeline.scaler = scaler
    return pipeline.save_model(model_path, scaler_path)

def load_model(model_path='kmeans_model.joblib', scaler_path='scaler.joblib'):
    pipeline = DeliveryTimePipeline()
    return pipeline.load_model(model_path, scaler_path)

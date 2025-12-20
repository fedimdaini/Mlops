

import argparse
import sys
import os
from model_pipeline import DeliveryTimePipeline

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Delivery Time Analysis Pipeline')
    parser.add_argument('--data_path', type=str, default='cleaned_datadelevry.csv', 
                       help='Path to the data file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'full'], 
                       default='full', help='Execution mode')
    parser.add_argument('--model_path', type=str, default='kmeans_model.joblib',
                       help='Path to save/load the model')
    parser.add_argument('--scaler_path', type=str, default='scaler.joblib',
                       help='Path to save/load the scaler')
    
    args = parser.parse_args()
    
    print("🚀 DELIVERY TIME ANALYSIS PIPELINE")
    print("=" * 50)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file '{args.data_path}' not found")
        sys.exit(1)
    
    try:
        pipeline = DeliveryTimePipeline()
        
        if args.mode in ['train', 'full']:
            # Complete pipeline execution
            print("\n📥 Loading and preparing data...")
            df = pipeline.prepare_data(args.data_path)
            
            print("\n🔧 Preparing features for clustering...")
            X_scaled, features = pipeline.prepare_clustering_features()
            
            print("\n🤖 Training clustering model...")
            model = pipeline.train_model(X_scaled)
            
            print("\n📊 Evaluating model...")
            results = pipeline.evaluate_model(X_scaled)
            
            print("\n💾 Saving model...")
            pipeline.save_model(args.model_path, args.scaler_path)
            
            # Print detailed analysis
            pipeline.print_detailed_analysis()
            
            # Print final results summary
            print("\n🎯 FINAL RESULTS SUMMARY:")
            print(f"Silhouette Score: {results['silhouette_score']:.3f}")
            print(f"Number of Clusters: {model.n_clusters}")
            print(f"Total Time Bins Analyzed: {len(pipeline.agg_data)}")
            print(f"Total Orders: {results['insights']['total_orders']}")
            
        elif args.mode == 'evaluate':
            # Load and evaluate existing model
            print("\n📂 Loading existing model...")
            model, scaler = pipeline.load_model(args.model_path, args.scaler_path)
            
            # Need to prepare data for evaluation
            df = pipeline.prepare_data(args.data_path)
            X_scaled, features = pipeline.prepare_clustering_features()
            
            print("\n📊 Evaluating loaded model...")
            results = pipeline.evaluate_model(X_scaled)
            
            # Print detailed analysis
            pipeline.print_detailed_analysis()
            
            print("\n📋 EVALUATION RESULTS:")
            print(f"Silhouette Score: {results['silhouette_score']:.3f}")
            print(f"Cluster Profiles:\n{results['cluster_profiles']}")
        
        print("\n✅ Pipeline execution completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {str(e)}")
        sys.exit(1)

def demonstrate_modular_functions():
    """
    Demonstrate the modularity by using individual functions
    """
    print("\n" + "="*50)
    print("MODULAR FUNCTIONS DEMONSTRATION")
    print("="*50)
    
    # Example 1: Data preparation only
    print("\n1. Data Preparation Only:")
    pipeline = DeliveryTimePipeline()
    data = pipeline.prepare_data('cleaned_datadelevry.csv')
    print(f"   Prepared data shape: {data.shape}")
    
    # Example 2: Feature preparation only
    print("\n2. Feature Preparation Only:")
    X_scaled, features = pipeline.prepare_clustering_features()
    print(f"   Features prepared: {len(features)} features")
    print(f"   Scaled data shape: {X_scaled.shape}")
    
    # Example 3: Model training only
    print("\n3. Model Training Only:")
    model = pipeline.train_model(X_scaled)
    print(f"   Model trained with {model.n_clusters} clusters")
    
    # Example 4: Model evaluation only
    print("\n4. Model Evaluation Only:")
    results = pipeline.evaluate_model(X_scaled)
    print(f"   Evaluation completed - Silhouette: {results['silhouette_score']:.3f}")
    
    print("\n✓ All modular functions work independently!")

if __name__ == "__main__":
    main()
    


"""
Train model on startup if model file doesn't exist.
This runs automatically when the app starts on Railway.
"""
import os
import sys

def train_if_needed():
    """Train model if it doesn't exist."""
    model_path = "sokoban_diffusion.pth"
    
    if os.path.exists(model_path):
        print(f"✅ Model file '{model_path}' exists. Skipping training.")
        return True
    
    print("⚠️ Model file not found. Starting training...")
    print("This may take 10-20 minutes on Railway...")
    
    try:
        # Check if dataset exists, if not generate it
        dataset_path = "sokoban_dataset.npy"
        if not os.path.exists(dataset_path):
            print("📊 Generating training dataset...")
            from sokoban_data_gen import generate_dataset
            # Generate smaller dataset for faster training on Railway
            generate_dataset(num_episodes=100, output_file=dataset_path)
            print("✅ Dataset generated!")
        
        # Train the model using the main training function
        # Configurable via environment variable: TRAIN_EPOCHS (default: 200 for Railway)
        train_epochs = int(os.environ.get('TRAIN_EPOCHS', '200'))
        use_val = os.environ.get('TRAIN_USE_VALIDATION', 'false').lower() == 'true'
        
        if train_epochs == 500:
            print("🏋️ Training diffusion model (full: 500 epochs)...")
        else:
            print(f"🏋️ Training diffusion model (Railway-optimized: {train_epochs} epochs)...")
        
        from sokoban_diffusion import train_diffusion
        
        # Use the main training function with configurable settings:
        # - Epochs: Set via TRAIN_EPOCHS env var (default: 200 for speed)
        # - Validation: Set via TRAIN_USE_VALIDATION env var (default: false for speed)
        train_diffusion(
            epochs=train_epochs,
            use_validation=use_val,
            dataset_path=dataset_path,
            model_path=model_path
        )
        
        if os.path.exists(model_path):
            print(f"✅ Model trained and saved to '{model_path}'!")
            return True
        else:
            print("❌ Training failed - model file not created.")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 train_on_startup.py: Starting...")
    success = train_if_needed()
    if success:
        print("✅ train_on_startup.py: Completed successfully")
    else:
        print("⚠️ train_on_startup.py: Completed with warnings (app will still start)")
    # Don't exit with error code - let app start regardless
    sys.exit(0)


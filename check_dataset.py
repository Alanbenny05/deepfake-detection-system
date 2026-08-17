import os

def check_dataset():
    print("🔍 Checking dataset structure...")
    print("=" * 60)
    
    base_path = 'dataset'
    total_images = 0
    
    for split in ['train', 'validation', 'test']:
        print(f"\n📁 {split.upper()}:")
        for cls in ['real', 'fake']:
            path = os.path.join(base_path, split, cls)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  - {cls}: {count} images")
                total_images += count
            else:
                print(f"  - {cls}: ❌ NOT FOUND")
    
    print(f"\n✅ Total images: {total_images}")
    
    if total_images < 1000:
        print("⚠️ Warning: You need at least 1000 images for training!")
    else:
        print("🎉 Dataset looks good! Ready for training.")

if __name__ == "__main__":
    check_dataset()
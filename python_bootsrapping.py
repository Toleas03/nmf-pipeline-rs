import os
import random
import shutil

def read_text_files(directory):
    text_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                text_files.append(os.path.join(root, file))
    return text_files

def create_bootstrap_samples(text_files, N_values, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for N in N_values:
        N_dir = os.path.join(output_dir, f"N_{N}")
        if not os.path.exists(N_dir):
            os.makedirs(N_dir)
        
        for i in range(100):  # Create 100 samples for each N
            sample_dir = os.path.join(N_dir, f"sample_{i+1}")
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            
            # Ensure no duplicates by using random.sample
            sample_files = random.sample(text_files, N)
            
            # Copy files and handle potential errors
            copied_files = 0
            for file in sample_files:
                try:
                    # Use unique file names to avoid overwriting
                    dest_file = os.path.join(sample_dir, os.path.basename(file))
                    if os.path.exists(dest_file):
                        dest_file = os.path.join(sample_dir, f"{copied_files}_{os.path.basename(file)}")
                    
                    shutil.copy(file, dest_file)
                    copied_files += 1
                except Exception as e:
                    print(f"Error copying file {file}: {e}")
            
            # Verify the number of files in the sample folder
            actual_files = len(os.listdir(sample_dir))
            if actual_files != N:
                print(f"Warning: Sample {i+1} in N_{N} has {actual_files} files instead of {N}. Retrying...")
                
                # Retry copying missing files
                missing_files = N - actual_files
                retry_files = random.sample(text_files, missing_files)
                for file in retry_files:
                    try:
                        dest_file = os.path.join(sample_dir, os.path.basename(file))
                        if os.path.exists(dest_file):
                            dest_file = os.path.join(sample_dir, f"{copied_files}_{os.path.basename(file)}")
                        
                        shutil.copy(file, dest_file)
                        copied_files += 1
                    except Exception as e:
                        print(f"Error retrying file copy {file}: {e}")

def main():
    dataset_dir = 'Datasets'
    output_dir = 'bootstrap_samples'
    N_values = [100, 250, 500, 750, 1000]  # Different values of N
    
    text_files = read_text_files(dataset_dir)
    print(f"Found {len(text_files)} text files in the dataset.")
    create_bootstrap_samples(text_files, N_values, output_dir)
    print(f"Created bootstrap samples for {N_values} in '{output_dir}' directory.")

if __name__ == "__main__":
    main()
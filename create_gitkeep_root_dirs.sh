#!/bin/bash
# Script to create .gitkeep files in root-owned directories
# Run with: sudo bash create_gitkeep_root_dirs.sh

DIRS=("batch_uploads" "cache" "gateway" "input_videos" "input_videos_1" "output_videos_1" "restored" "result_images" "results" "results_qwen_api")

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        touch "$dir/.gitkeep"
        chown vivek:vivek "$dir/.gitkeep" 2>/dev/null || chown $SUDO_USER:$SUDO_USER "$dir/.gitkeep" 2>/dev/null
        echo "Created $dir/.gitkeep"
    fi
done

echo "Done! Now run: git add */**/.gitkeep && git commit -m 'Add .gitkeep to preserve folder structure'"


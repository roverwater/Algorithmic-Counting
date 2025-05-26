import os
import math
import numpy as np
from PIL import Image, ImageDraw

DATASET_PARAMS = {
    'num_images': 100,              
    'prob_square': 0.33,            
    'prob_triangle': 0.33,          
    'size_range': (1, 3),           
    'offset_range': (-4, 4),        
    'num_patches': 36,              
    'patch_size': 14,               
    'output_size': 224              
}

def create_patches(params):
    """Generate a set of patches with random shapes and positions"""
    patches = []
    square_count = 0
    triangle_count = 0
    
    for _ in range(params['num_patches']):
        img = Image.new('RGB', (params['patch_size'], params['patch_size']), 'black')
        draw = ImageDraw.Draw(img)
        rand = np.random.rand()
        
        if rand < params['prob_square']:
            square_count += 1
            size = np.random.randint(*params['size_range'])
            h_offset = np.random.randint(*params['offset_range'])
            v_offset = np.random.randint(*params['offset_range'])
            
            base = (params['patch_size'] - size*3) // 2
            left = max(0, base + h_offset)
            top = max(0, base + v_offset)
            right = min(params['patch_size'], left + size*3)
            bottom = min(params['patch_size'], top + size*3)
            draw.rectangle([left, top, right, bottom], fill='red')
            
        elif rand < params['prob_square'] + params['prob_triangle']:
            triangle_count += 1
            size = np.random.randint(*params['size_range'])
            h_offset = np.random.randint(*params['offset_range'])
            v_offset = np.random.randint(*params['offset_range'])
            
            base_y = params['patch_size']//2 + size*2 + v_offset
            apex_y = params['patch_size']//2 - size*2 + v_offset
            points = [
                params['patch_size']//2 - size*2 + h_offset, base_y,
                params['patch_size']//2 + size*2 + h_offset, base_y,
                params['patch_size']//2 + h_offset, apex_y
            ]
            draw.polygon(points, fill='red')
            
        patches.append(img)
    
    return patches, square_count, triangle_count

def create_patch_grid(patches, params):
    """Combine patches into a grid and center in output image"""
    grid_size = int(math.sqrt(params['num_patches']))
    grid_image = Image.new('RGB', 
        (params['patch_size'] * grid_size, 
         params['patch_size'] * grid_size))
    
    for i, patch in enumerate(patches):
        x = (i % grid_size) * params['patch_size']
        y = (i // grid_size) * params['patch_size']
        grid_image.paste(patch, (x, y))
    
    final_image = Image.new('RGB', 
        (params['output_size'], params['output_size']), 'black')
    offset = (params['output_size'] - grid_image.width) // 2
    final_image.paste(grid_image, (offset, offset))
    
    return final_image

def generate_dataset():
    """Main function to generate the complete dataset"""
    params = DATASET_PARAMS
    
    folder_name = (
        f"n{params['num_images']}_sq{params['prob_square']:.2f}_tr{params['prob_triangle']:.2f}_"
        f"sz{params['size_range'][0]}-{params['size_range'][1]}_"
        f"off{params['offset_range'][0]}-{params['offset_range'][1]}_"
        f"pat{params['num_patches']}"
    ).replace('.', 'p')
    
    os.makedirs(folder_name, exist_ok=True)
    
    print(f"Generating dataset with {params['num_images']} images...")
    print(f"Output folder: {folder_name}")
    
    for i in range(params['num_images']):
        patches, squares, triangles = create_patches(params)
        
        image = create_patch_grid(patches, params)
        
        filename = f"{i:04d}_squares_{squares}_triangles_{triangles}.png"
        image.save(os.path.join(folder_name, filename))
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    generate_dataset()
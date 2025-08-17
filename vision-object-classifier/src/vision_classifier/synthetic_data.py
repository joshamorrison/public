import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import os
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class SyntheticDirtyDishGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define stain colors (food-based)
        self.stain_colors = [
            (139, 69, 19),   # Brown (chocolate, coffee)
            (255, 140, 0),   # Dark orange (sauce)
            (255, 69, 0),    # Red orange (tomato)
            (128, 128, 0),   # Olive (oil stains)
            (165, 42, 42),   # Brown red (meat juices)
            (255, 215, 0),   # Gold (mustard)
            (220, 20, 60),   # Crimson (ketchup)
            (107, 142, 35),  # Olive drab (vegetables)
        ]
        
        # Dirt and grime colors
        self.dirt_colors = [
            (105, 105, 105), # Dim gray
            (169, 169, 169), # Dark gray
            (128, 128, 128), # Gray
            (139, 129, 76),  # Dark khaki
            (160, 82, 45),   # Saddle brown
        ]
    
    def add_food_stains(self, image: np.ndarray, num_stains: int = None) -> np.ndarray:
        """Add realistic food stains to clean dish image"""
        if num_stains is None:
            num_stains = random.randint(2, 8)
        
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        height, width = image.shape[:2]
        
        for _ in range(num_stains):
            # Random stain properties
            color = random.choice(self.stain_colors)
            alpha = random.randint(30, 120)  # Transparency
            
            # Random position (avoid edges)
            x = random.randint(width//6, 5*width//6)
            y = random.randint(height//6, 5*height//6)
            
            # Random stain size and shape
            stain_type = random.choice(['circle', 'irregular', 'splash'])
            
            if stain_type == 'circle':
                radius = random.randint(10, 40)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                           fill=(*color, alpha))
            
            elif stain_type == 'irregular':
                # Create irregular stain shape
                points = []
                num_points = random.randint(6, 12)
                base_radius = random.randint(15, 35)
                
                for i in range(num_points):
                    angle = (2 * np.pi * i) / num_points
                    r = base_radius + random.randint(-10, 10)
                    px = x + r * np.cos(angle)
                    py = y + r * np.sin(angle)
                    points.append((px, py))
                
                draw.polygon(points, fill=(*color, alpha))
            
            elif stain_type == 'splash':
                # Create splash pattern
                main_radius = random.randint(8, 25)
                draw.ellipse([x-main_radius, y-main_radius, x+main_radius, y+main_radius],
                           fill=(*color, alpha))
                
                # Add smaller droplets around main stain
                for _ in range(random.randint(3, 8)):
                    dx = random.randint(-50, 50)
                    dy = random.randint(-50, 50)
                    small_r = random.randint(2, 8)
                    draw.ellipse([x+dx-small_r, y+dy-small_r, x+dx+small_r, y+dy+small_r],
                               fill=(*color, alpha//2))
        
        # Apply blur to make stains more realistic
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Composite with original image
        image_pil = image_pil.convert('RGBA')
        result = Image.alpha_composite(image_pil, overlay)
        result = result.convert('RGB')
        
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    
    def add_grease_stains(self, image: np.ndarray) -> np.ndarray:
        """Add grease/oil stains with transparency effect"""
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        height, width = image.shape[:2]
        num_grease = random.randint(1, 4)
        
        for _ in range(num_grease):
            x = random.randint(width//4, 3*width//4)
            y = random.randint(height//4, 3*height//4)
            
            # Grease stains are typically larger and more transparent
            radius = random.randint(30, 80)
            alpha = random.randint(20, 60)
            
            # Use darker, more muted colors for grease
            color = random.choice([(139, 129, 76), (128, 128, 0), (105, 105, 105)])
            
            draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                        fill=(*color, alpha))
        
        # Apply stronger blur for grease effect
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
        
        image_pil = image_pil.convert('RGBA')
        result = Image.alpha_composite(image_pil, overlay)
        result = result.convert('RGB')
        
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    
    def add_residue_buildup(self, image: np.ndarray) -> np.ndarray:
        """Add soap residue or water spots"""
        height, width = image.shape[:2]
        
        # Create subtle white/gray spots
        overlay = np.zeros_like(image, dtype=np.float32)
        
        num_spots = random.randint(5, 15)
        for _ in range(num_spots):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            radius = random.randint(3, 12)
            
            # Create circular spots
            y_coords, x_coords = np.ogrid[:height, :width]
            mask = (x_coords - x) ** 2 + (y_coords - y) ** 2 <= radius ** 2
            
            # Add white/light gray residue
            intensity = random.uniform(0.1, 0.3)
            overlay[mask] = [intensity * 255] * 3
        
        # Blend with original image
        result = image.astype(np.float32) + overlay
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def add_scratches_and_wear(self, image: np.ndarray) -> np.ndarray:
        """Add subtle scratches and wear marks"""
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        
        height, width = image.shape[:2]
        num_scratches = random.randint(2, 6)
        
        for _ in range(num_scratches):
            # Random scratch line
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-50, 50)
            y2 = y1 + random.randint(-50, 50)
            
            # Light gray scratches
            color = random.choice([(180, 180, 180), (200, 200, 200), (160, 160, 160)])
            width_scratch = random.randint(1, 2)
            
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width_scratch)
        
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    def generate_dirty_dish(self, clean_image_path: str, output_path: str, 
                           dirty_level: str = 'medium') -> bool:
        """Generate a dirty version of a clean dish image"""
        try:
            # Load clean image
            image = cv2.imread(clean_image_path)
            if image is None:
                return False
            
            # Apply different levels of dirtiness
            if dirty_level == 'light':
                # Light staining
                if random.random() < 0.7:
                    image = self.add_food_stains(image, num_stains=random.randint(1, 3))
                if random.random() < 0.5:
                    image = self.add_residue_buildup(image)
                    
            elif dirty_level == 'medium':
                # Medium staining
                if random.random() < 0.9:
                    image = self.add_food_stains(image, num_stains=random.randint(2, 6))
                if random.random() < 0.6:
                    image = self.add_grease_stains(image)
                if random.random() < 0.4:
                    image = self.add_residue_buildup(image)
                    
            elif dirty_level == 'heavy':
                # Heavy staining
                image = self.add_food_stains(image, num_stains=random.randint(5, 10))
                if random.random() < 0.8:
                    image = self.add_grease_stains(image)
                if random.random() < 0.6:
                    image = self.add_residue_buildup(image)
                if random.random() < 0.3:
                    image = self.add_scratches_and_wear(image)
            
            # Save result
            cv2.imwrite(output_path, image)
            return True
            
        except Exception as e:
            print(f"Error generating dirty dish: {e}")
            return False
    
    def batch_generate(self, clean_images_dir: str, num_variations: int = 3):
        """Generate multiple dirty variations for each clean image"""
        clean_files = [f for f in os.listdir(clean_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        total_generated = 0
        dirty_levels = ['light', 'medium', 'heavy']
        
        for clean_file in clean_files:
            clean_path = os.path.join(clean_images_dir, clean_file)
            
            for i in range(num_variations):
                dirty_level = random.choice(dirty_levels)
                base_name = os.path.splitext(clean_file)[0]
                output_file = f"{base_name}_dirty_{dirty_level}_{i+1}.jpg"
                output_path = os.path.join(self.output_dir, output_file)
                
                if self.generate_dirty_dish(clean_path, output_path, dirty_level):
                    total_generated += 1
        
        print(f"Generated {total_generated} synthetic dirty dish images")
        return total_generated


if __name__ == "__main__":
    # Example usage
    generator = SyntheticDirtyDishGenerator("../data/synthetic")
    
    # Generate synthetic data from clean images
    clean_dir = "../data/clean"
    if os.path.exists(clean_dir):
        generator.batch_generate(clean_dir, num_variations=5)
    else:
        print(f"Clean images directory {clean_dir} not found")
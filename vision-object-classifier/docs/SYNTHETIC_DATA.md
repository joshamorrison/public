# Synthetic Data Generation Pipeline

Complete guide to generating realistic dirty dish images from clean dish photos using advanced computer vision techniques.

## Overview

The Synthetic Data Generation Pipeline addresses the critical challenge of limited training data by creating realistic dirty dish variants from clean images. This approach enables training production-ready models without extensive manual data collection or labeling.

## Core Technology

### SyntheticDirtyDishGenerator Class
```python
from src.vision_classifier.synthetic_data import SyntheticDirtyDishGenerator

# Initialize generator
generator = SyntheticDirtyDishGenerator(output_dir='data/dirty')

# Generate single dirty variant
generator.generate_dirty_dish(
    clean_image_path='data/clean/plate_01.jpg',
    output_path='data/dirty/plate_01_dirty.jpg',
    dirty_level='medium'
)

# Batch generation with multiple variations
total_generated = generator.batch_generate(
    clean_images_dir='data/clean/',
    num_variations=5  # Creates 5 dirty variants per clean image
)
```

## Stain Generation Techniques

### 1. Food Stain Simulation
**Realistic organic residue patterns**

```python
def generate_food_stain(self, image, intensity='medium'):
    """Generate realistic food stain patterns."""
    stain_colors = [
        (139, 69, 19),   # Chocolate/sauce
        (255, 140, 0),   # Orange/tomato
        (128, 128, 0),   # Olive/mustard
        (255, 255, 0),   # Butter/oil
        (165, 42, 42)    # Red sauce
    ]
    
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    for _ in range(self._get_stain_count(intensity)):
        # Random stain properties
        color = random.choice(stain_colors)
        alpha = self._get_alpha_for_intensity(intensity)
        position = self._get_random_position(image.size)
        size = self._get_stain_size(intensity)
        
        # Draw irregular stain shape
        stain_shape = self._generate_irregular_shape(position, size)
        draw.polygon(stain_shape, fill=(*color, alpha))
    
    return Image.alpha_composite(image.convert('RGBA'), overlay)
```

### 2. Grease Mark Generation
**Oil and fat stain simulation with transparency effects**

```python
def generate_grease_marks(self, image, intensity='medium'):
    """Generate realistic grease and oil stains."""
    grease_overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    
    # Create grease effect using Gaussian blur
    for _ in range(self._get_grease_count(intensity)):
        # Create base grease spot
        spot_size = random.randint(20, 80)
        spot_color = (255, 255, 150, random.randint(30, 80))
        
        # Generate organic grease shape
        grease_mask = self._create_grease_mask(spot_size)
        
        # Apply blur for realistic oil spread
        grease_mask = grease_mask.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Composite onto image
        position = self._get_random_position(image.size, spot_size)
        grease_overlay.paste(grease_mask, position, grease_mask)
    
    return Image.alpha_composite(image.convert('RGBA'), grease_overlay)
```

### 3. Water Spot Simulation
**Soap residue and mineral deposits**

```python
def generate_water_spots(self, image, intensity='light'):
    """Generate water spots and soap residue."""
    spots_overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(spots_overlay)
    
    for _ in range(self._get_spot_count(intensity)):
        # Water spot characteristics
        center = self._get_random_position(image.size)
        radius = random.randint(5, 25)
        
        # Create concentric circles for water spot effect
        for ring in range(3):
            ring_radius = radius - (ring * 3)
            if ring_radius > 0:
                alpha = 20 + (ring * 10)
                color = (200, 200, 255, alpha)  # Slightly blue tint
                
                bbox = [
                    center[0] - ring_radius, center[1] - ring_radius,
                    center[0] + ring_radius, center[1] + ring_radius
                ]
                draw.ellipse(bbox, fill=color)
    
    return Image.alpha_composite(image.convert('RGBA'), spots_overlay)
```

### 4. Wear Pattern Generation
**Subtle scratches and usage marks**

```python
def generate_wear_patterns(self, image, intensity='light'):
    """Generate subtle wear patterns and scratches."""
    wear_overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(wear_overlay)
    
    for _ in range(self._get_wear_count(intensity)):
        # Random scratch parameters
        start_point = self._get_random_position(image.size)
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(10, 50)
        
        # Calculate end point
        end_point = (
            start_point[0] + int(length * math.cos(angle)),
            start_point[1] + int(length * math.sin(angle))
        )
        
        # Draw subtle scratch
        scratch_color = (128, 128, 128, 25)  # Very subtle gray
        draw.line([start_point, end_point], fill=scratch_color, width=1)
    
    return Image.alpha_composite(image.convert('RGBA'), wear_overlay)
```

## Dirtiness Intensity Levels

### Light Contamination
**Minimal, realistic everyday use**

```python
LIGHT_CONTAMINATION = {
    'food_stains': 1-2,
    'grease_marks': 0-1,
    'water_spots': 2-4,
    'wear_patterns': 1-3,
    'alpha_range': (20, 40),
    'size_multiplier': 0.5
}
```

**Characteristics:**
- Few small stains or spots
- High transparency (subtle appearance)
- Minimal color deviation from original
- Realistic for recently used dishes

### Medium Contamination  
**Moderate use with visible residue**

```python
MEDIUM_CONTAMINATION = {
    'food_stains': 2-4,
    'grease_marks': 1-3,
    'water_spots': 3-6,
    'wear_patterns': 2-5,
    'alpha_range': (40, 80),
    'size_multiplier': 1.0
}
```

**Characteristics:**
- Multiple visible stains and marks
- Moderate transparency (clearly visible)
- Noticeable but not overwhelming
- Typical after normal meal use

### Heavy Contamination
**Significant buildup requiring thorough cleaning**

```python
HEAVY_CONTAMINATION = {
    'food_stains': 4-8,
    'grease_marks': 3-6,
    'water_spots': 5-10,
    'wear_patterns': 3-8,
    'alpha_range': (60, 120),
    'size_multiplier': 1.5
}
```

**Characteristics:**
- Numerous overlapping stains
- Higher opacity (obvious contamination)
- Multiple stain types present
- Realistic for dishes needing deep cleaning

## Advanced Techniques

### Realistic Shape Generation
```python
def _generate_irregular_shape(self, center, base_radius):
    """Generate organic, irregular stain shapes."""
    points = []
    num_points = random.randint(8, 16)
    
    for i in range(num_points):
        angle = (2 * math.pi * i) / num_points
        
        # Add randomness to radius for irregular shape
        radius_variation = random.uniform(0.7, 1.3)
        radius = base_radius * radius_variation
        
        # Calculate point
        x = center[0] + int(radius * math.cos(angle))
        y = center[1] + int(radius * math.sin(angle))
        points.append((x, y))
    
    return points
```

### Lighting-Aware Stain Placement
```python
def _adjust_for_lighting(self, image, stain_position, stain_intensity):
    """Adjust stain appearance based on image lighting."""
    # Analyze local brightness
    region = image.crop((
        max(0, stain_position[0] - 20),
        max(0, stain_position[1] - 20),
        min(image.width, stain_position[0] + 20),
        min(image.height, stain_position[1] + 20)
    ))
    
    avg_brightness = ImageStat.Stat(region).mean[0]
    
    # Adjust stain properties based on brightness
    if avg_brightness > 200:  # Bright area
        stain_intensity *= 1.2  # More visible stains
    elif avg_brightness < 100:  # Dark area
        stain_intensity *= 0.8  # Subtler stains
    
    return stain_intensity
```

### Texture-Based Stain Distribution
```python
def _analyze_surface_texture(self, image):
    """Analyze dish surface to place stains realistically."""
    # Convert to grayscale for edge detection
    gray = image.convert('L')
    
    # Apply edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # Identify flat surfaces (less edges) for stain placement
    flat_regions = []
    edge_array = np.array(edges)
    
    # Scan for low-edge density regions
    window_size = 30
    for y in range(0, edge_array.shape[0] - window_size, window_size):
        for x in range(0, edge_array.shape[1] - window_size, window_size):
            window = edge_array[y:y+window_size, x:x+window_size]
            edge_density = np.mean(window)
            
            if edge_density < 50:  # Low edge density = flat surface
                flat_regions.append((x + window_size//2, y + window_size//2))
    
    return flat_regions
```

## Batch Processing Pipeline

### Automated Batch Generation
```python
def batch_generate(self, clean_images_dir, num_variations=5, 
                  intensity_distribution=None):
    """Generate multiple dirty variants for all clean images."""
    
    if intensity_distribution is None:
        intensity_distribution = {
            'light': 0.3,
            'medium': 0.5,
            'heavy': 0.2
        }
    
    clean_images = self._get_image_files(clean_images_dir)
    total_generated = 0
    
    for clean_image_path in clean_images:
        base_name = os.path.splitext(os.path.basename(clean_image_path))[0]
        
        for i in range(num_variations):
            # Select intensity based on distribution
            intensity = self._select_intensity(intensity_distribution)
            
            # Generate output filename
            output_filename = f"{base_name}_dirty_{intensity}_{i+1}.jpg"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Generate dirty variant
            success = self.generate_dirty_dish(
                clean_image_path=clean_image_path,
                output_path=output_path,
                dirty_level=intensity
            )
            
            if success:
                total_generated += 1
                print(f"Generated: {output_filename}")
    
    return total_generated
```

### Quality Control Pipeline
```python
def validate_generated_images(self, output_dir):
    """Validate quality of generated synthetic images."""
    validation_results = {
        'total_images': 0,
        'valid_images': 0,
        'failed_images': [],
        'quality_scores': []
    }
    
    for image_file in os.listdir(output_dir):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        image_path = os.path.join(output_dir, image_file)
        validation_results['total_images'] += 1
        
        try:
            # Basic image validation
            with Image.open(image_path) as img:
                # Check image properties
                if img.size[0] < 100 or img.size[1] < 100:
                    validation_results['failed_images'].append(
                        (image_file, "Image too small")
                    )
                    continue
                
                # Check image quality (not completely black/white)
                img_array = np.array(img.convert('L'))
                brightness_std = np.std(img_array)
                
                if brightness_std < 10:  # Too uniform
                    validation_results['failed_images'].append(
                        (image_file, "Low detail/contrast")
                    )
                    continue
                
                # Image passed validation
                validation_results['valid_images'] += 1
                validation_results['quality_scores'].append(brightness_std)
                
        except Exception as e:
            validation_results['failed_images'].append(
                (image_file, f"Error: {str(e)}")
            )
    
    return validation_results
```

## Performance Optimization

### Memory-Efficient Processing
```python
def process_large_batch(self, clean_images_dir, batch_size=10):
    """Process large image sets with memory management."""
    clean_images = self._get_image_files(clean_images_dir)
    
    for i in range(0, len(clean_images), batch_size):
        batch = clean_images[i:i+batch_size]
        
        # Process batch
        for image_path in batch:
            self._process_single_image(image_path)
        
        # Memory cleanup
        gc.collect()
        
        print(f"Processed batch {i//batch_size + 1}/{(len(clean_images)-1)//batch_size + 1}")
```

### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

def parallel_batch_generate(self, clean_images_dir, num_variations=5, 
                           max_workers=None):
    """Generate dirty images using parallel processing."""
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)
    
    clean_images = self._get_image_files(clean_images_dir)
    tasks = []
    
    # Create task list
    for clean_image_path in clean_images:
        for i in range(num_variations):
            intensity = random.choice(['light', 'medium', 'heavy'])
            tasks.append((clean_image_path, intensity, i))
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(self._generate_single_variant, task)
            for task in tasks
        ]
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                print(f"Task failed: {e}")
    
    return len([r for r in results if r])
```

## Usage Examples

### Basic Usage
```python
# Initialize generator
generator = SyntheticDirtyDishGenerator('data/synthetic_dirty')

# Generate a few dirty variants
generator.generate_dirty_dish(
    'data/clean/plate_01.jpg',
    'data/dirty/plate_01_dirty_medium.jpg',
    'medium'
)
```

### Production Workflow
```python
# Production-scale synthetic data generation
generator = SyntheticDirtyDishGenerator('data/production_dirty')

# Generate balanced dataset
total_clean = len(os.listdir('data/clean'))
target_dirty = total_clean * 2  # 1:2 ratio

variations_per_image = target_dirty // total_clean

# Batch generate with quality control
generated_count = generator.batch_generate(
    clean_images_dir='data/clean',
    num_variations=variations_per_image,
    intensity_distribution={
        'light': 0.4,
        'medium': 0.4,
        'heavy': 0.2
    }
)

# Validate results
validation = generator.validate_generated_images('data/production_dirty')
print(f"Generated: {generated_count}")
print(f"Valid: {validation['valid_images']}/{validation['total_images']}")
```

### Custom Stain Pipeline
```python
# Create custom stain combination
def custom_dirty_generation(self, image, custom_config):
    """Generate dirty image with custom stain configuration."""
    result_image = image.copy()
    
    if custom_config.get('food_stains', False):
        result_image = self.generate_food_stain(
            result_image, 
            custom_config['food_intensity']
        )
    
    if custom_config.get('grease_marks', False):
        result_image = self.generate_grease_marks(
            result_image,
            custom_config['grease_intensity']
        )
    
    if custom_config.get('water_spots', False):
        result_image = self.generate_water_spots(
            result_image,
            custom_config['water_intensity']
        )
    
    return result_image

# Usage
custom_config = {
    'food_stains': True,
    'food_intensity': 'heavy',
    'grease_marks': True,
    'grease_intensity': 'medium',
    'water_spots': False
}

dirty_image = generator.custom_dirty_generation(clean_image, custom_config)
```

## Dataset Statistics

### Synthetic vs Real Data Comparison

| **Metric** | **Synthetic Data** | **Real Data** | **Combined** |
|------------|-------------------|---------------|--------------|
| **Generation Speed** | 100 images/minute | Manual collection | Varies |
| **Cost** | $0 | $5-20 per image | Mixed |
| **Consistency** | High | Variable | Balanced |
| **Variety** | Controlled | Natural | Optimal |
| **Scalability** | Unlimited | Limited | Excellent |

### Quality Metrics
```python
def calculate_dataset_quality_metrics(synthetic_dir, real_dir=None):
    """Calculate comprehensive quality metrics for dataset."""
    metrics = {
        'synthetic_stats': analyze_image_statistics(synthetic_dir),
        'diversity_score': calculate_diversity_score(synthetic_dir),
        'realism_score': calculate_realism_score(synthetic_dir)
    }
    
    if real_dir:
        metrics['real_stats'] = analyze_image_statistics(real_dir)
        metrics['distribution_similarity'] = compare_distributions(
            synthetic_dir, real_dir
        )
    
    return metrics
```

This comprehensive synthetic data generation pipeline enables the creation of production-quality training datasets without manual data collection, providing full control over data quality, quantity, and distribution while maintaining realistic appearance for effective model training.
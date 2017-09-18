import itk
import pyopenvdb as vdb
import numpy as np
import sys
import os

def smooth_step(edge0, edge1, x):
    # Implement smooth step function similar to the one used in C++
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def create_transfer_function():
    func_length = 100
    transfer_func = []
    
    # First section (1/5 of the function) - dark to red
    start_rgb = np.array([0.02, 0.02, 0.02])
    end_rgb = np.array([1.0, 0.02, 0.02])
    
    for i in range(int(func_length * 1/5)):
        t = i / func_length
        color = start_rgb + t * (end_rgb - start_rgb)
        transfer_func.append(color)  # RGB only, no alpha
    
    # Second section (4/5 of the function) - red to blue
    start_rgb = end_rgb
    end_rgb = np.array([0.0, 0.02, 1.0])
    
    for i in range(int(func_length * 4/5)):
        t = i / func_length
        color = start_rgb + t * (end_rgb - start_rgb)
        transfer_func.append(color)  # RGB only, no alpha
    
    return np.array(transfer_func)

def convert_mhd_to_vdb(input_path, output_path):
    # Read the MHD file
    ImageType = itk.Image[itk.F, 3]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(input_path)
    reader.Update()
    
    # Convert to numpy array and fix orientation
    image = itk.array_from_image(reader.GetOutput())
    
    # Get min/max values
    data_min = np.min(image)
    data_max = np.max(image)
    
    # Normalize and apply smooth step
    normalized = (image - data_min) / (data_max - data_min)
    smooth_step_edges = (0.2, 0.6)
    density_values = smooth_step(smooth_step_edges[0], smooth_step_edges[1], normalized)
    
    # Create density grid
    density_grid = vdb.FloatGrid()
    density_grid.name = 'density'
    density_grid.copyFromArray(density_values)
    density_grid.gridClass = vdb.GridClass.FOG_VOLUME
    
    # Create simple red albedo that follows density
    albedo_values = np.zeros(density_values.shape + (3,))  # RGB array
    albedo_values[..., 0] = density_values  # Red channel = density
    # Green and Blue channels remain 0
    
    # Create Vec3 grid for albedo
    albedo_grid = vdb.Vec3SGrid()
    albedo_grid.name = 'albedo'
    albedo_grid.copyFromArray(albedo_values)
    albedo_grid.gridClass = vdb.GridClass.FOG_VOLUME
    
    # Write all grids to the VDB file
    vdb.write(output_path, [density_grid, albedo_grid])
    
    print(f"Converted {input_path} to {output_path}")
    print(f"Created grids: density, albedo")

def main():
    if len(sys.argv) != 3:
        print("Usage: python mhd_to_vdb.py input.mhd output.vdb")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
        
    convert_mhd_to_vdb(input_path, output_path)

if __name__ == "__main__":
    main() 
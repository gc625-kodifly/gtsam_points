# -*- coding: utf-8 -*-
import numpy as np
import math

def matrix_to_rpy(transformation_matrix):
    """
    Convert a 4x4 transformation matrix to roll-pitch-yaw angles (in radians)
    
    Args:
        transformation_matrix: 4x4 numpy array
        
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Extract the 3x3 rotation matrix from the transformation matrix
    R = transformation_matrix[:3, :3]
    
    # Extract roll, pitch, yaw from rotation matrix
    # Using the ZYX convention (yaw around Z, pitch around Y, roll around X)
    
    # Pitch (around Y-axis)
    pitch = math.asin(-R[2, 0])
    
    # Check for gimbal lock (pitch = ±90 degrees)
    if abs(pitch) > math.pi/2 - 1e-6:
        # Gimbal lock case
        yaw = math.atan2(R[1, 2], R[0, 2])
        roll = 0.0
    else:
        # Normal case
        yaw = math.atan2(R[1, 0], R[0, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
    
    return roll, pitch, yaw

def radians_to_degrees(radians):
    """Convert radians to degrees"""
    return radians * 180.0 / math.pi

# Read the transformation matrix from the file
matrix_data = []
with open('data/chai_wan_light_public_housing/000000/2025-06-09-17-00-00_georef.txt', 'r') as f:
    for line in f:
        row = [float(x) for x in line.strip().split()]
        matrix_data.append(row)

# Convert to numpy array
transformation_matrix = np.array(matrix_data)

print("Transformation Matrix:")
print(transformation_matrix)
print()

# Convert to RPY angles
roll, pitch, yaw = matrix_to_rpy(transformation_matrix)

print("Roll-Pitch-Yaw Angles:")
print(f"Roll:  {roll:.6f} radians ({radians_to_degrees(roll):.6f} degrees)")
print(f"Pitch: {pitch:.6f} radians ({radians_to_degrees(pitch):.6f} degrees)")
print(f"Yaw:   {yaw:.6f} radians ({radians_to_degrees(yaw):.6f} degrees)")

# Also extract translation
translation = transformation_matrix[:3, 3]
print(f"\nTranslation:")
print(f"X: {translation[0]:.6f}")
print(f"Y: {translation[1]:.6f}")
print(f"Z: {translation[2]:.6f}") 
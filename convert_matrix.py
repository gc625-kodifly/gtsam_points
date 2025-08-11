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

# The transformation matrix you provided
matrix_data = [
    [6.694186443433113309e-01, -7.425830336019567657e-01, 2.119237627257075304e-02, 8.429600434498596005e+05],
    [7.417248882661189313e-01, 6.696948124470959440e-01, 3.678380496987999210e-02, 8.143905135895964922e+05],
    [-4.150745393512378367e-02, -8.904851933859736154e-03, 9.990985110988103157e-01, 7.000000000000000000e+00],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
]

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
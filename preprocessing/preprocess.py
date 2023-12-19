import numpy as np
import cv2

def convert_color_band_to_grayscale(src):
  #Gray scale
  blue_channel, green_channel, red_channel = cv2.split(src)
  # Convert each color channel to grayscale
  blue_grayscale = cv2.cvtColor(blue_channel, cv2.COLOR_GRAY2BGR)
  green_grayscale = cv2.cvtColor(green_channel, cv2.COLOR_GRAY2BGR)
  red_grayscale = cv2.cvtColor(red_channel, cv2.COLOR_GRAY2BGR)
  
  return blue_grayscale, green_grayscale, red_grayscale

def binary_hair_mask(grayScale):
#     Horizontal Element Structure
  structuring_element_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
  structuring_element_horizontal[:, 0] = 0
  structuring_element_horizontal[:, -1] = 0

#     Diagonal Element Structure
  structuring_element_diagonal = np.zeros((9, 9), dtype=np.uint8)
  np.fill_diagonal(structuring_element_diagonal, 1)
  structuring_element_diagonal[0][0] = 0
  structuring_element_diagonal[-1][-1] = 0
  
#     Vertical Element Structure
  structuring_element_vertical = np.transpose(structuring_element_horizontal)
  
#     Closing (Horizontal)
  closing_horizontal = cv2.morphologyEx(grayScale, cv2.MORPH_CLOSE, structuring_element_horizontal)
  
#     Closing (Diagonal)
  closing_diagonal = cv2.morphologyEx(grayScale, cv2.MORPH_CLOSE, structuring_element_diagonal)
  
#     CLosing (Vertical)
  closing_vertical = cv2.morphologyEx(grayScale, cv2.MORPH_CLOSE, structuring_element_vertical)
  
#     Maximum Pixelwise Matrix
  max_matrix = np.maximum.reduce([closing_horizontal, closing_diagonal, closing_vertical])
  #Black hat filter
  blackhat = cv2.absdiff(grayScale, max_matrix)
#     print("absDiff: ", cv2.absdiff(grayScale, max_matrix))
#     print("abs: ", np.abs(grayScale - max_matrix))

  binary_hair_mask = np.where(blackhat > 24, 1, 0)
#     ret,binary_hair_mask = cv2.threshold(bhg,127,1,cv2.THRESH_BINARY_INV)
  return binary_hair_mask

def final_binary_hair_mask(mask_blue, mask_green, mask_red):
  return np.bitwise_or.reduce([mask_blue, mask_green, mask_red])

def binary_dilation(image):
  # Create a structuring element
  kernel = np.ones((5, 5), dtype=np.uint8)

  # Perform binary dilation
  dilated_image = cv2.dilate(image, kernel, iterations=1)

  return dilated_image

def hair_structure_interpolation(original_image, hair_mask):
  height, width = hair_mask.shape[:2]
  interpolated_image = np.copy(original_image)

  for y in range(height):
      for x in range(width):
          if hair_mask[y, x] == 1:
              interpolated_image[y, x] = interpolate_img(original_image, hair_mask, y, x)

  return interpolated_image

def interpolate_img(original_image, hair_mask, y, x):
  # Check if the pixel is located within a hair structure in the binary hair mask
  # The criteria is based on the assumption that hair structures are connected and have a certain minimum size
  # Return True if the pixel is within a hair structure, False otherwise
  vertical_length = len(hair_mask)
  horizontal_length = len(hair_mask[0])
  
  # Check if the pixel is within the hair region
  if hair_mask[y, x] == 0:
      return original_image[y][x]

  # Get the dimensions of the matrix
  num_rows, num_cols = hair_mask.shape

  # Initialize a list to store dictionaries
  element_count = {
      'count_horizontal_1': 0,
      'count_vertical_1': 0,
      'count_diagonal_1_main': 0,
      'count_diagonal_1_flip': 0
  }
  # Count horizontally (in the current row)
  focus_point_horizontal = find_focus_point(hair_mask, y, x, direction='horizontal')
  element_count['count_horizontal_1'], left_h, right_h = find_consecutive_lengths(hair_mask[y], x)

  # Count vertically (in the current column)
  focus_point_vertical = find_focus_point(hair_mask, y, x, direction='vertical')
  element_count['count_vertical_1'], left_v, right_v = find_consecutive_lengths(hair_mask[:, x], y)

  # Count in the diagonal that traverses the matrix
  diagonal_traverse_values = np.diagonal(hair_mask, offset=x - y)
  focus_point_diag = find_focus_point(hair_mask, y, x, direction='diagonal', offset = x-y)
  element_count['count_diagonal_1_main'], left_d, right_d = find_consecutive_lengths(diagonal_traverse_values, focus_point_diag)

  diagonal_traverse_values_flip = np.fliplr(hair_mask).diagonal(offset=len(hair_mask[0])-x-y-1)
  focus_point_diag_flip = find_focus_point(hair_mask, y, x, direction='diagonal_flip', offset = len(hair_mask[0])-x-y-1)
  element_count['count_diagonal_1_flip'], left_df, right_df = find_consecutive_lengths(diagonal_traverse_values_flip, focus_point_diag_flip)
  
  # Get the counts from the dictionary
  count_horizontal = element_count['count_horizontal_1']
  count_vertical = element_count['count_vertical_1']
  count_diagonal_main = element_count['count_diagonal_1_main']
  count_diagonal_flip = element_count['count_diagonal_1_flip']

  # Create a dictionary with directions and their corresponding counts
  counts = {
      'horizontal': count_horizontal,
      'vertical': count_vertical,
      'diagonal': count_diagonal_main,
      'diagonal_flip': count_diagonal_flip
  }

  # Find the direction with the highest count using the max function
  longest_line_direction = max(counts, key=counts.get)
  shortest_line_direction = min(counts, key=counts.get)
  
  longest_line_value = counts[longest_line_direction]
#     shortest_line_value = counts[shortest_line_direction]

  # Create a list of directions
  directions = ['horizontal', 'vertical', 'diagonal', 'diagonal_flip']

  # Remove the direction with the highest count
  directions.remove(longest_line_direction)
  
  # Check if the highest length of direction has more than 50 pixels
  if longest_line_value <= 50:
      return original_image[y][x]
  
  if counts[directions[0]] >= 10 or counts[directions[1]] >= 10 or counts[directions[2]] >= 10:
      return original_image[y][x]
  
  # Take the value 11 pixels alongside from each side of hair border alongside the shortest line direction
  # Horizontal
  interpolate_result = 0
  if shortest_line_direction == 'horizontal':
      first_non_hair_pixel_idx = x - left_h - 11
      
      if first_non_hair_pixel_idx < 0:
          first_non_hair_pixel_idx = 0
      
      second_non_hair_pixel_idx = x + right_h + 11
      
      if second_non_hair_pixel_idx >= horizontal_length:
          second_non_hair_pixel_idx = horizontal_length-1
      
      first_non_hair_pixel = original_image[y][first_non_hair_pixel_idx]
      second_non_hair_pixel = original_image[y][second_non_hair_pixel_idx]
      
      interpolate_result = interpolate_pixel(original_image, 
                                              first_non_hair_pixel, 
                                              second_non_hair_pixel,
                                              (y, x),
                                              (y, first_non_hair_pixel_idx), 
                                              (y, second_non_hair_pixel_idx))
  
  # Vertical
  if shortest_line_direction == 'vertical':
      first_non_hair_pixel_idx = y - left_v - 11
      
      if first_non_hair_pixel_idx < 0:
          first_non_hair_pixel_idx = 0
      
      second_non_hair_pixel_idx = y + right_v + 11
      
      if second_non_hair_pixel_idx >= vertical_length:
          second_non_hair_pixel_idx = vertical_length-1
      
      first_non_hair_pixel = original_image[first_non_hair_pixel_idx][x]
      second_non_hair_pixel = original_image[second_non_hair_pixel_idx][x]
      
      interpolate_result = interpolate_pixel(original_image, 
                                              first_non_hair_pixel, 
                                              second_non_hair_pixel,
                                              (y, x),
                                              (first_non_hair_pixel_idx, x), 
                                              (second_non_hair_pixel_idx, x))
  
  # Diagonal
  if shortest_line_direction == 'diagonal':
      first_non_hair_pixel_idx_H = x - left_d - 11
      first_non_hair_pixel_idx_V = y - left_d - 11
      
      if first_non_hair_pixel_idx_H < 0:
          first_non_hair_pixel_idx_H = 0
          
      if first_non_hair_pixel_idx_V < 0:
          first_non_hair_pixel_idx_V = 0
      
      second_non_hair_pixel_idx_H = x + right_d + 11
      second_non_hair_pixel_idx_V = y + right_d + 11
      
      if second_non_hair_pixel_idx_H >= horizontal_length:
          second_non_hair_pixel_idx_H = horizontal_length-1
          
      if second_non_hair_pixel_idx_V >= vertical_length:
          second_non_hair_pixel_idx_V = vertical_length-1
      
      first_non_hair_pixel = original_image[first_non_hair_pixel_idx_V][first_non_hair_pixel_idx_H]
      second_non_hair_pixel = original_image[second_non_hair_pixel_idx_V][second_non_hair_pixel_idx_H]
      
      interpolate_result = interpolate_pixel(original_image, 
                                              first_non_hair_pixel, 
                                              second_non_hair_pixel,
                                              (y, x),
                                              (first_non_hair_pixel_idx_V, first_non_hair_pixel_idx_H), 
                                              (second_non_hair_pixel_idx_V, second_non_hair_pixel_idx_H))
  # Diagonal Flip
  if shortest_line_direction == 'diagonal_flip':
      first_non_hair_pixel_idx_H = x - left_df - 11
      first_non_hair_pixel_idx_V = y - left_df - 11
      
      if first_non_hair_pixel_idx_H < 0:
          first_non_hair_pixel_idx_H = 0
          
      if first_non_hair_pixel_idx_V < 0:
          first_non_hair_pixel_idx_V = 0
      
      second_non_hair_pixel_idx_H = x + right_df + 11
      second_non_hair_pixel_idx_V = y + right_df + 11
      
      if second_non_hair_pixel_idx_H >= horizontal_length:
          second_non_hair_pixel_idx_H = horizontal_length-1
          
      if second_non_hair_pixel_idx_V >= vertical_length:
          second_non_hair_pixel_idx_V = vertical_length-1
      
      first_non_hair_pixel = original_image[first_non_hair_pixel_idx_V][first_non_hair_pixel_idx_H]
      second_non_hair_pixel = original_image[second_non_hair_pixel_idx_V][second_non_hair_pixel_idx_H]
      
      interpolate_result = interpolate_pixel(original_image, 
                                              first_non_hair_pixel, 
                                              second_non_hair_pixel,
                                              (y, x),
                                              (first_non_hair_pixel_idx_V, first_non_hair_pixel_idx_H), 
                                              (second_non_hair_pixel_idx_V, second_non_hair_pixel_idx_H))
  
  return interpolate_result

def find_focus_point(matrix,y, x, direction, offset=0):
  focus_point = 0
  if direction == 'diagonal':
      temp=np.copy(matrix)
      temp[y][x]*=2
      temp_arr=np.diagonal(temp, offset=offset)
      focus_point = np.where(temp_arr==2)[0][0]
  
  elif direction == 'diagonal_flip':
      temp=np.copy(matrix)
      temp[y][x]*=2
      temp_arr=np.fliplr(temp).diagonal(offset=offset)
      focus_point = np.where(temp_arr==2)[0][0]
  
  elif direction == 'horizontal':
      focus_point = x
  
  else:
      focus_point = y
      
  return focus_point

def find_consecutive_lengths(matrix, focus_point):
  row = matrix
  left_part = row[:focus_point]
  right_part = row[focus_point:]
  reverse_left_part = left_part[::-1]
  left_length, right_length = 0, 0
  
  if len(reverse_left_part) > 0 and len(np.where(reverse_left_part == 0)[0]) > 0:
      left_length = np.where(reverse_left_part == 0)[0][0]
  else:
      left_length = len(left_part)
  
  if len(right_part) > 0 and len(np.where(right_part == 0)[0]) > 0:
      right_length = np.where(right_part == 0)[0][0]
  else:
      right_length = len(right_part)

  return left_length + right_length, left_length, right_length

def interpolate_pixel(original_image, first_pixel, second_pixel, current_pixel_coordinate, first_pixel_coordinate, second_pixel_coordinate):
  # Interpolate the pixel value in the original image using nearby non-hair pixel values
  # Bilinear interpolation is used to estimate the pixel value based on the neighboring non-hair pixels

  intensity_pixel = second_pixel * (distance_pixels(current_pixel_coordinate, first_pixel_coordinate)/distance_pixels(first_pixel_coordinate, second_pixel_coordinate)) +first_pixel *(distance_pixels(current_pixel_coordinate, second_pixel_coordinate)/distance_pixels(first_pixel_coordinate, second_pixel_coordinate))

  return intensity_pixel

def distance_pixels(coor_1, coor_2):
  b, a = coor_1
  d, c = coor_2
  
  return np.sqrt((c-a)**2 + (d-b)**2)

def remove_hair(src):
  blue, green, red = convert_color_band_to_grayscale(src)
  mask_blue = binary_hair_mask(blue)
  mask_green = binary_hair_mask(green)
  mask_red = binary_hair_mask(red)
  mask = final_binary_hair_mask(mask_blue, mask_green, mask_red)
  mask = mask.astype(np.uint8)
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  dst = hair_structure_interpolation(src, mask)
  binary_dilated = binary_dilation(dst)
  result = cv2.medianBlur(binary_dilated, 5)
#     print(mask_final)
#     print(mask.shape)
#     print(mask.dtype)
    
  return result

def shade_of_gray_cc(img, power=6, gamma=None):
  """
  img (numpy array): the original image with format of (h, w, c)
  power (int): the degree of norm, 6 is used in reference paper
  gamma (float): the value of gamma correction, 2.2 is used in reference paper
  """
  img_dtype = img.dtype

  if gamma is not None:
      img = img.astype('uint8')
      look_up_table = np.ones((256,1), dtype='uint8') * 0
      for i in range(256):
          look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
      img = cv2.LUT(img, look_up_table)

  img = img.astype('float32')
  img_power = np.power(img, power)
  rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
  rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
  rgb_vec = rgb_vec/rgb_norm
  rgb_vec = 1/(rgb_vec*np.sqrt(3))
  img = np.multiply(img, rgb_vec)

  # Andrew Anikin suggestion
  img = np.clip(img, a_min=0, a_max=255)
  
  return img.astype(img_dtype)
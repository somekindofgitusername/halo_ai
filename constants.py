# Define feature and target columns once
# natural
#FEATURE_COLUMNS = ['orient_w', 'orient_x', 'orient_y', 'orient_z', 'eyedir_x', 'eyedir_y', 'eyedir_z', 'azimuth', 'altitude']

#FEATURE_COLUMNS = ['eyedir_x', 'eyedir_y', 'eyedir_z', 'azimuth', 'altitude']

FEATURE_COLUMNS = ['dot_eyedir_sundir', 'altitude']

#minimal
#FEATURE_COLUMNS = ['eulerx', 'eulery', 'eulerz','dot_eyedir_sundir']

#maximal
#FEATURE_COLUMNS = ['orient_w', 'orient_x', 'orient_y', 'orient_z', 'eyedir_x', 'eyedir_y', 'eyedir_z', 'eulerx', 'eulery','sundir_x', 'sundir_y','sundir_z','dot', 'qdot','eulerz','dot_eyedir_sundir', 'azimuth', 'altitude']

#directional
#FEATURE_COLUMNS = ['orient_w', 'orient_x', 'orient_y', 'orient_z', 'eyedir_x', 'eyedir_y','eyedir_z', 'sundir_x', 'sundir_y','sundir_z']

#thoughtfull:
#FEATURE_COLUMNS = ['eulerx', 'eulery', 'eulerz', 'azimuth', 'altitude']

# natural thoughtful
#FEATURE_COLUMNS = ['eulerx', 'eulery', 'eulerz', 'eyedir_x', 'eyedir_y', 'eyedir_z', 'azimuth', 'altitude']




#FEATURE_COLUMNS = ['orient_w', 'orient_x', 'orient_y', 'orient_z', 'eyedir_x', 'eyedir_y', 'eyedir_z', 'azimuth', 'altitude', 'dot', 'qdot']
#FEATURE_COLUMNS = ['orient_w', 'orient_x', 'orient_y', 'orient_z', 'eyedir_x', 'eyedir_y', 'eyedir_z', 'azimuth', 'altitude', 'qdot']
#FEATURE_COLUMNS = ['azimuth', 'altitude', 'qdot']


#FEATURE_COLUMNS = ['orient_w', 'orient_x', 'orient_y', 'orient_z', 'eyedir_x', 'eyedir_y', 'eyedir_z', 'height', 'azimuth', 'altitude','dot_eyedir_sundir']



TARGET_COLUMNS = ['color_red', 'color_green', 'color_blue']
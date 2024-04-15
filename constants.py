# Define feature and target columns once
'''
    "altitude", 
    "angle",
    "avgx", "avgy", "avgz",
    "avghsvx", "avghsvy", "avghsvz",
    "avgxyzx", "avgxyzy", "avgxyzz",
    "azimuth",
    "Cdx", "Cdy", "Cdz",
    "dot",
    "dot_eyedir_sundir",
    "eulerx", "eulery", "eulerz",
    "eyedirx", "eyediry", "eyedirz",
    "frame",
    "height",
    "height",
    "hsvx", "hsvy", "hsvz",
    "orientw", "orientx", "orienty", "orientz",
    "qdot",
    "sundirx", "sundiry", "sundirz",
    "xyzx", "xyzy", "xyzz"  
'''

# natural
#FEATURE_COLUMNS = ["orientw", "orientx", "orienty", "orientz","eyedirx", "eyediry", "eyedirz", "azimuth", "altitude"]
#FEATURE_COLUMNS = ['eyedir_x', 'eyedir_y', 'eyedir_z', 'azimuth', 'altitude']

#FEATURE_COLUMNS = ['dot_eyedir_sundir','altitude']
#FEATURE_COLUMNS = ['sundirx', 'sundiry', 'sundirz','eyedirx', 'eyediry', 'eyedirz','altitude']

#minimal
#FEATURE_COLUMNS = ['eulerx', 'eulery', 'eulerz','dot_eyedir_sundir']
#FEATURE_COLUMNS = ['dot_eyedir_sundir']
    
#FEATURE_COLUMNS = ["eyedirz","dot_eyedir_sundir"]
FEATURE_COLUMNS = ["dot","dot_eyedir_sundir"]

TARGET_COLUMNS = ['Cdx', 'Cdy', 'Cdz']
#TARGET_COLUMNS = ['hsvx', 'hsvy', 'hsvz']
#TARGET_COLUMNS = ['xyzx', 'xyzy', 'xyzz']
#TARGET_COLUMNS = ['avgx', 'avgy', 'avgz']
#TARGET_COLUMNS = ['avghsvx', 'avghsvy', 'avghsvz']
#TARGET_COLUMNS = ['avgxyzx', 'avgxyzy', 'avgxyzz']

#TARGET_COLUMNS = []
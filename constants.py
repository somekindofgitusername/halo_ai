# Define feature and target columns once
'''
att_names = [
    "altitude", 
    "angle",
    "avg3x", "avg3y", "avg3z",
    "avghsvx", "avghsvy", "avghsvz",
    "avgx", "avgy", "avgz",
    "avgxyzx", "avgxyzy", "avgxyzz",
    "azimuth",
    "Cdx", "Cdy", "Cdz",
    "domnormx","domnormy","domnormz",
    "domx","domy","domz",
    "dot",
    "dot_eyedir_sundir",
    "eulerx", "eulery", "eulerz",
    "eyedirx", "eyediry", "eyedirz",
    "frame",
    "height",
    "hsvx", "hsvy", "hsvz",
    "navgx", "navgy", "navgz",
    "orientw", "orientx", "orienty", "orientz",
    "qdot",
    "ralti",
    "rangle",
    "razi",
    "rcrossx","rcrossy","rcrossz",
    "rdot",
    "rdot_eyedir_sundir",
    "reulerx", "reulery", "reulerz",
    "reyedirx", "reyediry", "reyedirz",
    "rncrossx","rncrossy","rncrossz",
    "rorientw", "rorientx", "rorienty", "rorientz",  
    "rsundirx", "rsundiry", "rsundirz",  
    "sundirx", "sundiry", "sundirz",
    "xyzx", "xyzy", "xyzz"
]
'''

# natural
#FEATURE_COLUMNS = ["eyedirz","rcrossx","rcrossy","rcrossz","dot_eyedir_sundir", "altitude"]

FEATURE_COLUMNS = ["rorientw", "rorientx", "rorienty", "rorientz",  "reyedirz", "altitude", "dot_eyedir_sundir", "azimuth"]

#FEATURE_COLUMNS = ["eyedirz","dot_eyedir_sundir", "altitude"]


#FEATURE_COLUMNS = ["reyedirz", "rdot_eyedir_sundir", "altitude", "angle"]
#FEATURE_COLUMNS = ["eyedirz","dot_eyedir_sundir", "altitude", "angle"]


FEATURE_COLUMNS.sort()

#TARGET_COLUMNS = ['Cdx','Cdy','Cdz']
#TARGET_COLUMNS = ['Cdx']
#TARGET_COLUMNS = ['hsvx']
#TARGET_COLUMNS = ['hsvx', 'hsvy', 'hsvz']
#TARGET_COLUMNS = ["domnormx","domnormy","domnormz"]
#TARGET_COLUMNS = ["domx","domy","domz"]

#TARGET_COLUMNS = ['xyzx', 'xyzy', 'xyzz']
#TARGET_COLUMNS = ['avgx', 'avgy', 'avgz']
#TARGET_COLUMNS = ['avg3x', 'avg3y', 'avg3z']
#TARGET_COLUMNS = ['navgx', 'navgy', 'navgz']
TARGET_COLUMNS = ['avghsvx', 'avghsvy', 'avghsvz']
#TARGET_COLUMNS = ['avgxyzx', 'avgxyzy', 'avgxyzz']

#TARGET_COLUMNS = []
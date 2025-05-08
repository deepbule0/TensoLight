from .TensoLight_rotation import TensoLight_Dataset_rotated_lights
from .TensoLight_rotation_real import TensoLight_Dataset_rotated_lights_real
from .TensoLight_relighting_test import TensoLight_Relighting_test
from .TensoLight_colmap import TensoLight_Dataset_colmap



dataset_dict = {'TensoLight_rotated_lights':TensoLight_Dataset_rotated_lights,
                'TensoLight_rotated_lights_real': TensoLight_Dataset_rotated_lights_real,
                'TensoLight_relighting_test':TensoLight_Relighting_test,
                'TensoLight_colmap':TensoLight_Dataset_colmap,
                }

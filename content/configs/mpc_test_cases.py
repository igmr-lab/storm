from storm_kit.util_file import get_configs_path, join_path

SCENARIOS = {'pole':        join_path(get_configs_path(),'pole.yml'),
             'pole2':       join_path(get_configs_path(),'pole2.yml'),
             'pole3':       join_path(get_configs_path(),'pole3.yml'),
             'obstacleFree':join_path(get_configs_path(),'obstacleFree.yml'),
             'wall':        join_path(get_configs_path(),'wall.yml')}


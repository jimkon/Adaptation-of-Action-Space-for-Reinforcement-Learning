import json
import os
from os.path import splitext, basename
import zipfile


def load(file_name):
    data = Data(None)

    if zipfile.is_zipfile(file_name):
        print('Data: Unziping ', file_name, '...')
        with zipfile.ZipFile(file_name) as myzip:
            string = (myzip.read(myzip.namelist()[0]).decode("utf-8"))
            data.set_data(json.loads(string))
    else:
        print('Data: Loading ', file_name, '...')
        with open(file_name, 'r') as f:
            data.set_data(json.load(f))
    return data


def merge(path_to_dir, zipfiles=True, delete_merged=True):

    print("Merging files in ", path_to_dir)

    import re
    import numpy as np

    assert os.path.exists(path_to_dir), "Path does not exist"

    all_paths = os.listdir(path_to_dir)

    file_indexes = np.where(list(os.path.isfile(path_to_dir+"/"+file) for file in all_paths))[0]

    file_names = list(all_paths[index] for index in file_indexes)

    file_pattern = None
    if not zipfiles:
        m = re.search("data_.*", file_names[0])
        if m is not None:
            file_pattern = "(\d)*"+m.group(0)
            file_template = "{}"+m.group(0)
        else:
            return None

    else:
        m = re.search("data_.*\.json.zip", file_names[0])
        if m is not None:
            m = re.search("data_.*#", file_names[0])
            file_pattern = m.group(0)+"(\d)+.json.zip"
            file_template = m.group(0)+"{}"+".json.zip"
        else:
            return None

    print("Pattern found:", file_pattern, "\nSearching files...")

    filtered_filenames = []
    for file in file_names:
        m = re.search(file_pattern, file)
        if m is not None:
            filtered_filenames.append(file)

    for f in filtered_filenames:
        print("Found:", f)

    print("Sorting files...")
    if not zipfiles:
        key = "(\d)*"
    else:
        key = "#(\d)*"

    nums = []
    for file in filtered_filenames:
        m = re.search(key, file)
        if m is not None:
            matched = m.group(0)
            # print("for", file, "key is", m.group(0))
            nums.append(int(matched) if not zipfiles else int(matched[1:]))
        else:
            print("Failed on sorting, return None")
            return None

    nums = sorted(nums)

    # print("final names")
    print("Merging...")
    prev_i = nums[0]
    result_data = load(path_to_dir+"/"+file_template.format(prev_i))

    for i in nums[1:]:
        file_name = path_to_dir+"/"+file_template.format(i)
        temp_data = load(file_name)
        result_data.merge(temp_data)

        if i-prev_i > 1:
            print("Warning: Not consecutive files: [",
                  file_template.format(prev_i), ",", file_template.format(i),
                  "]. Possible missing file", file_template.format(prev_i+1))

        prev_i = i

    result_data.path = path_to_dir
    result_data.data['experiment']['number_of_episodes'] *= len(nums)
    result_data.save()

    if delete_merged:
        print("Deleting merged files")
        for i in nums:
            os.remove(path_to_dir+"/"+file_template.format(nums[i]))

    print("Successfully merged files in", path_to_dir)


class Data:

    AUTOSAVE_BATCH_SIZE = 1e5

    DATA_TEMPLATE = '''
    {
        "id":0,
        "agent":{
          "name":"default_name",
          "max_actions":0,
          "k":0,
          "version":0
        },
        "experiment":{
          "name":"no_exp",
          "actions_low":null,
          "actions_high":null,
          "number_of_episodes":0
        },
        "simulation":{
          "episodes":[]
        }

    }
    '''

    EPISODE_TEMPLATE = '''
    {
        "id":0,
        "states":[],
        "actions":[],
        "actors_actions":[],
        "ndn_actions":[],
        "rewards":[],
        "action_space_sizes":[]
    }
    '''

    def __init__(self, path, comment=""):

        self.path = path
        self.comment = comment

        if path:
            self.path = "{}/data/{}/".format(self.path, comment)
            if not os.path.exists(self.path):
                os.makedirs(self.path, exist_ok=True)

        self.data = json.loads(self.DATA_TEMPLATE)
        self.episode = json.loads(self.EPISODE_TEMPLATE)
        self.episode_id = 0
        self.temp_saves = 0
        self.data_added = 0

    def __increase_data_counter(self, n=1):
        self.data_added += n

    def set_id(self, n):
        self.data['id'] = n

    def set_agent(self, name, max_actions, k, version):
        self.data['agent']['name'] = name
        self.data['agent']['max_actions'] = max_actions
        self.data['agent']['k'] = k
        self.data['agent']['version'] = version

    def set_experiment(self, name, low, high, eps):
        self.data['experiment']['name'] = name
        self.data['experiment']['actions_low'] = low
        self.data['experiment']['actions_high'] = high
        self.data['experiment']['number_of_episodes'] = eps

    def set_state(self, state):
        self.episode['states'].append(state)
        self.__increase_data_counter(len(state))

    def set_action(self, action):
        self.episode['actions'].append(action)
        self.__increase_data_counter(len(action))

    def set_actors_action(self, action):
        self.episode['actors_actions'].append(action)
        self.__increase_data_counter(len(action))

    def set_ndn_action(self, action):
        self.episode['ndn_actions'].append(action)
        self.__increase_data_counter(len(action))

    def set_reward(self, reward):
        self.episode['rewards'].append(reward)
        self.__increase_data_counter()

    def set_action_space_size(self, min_size, max_size):
        self.episode['action_space_sizes'].append(min_size)
        self.episode['action_space_sizes'].append(max_size)
        self.__increase_data_counter()

    def end_of_episode(self):
        self.data['simulation']['episodes'].append(self.episode)
        self.episode = json.loads(self.EPISODE_TEMPLATE)
        self.episode_id += 1
        self.episode['id'] = self.episode_id

    def finish_and_store_episode(self):
        self.end_of_episode()
        # print(self.data_added / self.AUTOSAVE_BATCH_SIZE)
        if self.data_added > self.AUTOSAVE_BATCH_SIZE:
            self.temp_save()

    def get_file_name(self):
        return 'data_{}_{}_{}{}k{}#{}'.format(self.get_episodes(),
                                              self.get_agent_name(),
                                              self.get_experiment()[:3],
                                              self.data['agent']['max_actions'],
                                              self.data['agent']['k'],
                                              self.get_id())

    def get_episodes(self):
        return self.data['experiment']['number_of_episodes']

    def get_agent_name(self):
        return '{}{}'.format(self.data['agent']['name'][:4],
                             self.data['agent']['version'])

    def get_id(self):
        return self.data['id']

    def get_experiment(self):
        return self.data['experiment']['name']

    def print_data(self):
        print(json.dumps(self.data, indent=2, sort_keys=True))

    def print_stats(self):
        for key in self.data.keys():
            d = self.data[key]
            if key == 'simulation':
                print('episodes:', len(d['episodes']))
            else:
                print(json.dumps(d, indent=2, sort_keys=True))

    def merge(self, data_in):
        if type(data_in) is Data:
            data = data_in.data
        else:
            data = data_in

        for ep in data['simulation']['episodes']:
            self.episode = ep
            self.end_of_episode()

    def set_data(self, data):
        self.data = data

    def save(self, prefix='', final_save=True, comment=""):

        if final_save and self.temp_saves > 0:
            if self.data_added > 0:
                # self.end_of_episode()
                self.temp_save()
            print('Data: Merging all temporary files')
            for i in range(self.temp_saves):

                file_name = '{}/temp/{}{}.json'.format(
                    self.path,
                    i,
                    self.get_file_name())
                temp_data = load(file_name)
                self.merge(temp_data)
                os.remove(file_name)

        directory = "{}/{}/".format(self.path, comment)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        final_file_name = "{}/{}{}.json".format(directory, prefix, self.get_file_name())
        if final_save:
            print('Data: Ziping', final_file_name)
            with zipfile.ZipFile(final_file_name + '.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as myzip:
                myzip.writestr(basename(final_file_name), json.dumps(
                    self.data, indent=2, sort_keys=True))
        else:
            with open(final_file_name, 'w', encoding="UTF-8") as f:
                print('Data: Saving', final_file_name)
                json.dump(self.data, f)

    def temp_save(self):
        if self.data_added == 0:
            return
        self.save(prefix=str(self.temp_saves), comment='temp', final_save=False)
        self.temp_saves += 1
        self.data['simulation']['episodes'] = []  # reset
        self.data_added = 0

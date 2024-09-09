import pandas as pd
from typing import List, Optional

class Data:
    def __init__(self, label=None):
        self.label = label

    def __str__(self):
        return f"Data Object: label={self.label}"

class Node(Data):
    def __init__(self, position, label=None):
        super().__init__(label)
        self.position = position

    def __str__(self):
        return f"Node Object: label={self.label}, position={self.position}"

class Bar(Data):
    def __init__(self, mass, density, from_node_label=None, to_node_label=None, alt_node_label:Optional[str]=None, label:Optional[str]=None):
        super().__init__(label)
        self.mass = mass
        self.density = density
        self.from_node_label = from_node_label
        self.alt_node_label = alt_node_label
        self.to_node_label = to_node_label

    def __str__(self):
        return f"Bar Object: label={self.label}, mass={self.mass}, density={self.density}, from_node_label={self.from_node_label}, to_node_label={self.to_node_label}"

class String(Data):
    def __init__(self, stiffness, ori_length, color=None, from_node_label=None, to_node_label=None, label=None):
        super().__init__(label)
        self.stiffness = stiffness
        self.ori_length = ori_length
        self.from_node_label = from_node_label
        self.to_node_label = to_node_label
        self.color = color

    def __str__(self):
        return f"String Object: label={self.label}, stiffness={self.stiffness}, ori_length={self.ori_length}, from_node_label={self.from_node_label}, to_node_label={self.to_node_label}"

class Ball(Data):
    def __init__(self, mass, density, radius, middle_node_label:str, connected_node_label_list:List[str], label=None):
        super().__init__(label)
        self.mass = mass
        self.density = density
        self.radius = radius
        self.middle_node_label = middle_node_label
        self.connected_node_label_list = connected_node_label_list

    def __str__(self):
        return f"Ball Object: label={self.label}, mass={self.mass}, density={self.density}, middle_node_label={self.middle_node_label}, \nconnected_node_label_list=\n{self.connected_node_label_list}"


class Structure:
    def __init__(self):
        self.data = pd.DataFrame(columns=['ID', 'Type', 'Data'])
        self.next_id = {}
    
    def add(self, item):
        data_type = type(item).__name__
        if data_type not in self.next_id:
            self.next_id[data_type] = 0

        new_row = pd.DataFrame({'ID': [self.next_id[data_type]], 'Type': [data_type], 'Data': [item.__dict__]})
        self.data = pd.concat([self.data, new_row], ignore_index=True)
        self.next_id[data_type] += 1

        self.sort_dataframe()

    def create_item(self, item_type, item_data: dict):
        if item_type == 'Node':
            item = Node(**item_data)
        elif item_type == 'Bar':
            item = Bar(**item_data)
        elif item_type == 'String':
            item = String(**item_data)
        elif item_type == 'Ball':
            item = Ball(**item_data)
        else:
            raise ValueError("Unsupported item type")
        return item
    
    def delete(self, item_id, item_type):
        self.data = self.data[(self.data['ID'] != item_id) | (self.data['Type'] != item_type)]

    def update(self, item_id, new_item):
        data_type = type(new_item).__name__
        new_data_dict = new_item.__dict__
        
        def update_data(row):
            if row['ID'] == item_id and row['Type'] == data_type:
                return new_data_dict
            return row['Data']
        
        self.data['Data'] = self.data.apply(update_data, axis=1)
    
    def search(self, item_id, item_type):
        return self.data[(self.data['ID'] == item_id) & (self.data['Type'] == item_type)]
    
    def reorder_ids(self):
        for data_type in self.data['Type'].unique():
            type_data = self.data[self.data['Type'] == data_type]
            new_ids = list(range(len(type_data)))
            self.data.loc[self.data['Type'] == data_type, 'ID'] = new_ids
    
    def sort_dataframe(self):
        type_order = {'Node': 1, 'Bar': 2, 'String': 3}
        self.data['Type_Order'] =self.data['Type'].map(type_order)
        self.data.sort_values(by=['Type_Order', 'ID'], inplace=True)
        self.data.drop(columns=['Type_Order'], inplace=True)


    def export_to_csv(self, filename):
        self.data.to_csv(filename, index=False)
    
    def import_from_csv(self, filename):
        self.data = pd.read_csv(filename)
        for data_type in self.data['Type'].unique():
            max_id = self.data[self.data['Type'] == data_type]['ID'].max()
            self.next_id[data_type] = max_id + 1 if not pd.isnull(max_id) else 0

if __name__ == "__main__":
    structure = Structure()

    node1 = Node((10, 20))
    node2 = Node((30, 40),label='11')
    node3 = Node((10, 50),label='qqq')

    bar1 = Bar(100, 2.5, from_node_label='qqq',to_node_label='11')
    bar2 = Bar(150, 3.0)

    string1 = String(stiffness=250, ori_length=5, from_node_label='1', to_node_label='2')
    string2 = String(stiffness=250, ori_length=5, from_node_label='3', to_node_label='2')
    string3 = String(stiffness=250, ori_length=5, from_node_label='4', to_node_label='2')
    
    
    ball1 = Ball(mass=1, density=1000, middle_node_label='11', connected_node_label_list=['11','22'], label='afsd')

    structure.add(node1)
    structure.add(node2)
    structure.add(node3)

    structure.add(bar1)
    structure.add(bar2)

    structure.add(string1)
    structure.add(string2)
    structure.add(string3)
    
    structure.add(ball1)
    # # 打印当前结构数据
    # print("Current structure data:")
    # print(structure.data)

    # # 删除ID为1的节点
    # print("\nDeleting Node 1:")
    # structure.delete(0, 'Node')
    # print(structure.data)

    ## 重新排序ID
    # structure.reorder_ids()

    # # 更新ID为1的杆件
    # new_bar2 = Bar(200, 4.0, 'New Bar 2')
    # print("\nUpdating Bar 1:")
    # structure.update(1, new_bar2)
    # print(structure.data)

    # # 搜索ID为1的Bar
    # print("\nSearching for ID 1:")
    # result = structure.search(1, 'Bar')
    # print(result)

    structure.export_to_csv('./data/test.csv')
    # print(structure.create_item('Bar', {'mass': 10, 'density': 5, 'from_node_label': 'Node 1', 'to_node_label': 'Node 2', 'label': 'Bar 1'}))


    # new_structure = Structure()
    # new_structure.import_from_csv('data.csv')

    # print(new_structure.data)
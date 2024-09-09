from structure import Structure, Data
from typing import Optional,Dict
import numpy as np
import pandas as pd

class Tensegrity(Structure):
    def __init__(self, Nodes_array:np.ndarray, Cb_in:Optional[np.ndarray]=None, Cs_in:Optional[np.ndarray]=None):
        super().__init__()

        self.N = Nodes_array

        if Cb_in is not None:
            self.C_b = self.tenseg_ind2C(Cb_in)
        else:
            self.C_b = None
        if Cs_in is not None:
            self.C_s = self.tenseg_ind2C(Cs_in)
        else:
            self.C_s = None

    
    def generate_Node_element(self, position:np.ndarray, label:Optional[str]=None):
        return self.create_item('Node', {'position':position, 'label':label})
    
    def generate_Bar_element(self, mass, density, from_node_label=None, to_node_label=None, label:Optional[str]=None):
        return self.create_item('Bar', 
                                {'mass':mass, 'density':density, 
                                 'from_node_label':from_node_label,
                                 'to_node_label':to_node_label, 'label':label})
    
    def generate_String_element(self, stiffness, ori_length, color=None, from_node_label=None, to_node_label=None, label=None):
        return self.create_item('String', 
                                {'stiffness':stiffness, 'ori_length':ori_length, 
                                 'color':color,
                                 'from_node_label':from_node_label,
                                 'to_node_label':to_node_label, 'label':label})

    def fill_Node_data(self):
            num_cols = self.N.shape[1]
            for col_index in range(num_cols):
                col_values = tuple(self.N[:, col_index])
                self.add(self.generate_Node_element(position=col_values, label=f'node_{col_index}'))

    def fill_Bar_data(self):
        if self.C_b is None or self.C_s is None or self.N is None:
            print('数据不完全')
            return False
        B = np.dot(self.N, self.C_b.T)
        for j in range(B.shape[1]):
            x_str= str(np.argmax(self.C_b[j, :] == -1))
            y_str = str(np.argmax(self.C_b[j, :] == 1))
            self.add(self.generate_Bar_element(mass=1, density=1e3,
                                            from_node_label=f'node_{x_str}',
                                            to_node_label=f'node_{y_str}',
                                            label=f'bar_{x_str}_{y_str}'))
    def fill_String_data(self, stiffness):
        if self.C_b is None or self.C_s is None or self.N is None:
            print('数据不完全')
            return False
        
        S = np.dot(self.N, self.C_s.T)
        for j in range(S.shape[1]):
            self.add(self.generate_String_element(stiffness=stiffness,
                                                ori_length=0.07,
                                                from_node_label=f'node_{np.argmax(self.C_s[j, :] == -1)}',
                                                to_node_label=f'node_{np.argmax(self.C_s[j, :] == 1)}',
                                                label=f'string_{np.argmax(self.C_s[j, :] == -1)}_{np.argmax(self.C_s[j, :] == 1)}'))

    def fill_all_data(self):             
        self.fill_Node_data()
        self.fill_Bar_data()
        self.fill_String_data()
    
    def modify_data_element(self, label, **kwargs):
        for index, row in self.data.iterrows():
            data_dict = row['Data']
            if data_dict.get('label') == label:
                # 更新找到的项
                data_dict.update(kwargs)
                self.data.at[index, 'Data'] = data_dict
                return
        print("Error: Label '{}' not found in data.".format(label))

    def get_data_element(self, label)->Data:
        for index, row in self.data.iterrows():
            data_dict = row['Data']
            # print(type(self.data))
            if data_dict.get('label') == label:
                # 更新找到的项
                item = self.create_item(item_type=row['Type'], item_data=data_dict)
                return item
        print("Error: Label '{}' not found in data.".format(label))

    def check_duplicate_labels(self):
        label_dict = {}
        for index, row in self.data.iterrows():
            data_dict = row['Data']
            label = data_dict.get('label')
            if label in label_dict:
                label_dict[label].append(data_dict)
            else:
                label_dict[label] = [data_dict]
        
        for label, data_list in label_dict.items():
            if len(data_list) > 1:
                print("Duplicate label '{}':".format(label))
                for data_dict in data_list:
                    print(data_dict)
                print()

    # 使连接的杆件对矩阵转换为连接矩阵
    def tenseg_ind2C(self, C_ind):
        """
        Creates a connectivity matrix from input index notation array and node matrix.

        Inputs:
            C_ind: index connectivity array (m x 2 array for m members)
            Nodes: node matrix (3 x n array for n nodes)

        Outputs:
            C_mat: connectivity matrix (m x n matrix satisfying M = N*C')

        Example: If given four nodes (N is a 3x4 matrix), and you want one bar to
        be the vector from node 1 to 3, and another bar from node 2 to 4, input
        C_ind would be: C_ind = np.array([[1, 3], [2, 4]])

        C_b = tenseg_ind2C(np.array([[1, 3], [2, 4]]), N)
        """
        nmembers = C_ind.shape[0]  # Number of members being created
        n = self.N.shape[1]  # Number of nodes in the structure

        # Initialize connectivity matrix
        C_mat = np.zeros((n, nmembers))

        for i in range(nmembers):  # Go through each member
            # Get indices for start and end points
            side1 = C_ind[i, 0]
            side2 = C_ind[i, 1]

            # Put -1 at index for start point, 1 at index for end point
            C_mat[side1 - 1, i] = -1
            C_mat[side2 - 1, i] = 1

        return C_mat.T

    def move_nodes(self, x, y, z):
        """使节点整体移动

        Args:
            x (_type_): x方向移动距离
            y (_type_): y方向移动距离
            z (_type_): z方向移动距离
        """
        self.N = self.N + np.array([[x], [y], [z]])

if __name__ == "__main__":
    # N = np.array([[0.5, 0, 0], [0, 0.866, 0], [-0.5, 0, 0], [0.5, 0, 1], [0, 0.866, 1], [-0.5, 0, 1]]).T
    # Cb_in = np.array([[3, 5], [1, 6], [2, 4]])  # Bar 1 connects node 3 to 5, etc
    # Cs_in = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 6], [6, 4], [1, 4], [2, 5], [3, 6]])  # String one is node 1 to 2
    phi = (1 + 5 ** 0.5) / 2
    temp = np.array([[0, -1, phi], [0, -1, -phi], [0, 1, phi], [0, 1, -phi]]) / (2 * phi)
    N = np.concatenate((temp, temp[:, [2, 0, 1]], temp[:, [1, 2, 0]], [[0, 0, 0]]), axis=0).T
    N = N[:, [1, 3, 5, 7, 9, 11, 0, 2, 4, 6, 8, 10, 12]]

    Cb_in = np.array([[1, 7], [2, 8], [3, 9], 
                      [4, 10], [5, 11], [6, 12]])  
    
    Cs_in = np.array([[1, 3], [1, 5], [1, 6], [1, 9],
                      [2, 3], [2, 9], [2, 11], [2, 12], 
                      [3, 5], [3, 11],
                      [4, 5], [4, 7], [4, 8], [4, 11],
                      [5, 7], 
                      [6, 7], [6, 9], [6, 10],
                      [7, 10], 
                      [8, 10], [8, 11], [8, 12],
                      [9, 12], 
                      [10, 12]])  
    
    
    tensegrity = Tensegrity(Nodes_array=N, Cb_in=Cb_in, Cs_in=Cs_in)
    tensegrity.move_nodes(0, 0, 1)
    tensegrity.fill_all_data()

    ## 修改某项
    # tensegrity.modify_data_element('node_1', position=(1, 1, 1))

    # # 获取某项
    # string_new = tensegrity.get_data_element('string_0_4')
    # print(string_new)

    # 增加某项
    # bar_new = tensegrity.get_data_element('bar_0_6') #string_new是series类型，不能直接添加
    # tensegrity.add(bar_new)
    # # 检查重复性
    # tensegrity.check_duplicate_labels()
    # # print(tensegrity.data)
    file_path = './data/data.csv'
    tensegrity.export_to_csv(file_path)


import numpy as np

class CleanMask():
    def __init__(self, stack, threshold):
        ## input 
        self.stack = stack #stack of masks
        self.num_row = self.stack.shape[0]
        self.num_col = self.stack.shape[1]
        self.masks = np.zeros((self.num_row, self.num_col), dtype=np.uint16)
        self.id = 1
        self.dict = {}
        self.unique = {}
        self.cells = {}
        self.threshold = threshold
        
    def getMasks(self):
        return self.masks
    
    def visited(self, row, col):
        if self.visitedPoints[row,col] == 1:
            return True
    
    def get_ids(self, array):
        ids = set()
        for value in array:
            if value > 0:
                ids.add(value)
        frozen_set_ids = frozenset(ids)
        return frozen_set_ids
    
    def at_least_one(self, values, ids):
        for value in values:
            if value in ids:
                return True
        return False
  

    def build_connectivity_matrix(self):
        """
        Build the connectivity matrix between cells ids
        """
        n_labels = int(np.max(self.stack))
        self.conn_matrix = np.zeros((n_labels + 1, n_labels + 1))

        #Fill up the connectivity matrix with weighted connections
        for row in range(self.num_row):
            for col in range(self.num_col):
                ids = np.unique(self.stack[row, col, :]).tolist()
                #print ids
                #Remove background from ids
                if 0 in ids:
                    ids.remove(0)
                #If it is not a background
                if ids == None:
                    continue
                ids_size = len(ids) 
                if ids_size > 1:
                    for i in range(ids_size - 1):
                        for j in range(i+1,ids_size):
                           self.conn_matrix[ids[i], ids[j] ] += 1
                           self.conn_matrix[ids[j], ids[i] ] += 1


    def merge_cells(self):
        """
        Find connected cells between multiple inferences and merge strongly connected 
        cells that have overlap more than a threshold.
        """
        self.build_connectivity_matrix()
        print("Connectivity matrix built")        
        #Filter out week connections
        self.conn_matrix[self.conn_matrix < self.threshold] = 0

        #Get connected components 
        np.fill_diagonal(self.conn_matrix, 1)

        from scipy.sparse.csgraph import csgraph_from_dense, connected_components
        graph = csgraph_from_dense(self.conn_matrix)
        n_conn_comp, graph_labels =  connected_components(graph, True) 

        print(self.conn_matrix.shape)
        print(n_conn_comp)
        print(graph_labels.shape)
        print(type(graph_labels))

        #Convert all labels to their group:
        updated_labels = graph_labels[self.stack]
        print("applied lookup")
        print(updated_labels.shape)
        self.masks = np.max(updated_labels, 2)
        return self.masks
        
      
    def save(self, save_path):
        np.save('inference-stack', self.stack)

        from libtiff import TIFF
        my_mask = self.getMasks().astype("int16")
        print(my_mask)
        tiff = TIFF.open(save_path, mode='w')
        tiff.write_image(my_mask)
        tiff.close()

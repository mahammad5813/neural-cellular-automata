import numpy as np
import cv2
import copy
import time
import json
from dataclasses import dataclass, field

def inverse_gaussian(x):
    return -1/pow(2,(0.6*pow(x,2)))+1

def reLU(x):
    return max(0,x)

def conway_gol(x):
    if x==3 or x==11 or x==12:
        return 1
    return 0

def activation_func(x):
    return inverse_gaussian(x)

@dataclass
class NeuralNetwork:
    size: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    biases: list = field(default_factory=list)
    
    def __post_init__(self):
        self.weights.append(np.zeros((self.size[0],self.size[1])))
        # self.weights.append(np.ones((self.size[0],self.size[1])))
        # self.weights[0][4] = 9
        # rand1 = np.random.uniform(-1,1)
        # rand2 = np.random.uniform(-1,1)
        # rand3 = np.random.uniform(-1,1)
        
        rand1 = 0.6800000071525574
        rand2 = -0.8999999761581421
        rand3 = -0.6600000262260437
        
        self.weights[0][0::2] = rand1
        self.weights[0][1::2] = rand2
        self.weights[0][4] = rand3
        
        # self.weights[0][0] = 0.5645999908447266
        # self.weights[0][2] = 0.5645999908447266
        # self.weights[0][6] = 0.5645999908447266
        # self.weights[0][8] = 0.5645999908447266
        
        # self.weights[0][1] = -0.7159000039100647
        # self.weights[0][3] = -0.7585999965667725
        # self.weights[0][5] = -0.7585999965667725
        # self.weights[0][7] = -0.7159000039100647
        
        # self.weights[0][4] = 0.6269000172615051
        
        # print(len(self.weights))
        # for i in range(len(self.size)):
            # self.weights.append(np.random.uniform(-1,1,(self.size[i],self.size[i+1])))
            # self.weights.append(np.random.randint(0,2,(self.size[i],self.size[i+1])))
    
            # if i == len(self.size)-2:
            #     break
        print(self.weights)
            
    def propogate(self, input_values):
        for i in range(len(self.weights)):
            # input_min = min(input_values)
            # input_max = max(input_values)
            # input_range = input_max-input_min
            # if input_range != 0:
            #     input_values = [(val-input_min)/input_range for val in input_values]
            
            input_values = np.dot(input_values, self.weights[i])
            input_values = [activation_func(val) for val in input_values] 
            
            if i == len(self.weights)-1:
                return input_values[0]
            
            
            
nn = NeuralNetwork([9,1])
resize_width = 600
nrows = 100
ncols = 100

# render_grid = np.zeros((nrows, ncols, 3))
grid = np.random.randint(0,2,(nrows,ncols))
# grid = np.zeros((nrows,ncols))
n = 0
######################################################
# mode = input("Mode(0-random, 1-loadWeights): ")
# if mode == "1":
#     path_to_file = input("Enter full path to json file: ")
#     with open(path_to_file, "r") as file:
#         nn.weights = [list(arr) for arr in eval(file.readline())]
#         print(nn.weights)
########################################################
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, ((600,round(600*(nrows/ncols)))))
# out = cv2.VideoWriter("output.avi", fourcc, 20.0, ((nrows, ncols)))

# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame = frame[40:440, 120:520]
# grid = cv2.resize(frame, (100,100))
# cv2.imshow('frame', grid)
# cv2.waitKey(0)
# cap.release()
fixed_pixels = []
btn_down = False
def on_mouse(event,x,y,flags,params):
    global fixed_pixels, btn_down
    x = int(x/(resize_width/nrows))
    y = int(y/(resize_width/ncols))
    if x<nrows and y<ncols:
        if btn_down:
            fixed_pixels = [y,x]
        if event == cv2.EVENT_LBUTTONDOWN:
            fixed_pixels = [y,x]
            btn_down = True
        if event == cv2.EVENT_LBUTTONUP:
            fixed_pixels = []
            btn_down = False
        
while True:
    n+=1
    # print(n)
    if fixed_pixels!=[]:
        grid[fixed_pixels[0]-1:fixed_pixels[0]+2,fixed_pixels[1]-1:fixed_pixels[1]+2]=1
    render_grid = np.zeros((nrows, ncols, 3), np.uint8)
    temp = np.zeros(grid.shape)
    for row_i in range(nrows):
        if row_i==0:
            row_range = np.concatenate(([grid[-1]],grid[:2]))
        elif row_i==nrows-1:
            row_range = np.concatenate((grid[row_i-1:], [grid[0]]))
        else:
            row_range = grid[row_i-1:row_i+2]
            
        for col_i in range(ncols):
            if col_i==0:
                col_range = np.column_stack((row_range[:, -1:], row_range[:, :2]))
            elif col_i == ncols-1:
                col_range = np.column_stack((row_range[:, col_i-1:], row_range[:, :1]))
            else:
                col_range = row_range[:,col_i-1:col_i+2]
            temp[row_i][col_i] = nn.propogate(col_range.flatten())
            if temp[row_i][col_i] > 1:
                temp[row_i][col_i] = 1
            if temp[row_i][col_i] < 0:
                temp[row_i][col_i] = 0
            
            
    grid = copy.deepcopy(temp)
    # grid = temp
    
    # temp = cv2.resize(temp, (resize_width,round(resize_width*(nrows/ncols))), interpolation=cv2.INTER_AREA)
    
    # print(render_grid.shape)
    
    if n%2 == 0:
        render_grid[:,:,-1] = temp*255
        render_grid = cv2.resize(render_grid, (resize_width,round(resize_width*(nrows/ncols))), interpolation=cv2.INTER_AREA)
        out.write(render_grid)
        cv2.imshow("NCA",render_grid)
        cv2.setMouseCallback('NCA', on_mouse)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(nn.weights)
        with open("weights.json", "w") as jsonFile:
            weights_l = [arr.tolist() for arr in nn.weights]
            json.dump(weights_l, jsonFile)
    # time.sleep(0.5)
    
out.release()

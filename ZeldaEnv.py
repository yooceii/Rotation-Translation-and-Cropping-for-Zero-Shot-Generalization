import gym
import numpy as np
import cv2

ac = ["ACTION_NIL",
      "ACTION_USE",
      "ACTION_LEFT",
      "ACTION_RIGHT",
      "ACTION_DOWN",
      "ACTION_UP"]
direction = {2:3, 3:1, 4:2, 5:0}

class ZeldaEnv(gym.Wrapper):
    def __init__(self, env, crop=False, rotate=False, shape=(84,84)):
        print(type(crop), type(rotate))
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.direction = 3
        self.crop = crop
        self.rotate = rotate
        self.env.observation_space.shape = shape + (3,)
        self.shape = shape
        if self.crop:
            print("translate")
        if self.rotate:
            print("rotate")
        # self.env.observation_space.shape = self.env.observation_space.shape[:-1] + (3,)

    def step(self, action):
        if action != 0 and action != 1:
            self.direction = action
        obs, reward, done, info = self.env.step(action)
        if done:
            # print(info)
            if info.get("winner") == 'PLAYER_WINS':
                info["episode"]['c'] = 1
            else:
                info["episode"]['c'] = 0
            # print(info)
        
        if self.crop:
            # print("crop")
            obs = mask(obs, info, direction.get(self.direction), rotate=self.rotate)
        elif self.rotate:
            # print("rotate")
            obs = np.rot90(obs, k=direction.get(self.direction))

        obs = cv2.resize(obs, self.shape)
        return obs[:,:,:-1], reward, done, info

    def get_action_meanings(self):    
        return self.unwrappered.get_action_meanings()

    def reset(self):
        obs = self.env.reset()
        if self.rotate:
            obs = np.rot90(obs, k=direction.get(self.direction))

        if self.crop:
            _, _, _, info = self.env.step(0)
            obs = self.env.reset()
            self.direction = 3
            obs = mask(obs, info, direction.get(self.direction), rotate=self.rotate)
        elif self.rotate:
            obs = np.rot90(obs, k=direction.get(self.direction))

        obs = cv2.resize(obs, self.shape)
        return obs[:,:,:-1]

def resize(image, shape):
    image = cv2.resize(image, shape)
    return image
    

def crop(image, mask, ascii, u, d, l, r, pixel):
    # print("crop")
    # loc = np.asarray(np.where((ascii=="nokey")|(ascii=="withkey"))).T[0]
    loc = np.asarray(np.where((np.core.defchararray.find(ascii,"nokey")!=-1)|(np.core.defchararray.find(ascii,"withkey")!=-1))).T[0]

    blank = np.full(((u+d+1)*pixel, (l+r+1)*pixel, image.shape[2]), 0, dtype='uint8')
    for i in range(-u, d+1):
        for j in range(-l, r+1):
            if loc[0] + i >= 0 and loc[1] + j >= 0 and loc[0] + i <= ascii.shape[0] - 1  and loc[1] + j <= ascii.shape[1] - 1 and mask[i+u,j+l] != 'b':
                # pos on mask
                pos = [i+u, j+l]
                # print(loc[0]+i, loc[1]+j, ascii.shape)
                # print(image.shape)
                blank[pos[0]*pixel:(1+pos[0])*pixel, pos[1]*pixel:(pos[1]+1)*pixel, :] = image[(loc[0]+i)*pixel:(loc[0]+i+1)*pixel,(loc[1]+j)*pixel:(loc[1]+j+1)*pixel,:]
    return blank


def mask(image, info, k=0, pixel=10, rotate=True):
    mask = 's,s,s,s,s\nb,s,s,s,b\nb,s,a,s,b\nb,b,s,b,b'
    mask_list = np.array([l.split(",") for l in mask.split("\n")])
    mask_pos = np.asarray(np.where(mask_list=='a')).T[0]
    ascii = np.rot90([l.split(",") for l in info["ascii"].split("\n")], k=k)
    image = np.rot90(image, k=k)
    obs = crop(image, mask_list, ascii, mask_pos[0], mask_list.shape[0]-mask_pos[0]-1, mask_pos[1], mask_list.shape[1]-mask_pos[1]-1, pixel=pixel)
    return np.rot90(obs, k=4-k) if not rotate else obs

def create_mask(mask_ascii, pixel):
    seen = np.full((pixel, pixel, 1), 255)
    black = np.full((pixel, pixel, 1), 0)
    layer = None
    mask_ascii_list = [l.split(',') for l in mask_ascii.split("\n")]
    for i in mask_ascii_list:
        line = None
        for j in i:
            if j == 's' or j == 'a':
                if line is None:
                    line = seen
                else:
                    line = np.concatenate((line, seen), axis=1)
            elif j == 'b':
                if line is None:
                    line = black
                else:
                    line = np.concatenate((line, black), axis=1)
        if layer is None:
            layer = line
        else:
            layer = np.concatenate((layer, line), axis=0)
    return np.concatenate((layer, layer, layer), axis=2)

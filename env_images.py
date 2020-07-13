from multi_goal.envs.pybullet_labyrinth_env import Labyrinth

env = Labyrinth(visualize=True)
pb = env._sim._bullet


cam_height = 30
cam_pos = [7.5, 2.5, cam_height]
target = [*cam_pos[:2], 0]
viewMatrix = pb.computeViewMatrix(
    cameraEyePosition=cam_pos,
    cameraTargetPosition=target,
    cameraUpVector=[0, 1, 0])

projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1,
    nearVal=0.1,
    farVal=cam_height+30)

resolution = (64, 64)
width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
    width=resolution[0],
    height=resolution[1],
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix)

import matplotlib.pyplot as plt
plt.ion()
plt.imshow(rgbImg)
input("Exit")

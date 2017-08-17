% This script explains the projective relations between joints2D and joints3D
% variables in the SURREAL dataset saved during Blender rendering.

clear; clc; close all;

data = load('01_01_c0001_info.mat');

for joint = 1:size(data.joints3D, 2)
    for frame = 1:size(data.joints3D, 3)
        J3D = data.joints3D(:, joint, frame); % e.g. [ -0.8645; -1.0915; -1.0260 ]
        J2D = data.joints2D(:, joint, frame); % e.g. [ 89; 112 ]
 
        T = data.camLoc; % e.g. [6.7456; -0.9839; -1.9293] (*cam_ob.location)

        intrinsic = getIntrinsicBlender();
        % e.g.
        %  [ 600     0   160  ;
        %      0   600   120  ;
        %      0     0     1 ];
        
        extrinsic = getExtrinsicBlender(T);
        % e.g.
        % [        0         0   -1.0000   -1.9293  ;
        %          0    1.0000         0    0.9839  ;
        %    -1.0000         0         0    6.7456 ];

        % Project 3D point (J3D) to get 2D point  (J2D).
        P2D = intrinsic * extrinsic * [J3D ; 1];
        P2D = [round(P2D(1)/P2D(3)); round(P2D(2)/P2D(3))];

        % Check that projected 2D points (P2D) match previously saved 2D points (J2D).
        assert(all(P2D == J2D));

        fprintf(['correct   x:\t%d \nprojected x:\t%d\n'...
               '\ncorrect   y:\t%d \nprojected y:\t%d\n\n'],...
                J2D(1), P2D(1), J2D(2), P2D(2));
    end
end

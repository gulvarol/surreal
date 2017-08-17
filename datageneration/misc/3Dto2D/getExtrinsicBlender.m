function RT = getExtrinsicBlender(T)
% function RT = getExtrinsicBlender(T)
%   returns extrinsic camera matrix
%   
%   T : translation vector from Blender (*cam_ob.location)
%   RT: extrinsic computer vision camera matrix 
%   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera

% Take the first 3 columns of the matrix_world in Blender and transpose.
% This is hard-coded since all images in SURREAL use the same.
R_world2bcam = [ 0  0  1   ;
                 0 -1  0   ;
                -1  0  0 ]';
% *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
%                               (0., -1, 0., -1.0),
%                               (-1., 0., 0., 0.),
%                               (0.0, 0.0, 0.0, 1.0)))

% Convert camera location to translation vector used in coordinate changes
T_world2bcam = -1 * R_world2bcam * T;

% Following is needed to convert Blender camera to computer vision camera
R_bcam2cv = [ 1  0  0 ;
              0 -1  0 ;
              0  0 -1];

% Build the coordinate transform matrix from world to computer vision camera
R_world2cv = R_bcam2cv*R_world2bcam;
T_world2cv = R_bcam2cv*T_world2bcam;

% Put into 3x4 matrix
RT = [ R_world2cv  T_world2cv ];

end

function K = getIntrinsicBlender()
% function K = getIntrinsicBlender
%   returns intrinsic camera matrix
%   
%   Parameters are hard-coded since all SURREAL images use the same.


% These are set in Blender (datageneration/main_part1.py)
res_x_px         = 320; % *scn.render.resolution_x
res_y_px         = 240; % *scn.render.resolution_y
f_mm             = 60;  % *cam_ob.data.lens
sensor_w_mm      = 32;  % *cam_ob.data.sensor_width
sensor_h_mm = sensor_w_mm * res_y_px / res_x_px; % *cam_ob.data.sensor_height (function of others)

scale = 1; % *scn.render.resolution_percentage/100
skew  = 0; % only use rectangular pixels
pixel_aspect_ratio = 1;

% From similar triangles:
% sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
fx_px = f_mm * res_x_px * scale / sensor_w_mm;
fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm;

% Center of the image
u = res_x_px * scale / 2;
v = res_y_px * scale / 2;

% Intrinsic camera matrix
K = [ fx_px  skew   u  ;
      0      fy_px  v  ;
      0      0      1 ];

end

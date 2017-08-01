-- From https://github.com/joeyhng/trainplot

require 'torch'
require 'optim'
local json = require 'cjson'

local TrainPlotter = torch.class('TrainPlotter')

function TrainPlotter:__init(path)
  --If the path already exists, then concatenate the information
  if(file_exists(path)) then
    local file = assert(io.open(path, "r"))
    local json_str = file:read("*all")
    self.figures = json.decode(json_str)
  else
  -- Otherwise initialize empty
    self.figures = {}
  end
  self.path = path
end

-- Information string displayed in the page
function TrainPlotter:info(s)
  self.figures['info_str'] = s
end

function TrainPlotter:add(fig_id, plot_id, iter, data)
    if data ~= data then data = -1 end
    if data == 1/0 then data = 1e-30 end
    if data == -1/0 then data = -1e-30 end

    if not fig_id then
        fig_id = plot_id
    end

    -- create figure if not exists
    if not self.figures[fig_id] then
      self.figures[fig_id] = {}
      self.figures[fig_id]['data'] = {}
      self.figures[fig_id]['layout'] = {['title']=fig_id}
    end

    local fig_data = self.figures[fig_id]['data']
    local plot
    for k, v in pairs(fig_data) do
      if v['name'] == plot_id then plot = v end
    end
    if not plot then
      plot = {['name'] = plot_id, ['x'] = {}, ['y'] = {}}
      table.insert(fig_data, plot)
    end
    table.insert(plot['x'], iter)
    table.insert(plot['y'], data)

    local file = io.open(self.path, 'w')
    file:write(json.encode(self.figures))
    file:close()
end

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

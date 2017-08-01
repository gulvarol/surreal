local ffi=require 'ffi'
------ Some FFI stuff used to pass storages between threads ------------------
ffi.cdef[[
void THFloatStorage_free(THFloatStorage *self);
void THLongStorage_free(THLongStorage *self);
]]

function makeDataParallel(model, nGPU)
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
      cutorch.setDevice(opt.GPU)
   end
   return model
end

local function cleanDPT(module)
    -- This assumes this DPT was created by the function above: all the
    -- module.modules are clones of the same network on different GPUs
    -- hence we only need to keep one when saving the model to the disk.
    local newDPT = nn.DataParallelTable(1)
    cutorch.setDevice(opt.GPU)
    newDPT:add(module:get(1), opt.GPU)
    return newDPT
end

function saveDataParallel(filename, model)
    if torch.type(model) == 'nn.DataParallelTable' then
        torch.save(filename, cleanDPT(model))
    elseif torch.type(model) == 'nn.Sequential' then
        local temp_model = nn.Sequential()
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                temp_model:add(cleanDPT(module))
            else
                temp_model:add(module)
            end
        end
        torch.save(filename, temp_model)
    else
        torch.save(filename, model)
        print('The saved model is not a Sequential or DataParallelTable module.')
    end
end

function loadDataParallel(filename, nGPU)
    local model = torch.load(filename)
    if torch.type(model) == 'nn.DataParallelTable' then
        return makeDataParallel(model:get(1):float(), nGPU)
    elseif torch.type(model) == 'nn.Sequential' then
        for i,module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
            end
        end
        return model
    else
        print('The loaded model is not a Sequential or DataParallelTable module.')
        return model
    end
end

function setFloatStorage(tensor, storage_p)
    assert(storage_p and storage_p ~= 0, "FloatStorage is NULL pointer");
    local cstorage = ffi.cast('THFloatStorage*', torch.pointer(tensor:storage()))
    if cstorage ~= nil then
        ffi.C['THFloatStorage_free'](cstorage)
    end
    local storage = ffi.cast('THFloatStorage*', storage_p)
    tensor:cdata().storage = storage
end

function setLongStorage(tensor, storage_p)
    assert(storage_p and storage_p ~= 0, "LongStorage is NULL pointer");
    local cstorage = ffi.cast('THLongStorage*', torch.pointer(tensor:storage()))
    if cstorage ~= nil then
       ffi.C['THLongStorage_free'](cstorage)
    end
    local storage = ffi.cast('THLongStorage*', storage_p)
    tensor:cdata().storage = storage
end

function sendTensor(inputs)
    local size = inputs:size()
    local ttype = inputs:type()
    local i_stg =  tonumber(ffi.cast('intptr_t', torch.pointer(inputs:storage())))
    inputs:cdata().storage = nil
    return {i_stg, size, ttype}
end

function receiveTensor(obj, buffer)
    local pointer = obj[1]
    local size = obj[2]
    local ttype = obj[3]
    if buffer then
        buffer:resize(size)
        assert(buffer:type() == ttype, 'Buffer is wrong type')
    else
        buffer = torch[ttype].new():resize(size)      
    end
    if ttype == 'torch.FloatTensor' then
        setFloatStorage(buffer, pointer)
    elseif ttype == 'torch.LongTensor' then
        setLongStorage(buffer, pointer)
    else
       error('Unknown type')
    end
    return buffer
end

function getDir(dirName)
    dirs = paths.dir(dirName)
    table.sort(dirs, function (a,b) return a < b end)
    for i = #dirs, 1, -1 do
        if(dirs[i] == '.' or dirs[i] == '..') then
            table.remove(dirs, i)
        end
    end
    return dirs
end

function getTightBox(label)
    -- Tighest bounding box covering the joint positions
    local tBox = {}
    tBox.x_min = label[{{}, {1}}]:min()
    tBox.y_min = label[{{}, {2}}]:min()
    tBox.x_max = label[{{}, {1}}]:max()
    tBox.y_max = label[{{}, {2}}]:max()
    tBox.humWidth  = tBox.x_max - tBox.x_min + 1
    tBox.humHeight = tBox.y_max - tBox.y_min + 1

    -- Slightly larger area to cover the head/feet of the human
    tBox.x_min = tBox.x_min - 0.25*tBox.humWidth -- left
    tBox.y_min = tBox.y_min - 0.35*tBox.humHeight -- top
    tBox.x_max = tBox.x_max + 0.25*tBox.humWidth -- right
    tBox.y_max = tBox.y_max + 0.25*tBox.humHeight -- bottom
    tBox.humWidth  = tBox.x_max - tBox.x_min + 1
    tBox.humHeight = tBox.y_max - tBox.y_min +1

    return tBox
end

function getCenter(label)
    local tBox = getTightBox(label)
    local center_x = tBox.x_min + tBox.humWidth/2
    local center_y = tBox.y_min + tBox.humHeight/2

    return {center_x, center_y}
end

function getScale(label, imHeight)
    local tBox = getTightBox(label)
    return math.max(tBox.humHeight/240, tBox.humWidth/240)
end

function pause()
    io.stdin:read'*l'
end

function table2str ( v )
    if "string" == type( v ) then
        v = string.gsub( v, "\n", "\\n" )
        if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
          return "'" .. v .. "'"
        end
        return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
    else
        return "table" == type( v ) and table.tostring( v ) or
          tostring( v )
    end
end

function table.key_to_str ( k )
    if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
        return k
    else
        return "[" .. table.val_to_str( k ) .. "]"
    end
end

function table.tostring( tbl )
    local result, done = {}, {}
    for k, v in ipairs( tbl ) do
        table.insert( result, table.val_to_str( v ) )
        done[ k ] = true
    end
    for k, v in pairs( tbl ) do
        if not done[ k ] then
            table.insert( result,
              table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
        end
    end
    return "{" .. table.concat( result, "," ) .. "}"
end

local sim = require("sim")

string.startswith = function(self, str)
    return self:find("^" .. str) ~= nil
end

local food_collected
local already_eaten

function sysCall_init()
    -- do some initialization here:

    local handle
    local pos

    handle = sim.getObject("/Food")
    pos = sim.getObjectPosition(handle, -1)
    pos[1] = ((math.random() * 2) - 4.1)
    pos[2] = ((math.random() * 2) - 0.2)
    if pos[2] > 0.6 and pos[2] < 1.1 then
        pos[2] = pos[2] + 0.5
    end
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectInt32Parameter(handle, sim.objintparam_visibility_layer, 1)  -- Ensure visibility

    handle = sim.getObject("/Food0")
    pos = sim.getObjectPosition(handle, -1)
    pos[1] = ((math.random() * 2) - 4.1)
    pos[2] = ((math.random() * 2) - 0.2)
    if pos[2] > 0.6 and pos[2] < 1.1 then
        pos[2] = pos[2] + 0.5
    end
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectInt32Parameter(handle, sim.objintparam_visibility_layer, 1)  -- Ensure visibility

    handle = sim.getObject("/Food1")
    pos = sim.getObjectPosition(handle, -1)
    pos[1] = ((math.random() * 2) - 4.1)
    pos[2] = ((math.random() * 2) - 0.2)
    if pos[2] > 0.6 and pos[2] < 1.1 then
        pos[2] = pos[2] + 0.5
    end
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectInt32Parameter(handle, sim.objintparam_visibility_layer, 1)  -- Ensure visibility

    handle = sim.getObject("/Food2")
    pos = sim.getObjectPosition(handle, -1)
    pos[1] = ((math.random() * 2) - 4.1)
    if pos[2] > 0.6 and pos[2] < 1.1 then
        pos[1] = pos[1] + 0.6
    end
    pos[2] = ((math.random() * 2) - 0.2)
    if pos[2] > 0.6 and pos[2] < 1.1 then
        pos[2] = pos[2] + 0.5
    end
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectInt32Parameter(handle, sim.objintparam_visibility_layer, 1)  -- Ensure visibility

    handle = sim.getObject("/Food3")
    pos = sim.getObjectPosition(handle, -1)
    pos[1] = ((math.random() * 2) - 4.1)
    pos[2] = ((math.random() * 2) - 0.2)
    if pos[2] > 0.6 and pos[2] < 1.1 then
        pos[2] = pos[2] + 0.5
    end
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectInt32Parameter(handle, sim.objintparam_visibility_layer, 1)  -- Ensure visibility

    handle = sim.getObject("/Food4")
    pos = sim.getObjectPosition(handle, -1)
    pos[1] = ((math.random() * 2) - 4.1)
    pos[2] = ((math.random() * 2) - 0.2)
    if pos[2] > 0.6 and pos[2] < 1.1 then
        pos[2] = pos[2] + 0.5
    end
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectInt32Parameter(handle, sim.objintparam_visibility_layer, 1)  -- Ensure visibility

    handle = sim.getObject("/Food5")
    pos = sim.getObjectPosition(handle, -1)
    pos[1] = ((math.random() * 2) - 4.1)
    pos[2] = ((math.random() * 2) - 0.2)
    if pos[2] > 0.6 and pos[2] < 1.1 then
        pos[2] = pos[2] + 0.5
    end
    sim.setObjectPosition(handle, -1, pos)
    sim.setObjectInt32Parameter(handle, sim.objintparam_visibility_layer, 1)  -- Ensure visibility

    food_collected = 0
    already_eaten = {}
    print("Scene initialized")
end

function sysCall_actuation()
    -- put your actuation code here
end

function sysCall_sensing()
    -- put your sensing code here
end

function sysCall_cleanup()
    -- do some clean-up here
end

function collect_food(handle)
    if not already_eaten[handle] then
        --print(handle)
        already_eaten[handle] = true
        local pos = sim.getObjectPosition(handle, -1)
        pos[3] = pos[3] + 1
        sim.setObjectPosition(handle, -1, pos)
        sim.setObjectInt32Parameter(handle, sim.objintparam_visibility_layer, 0)  -- Hide the food object
        food_collected = food_collected + 1
        print("Collected food " .. food_collected)
    end
end

function sysCall_contactCallback(inData)
    h1 = sim.getObjectName(inData.handle1)
    if h1:startswith("Food") then
        collect_food(inData.handle1)
    end
    return inData  -- Return input data unmodified
end

function remote_get_collected_food(inInts, inFloats, inStrings, inBuffer)
    return { food_collected }, {}, {}, ""
end

-- Additional system calls can be defined here
--[[
function sysCall_suspend()
end

function sysCall_resume()
end

function sysCall_dynCallback(inData)
end

function sysCall_jointCallback(inData)
    return outData
end

function sysCall_beforeCopy(inData)
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." will be copied")
    end
end

function sysCall_afterCopy(inData)
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." was copied")
    end
end

function sysCall_beforeDelete(inData)
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." will be deleted")
    end
    -- inData.allObjects indicates if all objects in the scene will be deleted
end

function sysCall_afterDelete(inData)
    for key,value in pairs(inData.objectHandles) do
        print("Object with handle "..key.." was deleted")
    end
    -- inData.allObjects indicates if all objects in the scene were deleted
end
--]]

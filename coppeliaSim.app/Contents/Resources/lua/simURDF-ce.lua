local codeEditorInfos = [[
string robot_name = simURDF.import(string urdf, int options=0, string packageStrReplace=nil)
simURDF.export(int origModel, string fileName, int options=0)
simURDF.sendTF(int modelHandle, string fileName)
]]

registerCodeEditorInfos("simURDF", codeEditorInfos)

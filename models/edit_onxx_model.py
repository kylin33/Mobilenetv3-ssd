import onnx
from onnx import helper, checker
from onnx import TensorProto

onnx_model = onnx.load("/home/whf/Temp/11-扫地机/projects/anno_model/new_model_mos_backaug_0215/best_sim.onnx")
graph = onnx_model.graph
node = graph.node

def createGraphMemberMap(graph_member_list):
    member_map=dict();
    for n in graph_member_list:
        member_map[n.name]=n;
    return member_map

def split_io_list(io_list,new_names_all):
    #splits input/output list to identify removed, retained and totally new nodes    
    removed_names=[]
    retained_names=[]
    for n in io_list:
        if n.name not in new_names_all:                
            removed_names.append(n.name)              
        if n.name in new_names_all:                
            retained_names.append(n.name)                      
    new_names=list(set(new_names_all)-set(retained_names)) 
    return [removed_names,retained_names,new_names]

def traceDependentNodes(graph,name,node_input_names,node_map, initializer_map):
    # recurisvely traces all dependent nodes for a given output nodes in a graph    
    for n in graph.node:
        for noutput in n.output:       
            if (noutput == name) and (n.name not in node_input_names):
                # give node "name" is node n's output, so add node "n" to node_input_names list 
                node_input_names.append(n.name)
                if n.name in node_map.keys():
                    for ninput in node_map[n.name].input:
                        # trace input node's inputs 
                        node_input_names = traceDependentNodes(graph,ninput,node_input_names,node_map, initializer_map)                                        
    # don't forget the initializers they can be terminal inputs on a path.                    
    if name in initializer_map.keys():
        node_input_names.append(name)                    
    return node_input_names   

remove_node = []
remove_output = ["1198", "1199"]
new_output_name = []
new_outout = []

def remove_name(node):
    for i in node.input:
        if i in remove_output:
            remove_node.append(node.name)
            remove_output.append(node.output[0])
            break

for i in range(len(node)):
    remove_name(node[i])
    if node[i].op_type == "Concat":
        node_ = node[i]
        if node_.output[0] == "1198" or node_.output[0] == "1199":
            new_output_name.append(node_.name)
            new_outout.append(node_.output[0])
            print(node_)

# print(remove_node)




# print(remove_node, new_output_name)
output_map = createGraphMemberMap(graph.output)
output_shape_map = {new_outout[0]:[1,3000,6],new_outout[1]:[1,3000,4]}
[removed_names,retained_names,new_names]=split_io_list(graph.output,new_output_name)
for name in removed_names:
    if name in output_map.keys():
        # print('************************',output_map[name])
        graph.output.remove(output_map[name])                              
for name in new_outout:
    new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape_map[name])
    graph.output.extend([new_nv])
    # graph.output.extend([onnx.ValueInfoProto()])


# reshape_input = []
# remove_reshape_name = []
# for i in range(len(node)):
#     remove_name(node[i])
#     if node[i].op_type == "Concat":
#         node_ = node[i]
#         if node_.output[0] == "1206" or node_.output[0] == "1207":
#             for input in node_.input:
#                 reshape_input.append(input)
# for i in range(len(node)):
#     if node[i].op_type == "Reshape" and node[i].output[0] in reshape_input:
#         remove_reshape_name.append(node[i].name)



node_map = createGraphMemberMap(graph.node)
initializer_map = createGraphMemberMap(graph.initializer)


valid_node_names=[]
for new_output_node_name in new_output_name:
    valid_node_names=traceDependentNodes(graph,new_output_node_name,valid_node_names,node_map, initializer_map)
    valid_node_names=list(set(valid_node_names))
# invalid_node_names = list( (set(node_map.keys()) | set(initializer_map.keys())) - set(valid_node_names))
invalid_node_names = remove_node
# Remove all the invalid nodes from the graph               
for name in invalid_node_names:
    if name in node_map.keys():
        graph.node.remove(node_map[name])        
    if name in initializer_map.keys():
        graph.initializer.remove(initializer_map[name])


# print(graph.output)
print("output model Errors: ", onnx.checker.check_model(onnx_model))
onnx.save(onnx_model, "edit_onnx_model.onnx")
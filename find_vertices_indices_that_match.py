#
#
# corresponding vertices indices of two meshes

import bpy
import bmesh

tshirt = bpy.data.objects["V1:Body"]
tshirt_low_poly = bpy.data.objects["Low_poly_man:Proxy741"]


# from all vertices of low poly mesh
match_list = []
match_list_lp = []

mesh = tshirt.data
mesh_low_poly = tshirt_low_poly.data

for vert_lp in mesh_low_poly.vertices:
	w_v_lp = tshirt_low_poly.matrix_world * vert_lp.co
	
	# now we have world position vertex, try to find match in other mesh
	for vert in mesh.vertices:
		w_v = tshirt.matrix_world * vert.co
		
		# we have a match
		if ((w_v_lp - w_v).length < 0.01):
			match_list.append(vert.index)
			match_list_lp.append(vert_lp.index)
	

# check is selecting correctly the vertices
#print(match_list)
#print(match_list_lp)
#print(len(match_list_lp))

for idx in range(0,len(match_list)):
	mesh.vertices[match_list[idx]].select = True

for idx in range(0,len(match_list_lp)):
	mesh_low_poly.vertices[idx].select = True
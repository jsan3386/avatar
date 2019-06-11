import os
import time
import numpy as np

# save path

############# DECLARE PART ##############
part = "strength"     #Options are: stomach, height, breast, torso, armslegs, hip, gender, weight, muscle, length, strength
######################################

root_models_path = "/home/aniol/IRI/DadesMarta/models/parts/" + part

# Can't just pass an exporter a path, must give it a validation function
def filename(targetExt, different = False):
	global root_models_path, model_name
	if  targetExt.lower() != 'obj':
		log.warning("expected extension '.%s' but got '%s'", targetExt,  'obj')
	return  os.path.join(root_models_path, 'humans', model_name + '.' + targetExt)

# Can't just pass an exporter a path, must give it a validation function
def naked_filename(targetExt, different = False):
	global root_models_path, naked_model_name
	if  targetExt.lower() != 'obj':
		log.warning("expected extension '.%s' but got '%s'", targetExt,  'obj')
	return  os.path.join(root_models_path, 'humans', naked_model_name + '.' + targetExt)



# --- Generate several human models ------
   
# get current human
human = G.app.selectedHuman

# Getting exporter
exporter = G.app.getCategory('Files').getTaskByName('Export').getExporter('Wavefront obj')

shape = G.app.getCategory('Modelling').getTaskByName('Random')
eyes = G.app.getCategory('Geometries').getTaskByName('Eyes')
eyes.selectProxy(None)
#tongue = G.app.getCategory('Geometries').getTaskByName('Tongue')
#tongue.selectProxy(None)


numModels = 1

for m in range(0,numModels):

	# First letter needs to be capital
	model_name = "Model%04d" %  m
	naked_model_name = "Model%04d_naked" %  m

	# set random body shape
	shape.myRandomize(part,m,numModels)

	# Save model without clothes
	exporter.export(human, naked_filename)

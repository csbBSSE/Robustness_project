import shutil
import os
from os import listdir
from os.path import isfile, join
import time

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

for i in topofiles:
	print(i)
	
	os.system("./RACIPE {}.topo {}.ids -SBML_model 1 -num_stability 4 -Toggle_f_p 0 -Toggle_T_test 0".format(i, i))
	time.sleep(2)
	
	try:
		os.remove("{}_solution.dat".format(i))
	except:
		pass
	try:
		os.remove("{}_solution_1.dat".format(i))
	except:
		pass 
	try:
		os.remove("{}_solution_2.dat".format(i))
	except:
		pass
	try:
		os.remove("{}_solution_3.dat".format(i))
	except:
		pass
	try:
		os.remove("{}_solution_4.dat".format(i))
	except:
		pass	
	try:
		os.remove("{}_parameters.dat".format(i))
	except:
		pass	
	try:
		os.remove("{}.prs".format(i))
	except:
		pass	
	try:
		os.remove("{}.cfg".format(i))
	except:
		pass	
	try:
		os.remove("{}_T_test.dat".format(i))
	except:
		pass

for i in topofiles:
	try:
		os.remove("{}_solution.dat".format(i))
	except:
		pass
	try:
		os.remove("{}_solution_1.dat".format(i))
	except:
		pass 
	try:
		os.remove("{}_solution_2.dat".format(i))
	except:
		pass
	try:
		os.remove("{}_solution_3.dat".format(i))
	except:
		pass
	try:
		os.remove("{}_solution_4.dat".format(i))
	except:
		pass	
	try:
		os.remove("{}_parameters.dat".format(i))
	except:
		pass	
	try:
		os.remove("{}.prs".format(i))
	except:
		pass	
	try:
		os.remove("{}.cfg".format(i))
	except:
		pass	
	try:
		os.remove("{}_T_test.dat".format(i))
	except:
		pass
		
time.sleep(2)
	
xml = [f for f in listdir("./") if isfile(join("./", f)) and 'xml' in f]
print(xml)
for i in xml:
	shutil.move(i, "xml/", copy_function = shutil.copy2)

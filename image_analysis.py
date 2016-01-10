"""
testing
"""
import pdb
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
from scipy import ndimage, misc
import scipy as sp
from transforms import *
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import glob
import colorcorrect.algorithm as cca
from skimage import color
from sklearn import svm
from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree, neighbors
from sklearn.externals.six import StringIO 
import pydot
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

showGraphs = True

def draw(img):
	fig = plt.figure()
	plt.imshow(img)
	plt.show()
	return fig

try:
	pf = open('classifier.pdata','r')
	clf = pickle.load(pf)
	print clf
	pf.close()
except Exception as e:
	print type(e)

	sources = glob.glob("images/*.data")
	print "analyzing files : "+ str(sources) 
	imgs = []

	cumulative = dict()
	cumulative2 = dict()

	for source in sources:
		s = source.split('/')
		filename = s[len(s)-1].split('.')[0]
		if filename == "rs9":
			filename+='.png'
		else:
			filename+='.JPG'
		print filename

		img = sp.ndimage.imread('images/'+filename)
		#img = cca.luminance_weighted_gray_world(img)
		img_lab = color.rgb2lab(copy.deepcopy(img))
		if showGraphs:
			plt.figure()
			plt.imshow(img)
			plt.show()
			fig = plt.figure()
			ax = fig.add_subplot(121, projection='3d',title="RGB")
			ax2 = fig.add_subplot(122, projection='3d',title="LAB")
		f = open(source,'r')
		lines = f.readlines()
		for line in lines:
			line = line.replace('\n','')
			#pdb.set_trace()
			if line == '':
				continue
			[x,y,c] = line.split(',')
			x = int(x)
			y = int(y)
			red = img[y][x][0]
			green = img[y][x][1]
			blue = img[y][x][2]
			[l,a,b] = img_lab[y][x]
			if showGraphs:
				ax.scatter(xs=red,ys=green,zs=blue,c=c,marker='x')
				ax2.scatter(xs=a,ys=b,zs=l,c=c,marker='x')
			if not c in cumulative.keys():
				cumulative[c] = [(red,green,blue)]
				cumulative2[c] = [(a,b,l)]
			else:
				cumulative[c].append((red,green,blue))
				cumulative2[c].append((a,b,l))
			print x,y,c, l, a, b
		if showGraphs:
			ax.set_xlabel('RED')
			ax.set_ylabel('GREEN')
			ax.set_zlabel('BLUE')
			ax2.set_xlabel('A')
			ax2.set_ylabel('B')
			ax2.set_zlabel('L')
			plt.show()
	if showGraphs:
		plt.close()
		fig=plt.figure()
		ax = fig.add_subplot(121, projection='3d')
		ax2 = fig.add_subplot(122, projection='3d')
	rgb_data = []
	rgb_labels = []
	lab_data = []
	lab_labels = []
	for c in cumulative.keys():
		for (r,b,g) in cumulative[c]:
			if showGraphs:
				ax.scatter(xs=r,ys=g,zs=b,c=c,marker='x')
			rgb_data.append([r,g,b,c])
		for (a,b,l) in cumulative2[c]:
			if showGraphs:
				ax2.scatter(xs=a,ys=b,zs=l,c=c,marker='x')
			lab_data.append([a,b,l,c])
	rgb_data_temp = np.random.permutation(rgb_data)
	lab_data_temp = np.random.permutation(lab_data)
	rgb_data = []
	lab_data = []
	lab_labels = []
	lab_labels = []
	for [r,g,b,c] in rgb_data_temp:
		rgb_data.append(np.array([r,g,b]).astype(np.float))
		rgb_labels.append(c)
	for [a,b,l,c] in lab_data_temp:
		lab_data.append(np.array([a,b,l]).astype(np.float))
		lab_labels.append(c)
	if showGraphs:
		ax.set_xlabel('RED')
		ax.set_ylabel('GREEN')
		ax.set_zlabel('BLUE')
		ax2.set_xlabel('A')
		ax2.set_ylabel('B')
		ax2.set_zlabel('L')
		plt.show()
		plt.close()


	# clf = tree.DecisionTreeClassifier(max_depth=None,min_samples_split=1)
	# clf.fit(rgb_data, rgb_labels)
	# print rgb_data
	# print rgb_labels
	# print clf
	# dot_data = StringIO() 
	# tree.export_graphviz(clf, out_file=dot_data) 
	# graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
	# graph.write_pdf("colors.pdf") 

	# print lab_data
	# print lab_labels
	# clf = AdaBoostClassifier(n_estimators=120).fit(lab_data, lab_labels)
	# scores = cross_val_score(clf, lab_data, lab_labels)
	# print scores
	# print clf


	print(lab_data)
	print(lab_labels)
	clf = RandomForestClassifier(n_estimators=20)
	clf.fit(lab_data, lab_labels)
	print clf

	pf = open('classifier.pdata','w')
	pickle.dump(clf,pf)
	pf.close()
	# print lab_data
	# print lab_labels
	# clf = svm.SVC(tol=0.001,gamma=0.01)
	# clf.fit(lab_data, lab_labels)
	# print clf 

class App:
	
	def __init__(self, master):

		self.root = master
		root.protocol('WM_DELETE_WINDOW', self.close)

		menubar = Menu(master)
		menubar.add_command(label="open", command=self.openFile)
		master.config(menu=menubar)

		frame = Frame(master)
		frame.grid(row=0,column=0)

		button_frame = Frame(frame)
		button_frame.grid(row=2,column=0)
		self.button = Button(button_frame, text="Quit", command=frame.quit)
		self.button.grid(row=3,column=0)

		self.fileName = ""
		self.imgDisplayed = None
		self.imgHeight = 0
		self.imgWeight = 0
		self.rgbImg = None
		self.labImg = None


		self.cframe = Frame(frame)
		self.canvas = Canvas(self.cframe,width=1200,height=800)		
		self.canvas.grid(row=0,column=0)
		self.cframe.grid(row=0,column=0)
		self.hbar=Scrollbar(self.cframe,orient=HORIZONTAL)
		self.hbar.grid(row=1,column=0, sticky=W+E)
		self.vbar=Scrollbar(self.cframe,orient=VERTICAL)
		self.vbar.grid(row=0,column=1, sticky=N+S)
		self.vbar.config(command=self.canvas.yview)
		self.hbar.config(command=self.canvas.xview)
		self.canvas.config(width=1200,height=800)
		self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
		self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

		self.canvas.bind("<Button 1>", self.predict_sample)


	def predict_sample(self,event):
		canvas = event.widget
		x = canvas.canvasx(event.x)
		y = canvas.canvasy(event.y)
		x,y = int(x),int(y)
		print [self.rgbImg[y][x]]
		print str(x)+" "+str(y)+" "+str(clf.predict(np.array([self.labImg[y][x][1],self.labImg[y][x][2],self.labImg[y][x][0]]).reshape(1,-1)))

	def openFile(self):
		self.fileName = tkFileDialog.askopenfilename(parent=self.root)
		print self.fileName
		self.canvas.delete(self.imgDisplayed)
		self.labels = dict() # wipe records
		f = sp.ndimage.imread(self.fileName)
		#f = cca.luminance_weighted_gray_world(f) # optional color correction algorithms
		misc.imsave('temp.png', f)
		f = Image.open('temp.png')
		photo = ImageTk.PhotoImage(f)
		self.imgDisplayed = self.canvas.create_image(0,0,image=photo, anchor='nw', state=NORMAL)
		self.canvas.config(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set,scrollregion=(0, 0, photo.width(), photo.height()))
		self.canvas.image = photo 
		self.canvas.grid(row=0,column=0)
		self.rgbImg = sp.ndimage.imread('temp.png')
		self.labImg = color.rgb2lab(copy.deepcopy(self.rgbImg))


	def close(self):
		self.root.destroy()





root = Tk()

app = App(root)
root.mainloop()
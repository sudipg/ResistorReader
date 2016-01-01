"""
Collect data points using tkinter GUI
Examine the data, see if separable
"""

from Tkinter import *
import tkFileDialog
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
from scipy import ndimage
import matplotlib.image as mpimg
from PIL import Image, ImageTk
import pdb
import copy
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil


class App:
	
	def __init__(self, master):

		self.root = master
		root.protocol('WM_DELETE_WINDOW', self.close)

		menubar = Menu(master)
		menubar.add_command(label="open", command=self.openFile)
		menubar.add_command(label="save", command=self.saveFile)
		master.config(menu=menubar)

		frame = Frame(master)
		frame.grid(row=0,column=0)

		button_frame = Frame(frame)
		button_frame.grid(row=1,column=0)
		self.button = Button(button_frame, text="Quit", command=frame.quit)
		self.button.grid(row=2,column=0)

		self.fileName = ""
		self.imgDisplayed = None
		self.colors = ["red", "brown", "blue", "black","brown","yellow","violet","green"]


		self.canvas = Canvas(frame, width=900, height=900, cursor="cross")		
		#self.imgDisplayed = self.canvas.create_image(0,0,image=photo, anchor='nw', state=NORMAL)
		#self.canvas.image = photo 
		self.canvas.grid(row=0,column=0)

		self.current_color = ""


		self.canvas.bind("<Button 1>", self.get_sample)

		
		self.color_selectors = dict()
		self.color_selectors_fn = dict()
		# pdb.set_trace()

		for color in self.colors:
			self.color_selectors_fn[color] = self.set_color(color)
			self.color_selectors[color] = Button(button_frame, text=color, command=self.color_selectors_fn[color])
			self.color_selectors[color].grid(row=1, column=self.colors.index(color))


		"""
		Data Structures for labels
		"""
		self.labels = dict() #disctionary color: list of labeled points



	def set_color(self, color):
		clr = copy.deepcopy(color)
		
		def setter():
			self.current_color = clr
			print self.current_color

		return setter

	def get_sample(self,event):
		canvas = event.widget
		x = canvas.canvasx(event.x)
		y = canvas.canvasy(event.y)
		x,y = int(x),int(y)
		canvas.create_oval(max(0,x-1),max(0,y-1),min(canvas.winfo_width(),x+1),min(canvas.winfo_height(),y+1), fill="red")
		print str(x) + " " + str(y) + " " + self.current_color
		if not self.current_color in self.labels.keys():
			self.labels[self.current_color] = [(x, y)]
		else:
			self.labels[self.current_color].append((x,y))

	def openFile(self):
		self.fileName = tkFileDialog.askopenfilename(parent=self.root)
		print self.fileName
		self.canvas.delete(self.imgDisplayed)
		f = Image.open('rs9.png')
		f = to_pil(cca.stretch(from_pil(f)))
		photo = ImageTk.PhotoImage(f)
		self.imgDisplayed = self.canvas.create_image(0,0,image=photo, anchor='nw', state=NORMAL)
		self.canvas.image = photo 
		self.canvas.grid(row=0,column=0)
		imgRGB = cv2.cvtColor(cv2.imread('rs9.png'), cv2.COLOR_BGR2RGB)


	def close(self):
		self.root.destroy()

	def saveFile(self):
		targetFileName = self.fileName.split(".")[0]+".data"
		f = open(targetFileName, 'a')
		for color in self.labels.keys():
			for (x,y) in self.labels[color]:
				f.write(str(x)+","+str(y)+","+color+"\n")
		print "data (coords and labels) saved at "+targetFileName
		f.close()




root = Tk()

app = App(root)
root.mainloop()



"""
Collect data points using tkinter GUI
Examine the data, see if separable
"""

from Tkinter import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import math
from scipy import ndimage
import matplotlib.image as mpimg
import pygame
from PIL import Image, ImageTk
import pdb
import copy


class App:
	
	def __init__(self, master):
		frame = Frame(master)
		frame.grid(row=0,column=0)

		button_frame = Frame(frame)
		button_frame.grid(row=1,column=0)
		self.button = Button(button_frame, text="Quit", command=frame.quit)
		self.button.grid(row=2,column=0)

		f = Image.open("rs9.png")
		photo = ImageTk.PhotoImage(f)
		img = cv2.imread('rs9.png', 1)

		self.colors = ["red", "brown", "blue", "black","brown","yellow","violet","green"]


		self.canvas = Canvas(frame, width=len(img[0]), height=len(img), cursor="cross")		
		self.canvas.create_image(0,0,image=photo, anchor='nw', state=NORMAL)
		self.canvas.image = photo 
		self.canvas.grid(row=0,column=0)

		self.current_color = ""

		def get_sample(event):
			canvas = event.widget
			x = canvas.canvasx(event.x)
			y = canvas.canvasy(event.y)
			x,y = int(x),int(y)
			self.canvas.create_oval(max(0,x-1),max(0,y-1),min(canvas.winfo_width(),x+1),min(canvas.winfo_height(),y+1), fill="red")
			print str(x) + " " + str(y) + " " + self.current_color

		self.canvas.bind("<Button 1>", get_sample)

		
		self.color_selectors = dict()
		self.color_selectors_fn = dict()
		# pdb.set_trace()

		for color in self.colors:
			self.color_selectors_fn[color] = self.set_color(color)
			self.color_selectors[color] = Button(button_frame, text=color, command=self.color_selectors_fn[color])
			self.color_selectors[color].grid(row=1, column=self.colors.index(color))

	def set_color(self, color):
		clr = copy.deepcopy(color)
		
		def setter():
			self.current_color = clr
			print self.current_color

		return setter





root = Tk()

app = App(root)
root.mainloop()
root.destroy()



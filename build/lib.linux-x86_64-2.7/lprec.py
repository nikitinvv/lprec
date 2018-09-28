import sys
import numpy
import Tkinter, tkFileDialog
import ttk 

import lprecmods.processing as processing
import lprecmods.fileParser as fileParser

class MainApplication:
	def __init__(self, master):
		print 'init';
	        self.master = master
		Tkinter.Label(master, text="h5file").grid(row=0,column=0)
		Tkinter.Label(master, text="N").grid(row=1,column=0)
		Tkinter.Label(master, text="Nproj").grid(row=2,column=0)
		Tkinter.Label(master, text="Nslice").grid(row=3,column=0)
		Tkinter.Label(master, text="pad").grid(row=4,column=0)

		Tkinter.Label(master, text="filter").grid(row=5,column=0)
		Tkinter.Label(master, text="recdir").grid(row=6,column=0)
		Tkinter.Label(master, text="center").grid(row=7,column=0)
		Tkinter.Label(master, text="idslice").grid(row=8,column=0)
		Tkinter.Label(master, text="Amp").grid(row=9,column=0)


		self.e=[None]*9;
		for k in range(0,9):
			self.e[k]=Tkinter.Entry();
		self.e[0].grid(row=0, column=1)
		self.e[1].grid(row=1, column=1)
		self.e[2].grid(row=2, column=1)
		self.e[3].grid(row=3, column=1)
		self.e[4].grid(row=4, column=1)
		self.e[5].grid(row=6, column=1)
		self.e[6].grid(row=7, column=1)
		self.e[7].grid(row=8, column=1)
		self.e[8].grid(row=9, column=1)


#		self.e[0].insert(0,'/home/viknik/s35_bmpzo_s1top__1_.h5');
		self.cb0=ttk.Combobox(master,values=['False','True']);self.cb0.grid(row=4,column=1);self.cb0.current(0);		
		self.cb=ttk.Combobox(master,values=['None','ramp','shepp-logan','hann','hamming']);self.cb.grid(row=5,column=1);self.cb.current(0);		


		Tkinter.Button(master, text='Read h5', command=self.button_readh5).grid(row=0, column=2,  pady=0)
		Tkinter.Button(master, text='LP plan', command=self.button_create).grid(row=5, column=2, pady=0)
		Tkinter.Button(master, text='Rec', command=self.button_rec).grid(row=10, column=2, pady=0)
		Tkinter.Button(master, text='Set dir', command=self.button_setrecdir).grid(row=6, column=2, pady=0)

	def button_create(self):
		N=numpy.int32(self.e[1].get());
		Nproj=numpy.int32(self.e[2].get());
		filter_type=self.cb.get();	
		pad=eval(self.cb0.get());	
		fname=self.e[0].get()
		self.prochandle=[]
		self.prochandle=processing.processing(N,Nproj,filter_type,pad,fname) 
		print 'done';


	def button_rec(self):
		idslicestr=self.e[7].get();
		center=numpy.int(self.e[6].get());		
		recfname=self.e[5].get()+'/rec';
		amp=self.e[8].get(); 
		if(not amp): 
			amp=None;
		else: amp=numpy.float32(amp);
 		if (not idslicestr):
			idslice=numpy.arange(0,numpy.int32(self.e[3].get()));
		else:
			idslice=numpy.tile(numpy.int32(eval(idslicestr)),[1]);		

		self.prochandle.rec(idslice,center,amp,recfname);
		print 'done';

	def button_readh5(self):
		path=tkFileDialog.askopenfilename(initialdir = ".",title='Select HDF5 file',filetypes = (("h5 files","*.h5"),("all files","*.*")))
		print path
		if (not path): 
			return;

		self.e[0].delete(0,Tkinter.END)
		self.e[0].insert(0,path);
		fname=self.e[0].get()
		fphandle=fileParser.fileParser(fname)
		shape=fphandle.takePars()
		self.e[1].delete(0,Tkinter.END);
		self.e[1].insert(0,shape[2]);
		self.e[2].delete(0,Tkinter.END);
		self.e[2].insert(0,shape[0]-1);#don't read last projection
		self.e[3].delete(0,Tkinter.END);
		self.e[3].insert(0,shape[1]);
	
		#extended size
		self.e[4].delete(0,Tkinter.END);
		self.e[4].insert(0,3*shape[2]/2);

	def button_setrecdir(self):
		path=tkFileDialog.askdirectory(initialdir = ".",title='Select directory for recosnstructions')
		self.e[5].delete(0,Tkinter.END)
		self.e[5].insert(0,path);
	

master = Tkinter.Tk()
app=MainApplication(master);
master.mainloop()


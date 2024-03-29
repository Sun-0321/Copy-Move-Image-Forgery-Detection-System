from tkinter import *
from tkinter import filedialog, ttk, messagebox
from PIL import ImageTk, Image
from detector import detectCopyMove, readImage

IMG_WIDTH = 400
IMG_HEIGHT = 400



def getImage(path, width, height):
    img = Image.open(path)
    img = img.resize((width, height), Image.ANTIALIAS)
    return ImageTk.PhotoImage(img)


class GUI(Frame):
    def __init__(self, parent=None):
        self.uploaded_image = None

    
        Frame.__init__(self, parent)
        self.pack()

    
        self.resultLabel = Label(self, text="COPY MOVE DETECTOR", font = ("Courier", 50))
        self.resultLabel.grid(row=0, column=0, columnspan=2)
        Grid.rowconfigure(self, 0, weight=1)

    
        blank_img = getImage("images/blank.png", IMG_WIDTH, IMG_HEIGHT)

    
        self.imagePanel = Label(self, image = blank_img)
        self.imagePanel.image = blank_img
        self.imagePanel.grid(row=1, column=0, padx=5)

    
        self.resultPanel = Label(self, image = blank_img)
        self.resultPanel.image = blank_img
        self.resultPanel.grid(row=1, column=1, padx=5)

    
        self.fileLabel = Label(self, text="No file selected", fg="grey", font = ("Times", 15))
        self.fileLabel.grid(row=2, column=0, columnspan=2)


    
        self.progressBar = ttk.Progressbar(self, length=500)
        self.progressBar.grid(row=3, column=0, columnspan=2)


    
        s = ttk.Style()
        s.configure('my.TButton', font=('Times', 15))

    
        self.uploadButton = ttk.Button(self, text="Upload Image", style="my.TButton", command=self.browseFile)
        self.uploadButton.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=5)

    
        self.startButton = ttk.Button(self, text="Start", style="my.TButton", command=self.runProg)
        self.startButton.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=5)

    
        self.quitButton = ttk.Button(self, text="Exit program", command=parent.quit)
        self.quitButton.grid(row=6, column=0, columnspan=2, sticky="e", pady=5)


    def browseFile(self):
        
        filename = filedialog.askopenfilename(title="Select an image", filetype=[("Image file", "*.jpg *.png")])

    
        if filename == "":
            return

        self.uploaded_image = filename
        self.progressBar['value'] = 0  
        self.fileLabel.configure(text=filename)    

    
        img = getImage(filename, IMG_WIDTH, IMG_HEIGHT)
        self.imagePanel.configure(image=img)
        self.imagePanel.image = img

    
        blank_img = getImage("images/blank.png", IMG_WIDTH, IMG_HEIGHT)
        self.resultPanel.configure(image=blank_img)
        self.resultPanel.image = blank_img

    
        self.resultLabel.configure(text="READY TO SCAN", foreground="black")



    def runProg(self):
        
    
        path = self.uploaded_image

    
        if path is None:
            messagebox.showerror('Error', "Please select image")   
            return

    
        img = readImage(path)

        result = detectCopyMove(img)    
        self.progressBar['value'] = 100
    
        if result:
        
            img = getImage("results.png", IMG_WIDTH, IMG_HEIGHT)
            self.resultPanel.configure(image=img)
            self.resultPanel.image = img

        
            self.resultLabel.configure(text="COPY-MOVE DETECTED", foreground="red")

        else:
        
            img = getImage("images/thumbs_up.png", IMG_WIDTH, IMG_HEIGHT)
            self.resultPanel.configure(image=img)
            self.resultPanel.image = img

        
            self.resultLabel.configure(text="ORIGINAL IMAGE", foreground="green")


def main():


    root = Tk()
    root.title("Copy-Move Detector")
    root.iconbitmap('images\SASTRA-logo.ico')


    root.protocol("WM_DELETE_WINDOW", root.quit)


    root.state("zoomed")


    GUI(parent=root)


    root.mainloop()



if __name__ == "__main__":
    main()

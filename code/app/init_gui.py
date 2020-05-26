import tkinter as tk
import os
from pathlib import Path
from PIL import Image
from PIL import ImageTk
import sys
import init_paths

"""
    Grafine programele nuotrauku atpazinimui.
"""

init_paths.init_sys_folders()

import or_cnn
import mgi_cnn
import vgg16_cnn

# data_navigation
path = os.path.join(Path(Path(os.getcwd()).parent).parent, 'data', 'pics', 'candidate_0_subset_9_class_0.tiff')
#

# teksto stringai:
title_text = 'Plaučių nuotraukų klasifikavimas'
pic_comment_text = 'Rodoma atsitiktinė nuotrauka'
found_class_text = 'Nustatyta klasė: '
real_class_text = 'Tikroji klasė: '
positive_text = 'teigiama.'
negative_text = 'neigiama.'
open_button_text = 'Rasti nuotrauką diske'
random_button_text = 'Atsitiktinė nuotrauka'
testing_mnist_text = 'Tinklai testuojami naudojant MNIST rašto ženklus . . .'
result_text = 'Rezultatai: '
original_cnn_text = 'Tradicinis konv. tinklas: '
mask_r_cnn_text = 'MGI CNN tinklas: '
vgg16_cnn_text = 'VGG16 konv. tinklas: '
mgi_cnn_text = 'MGI-CNN tinklas: '
newline = '\n'


# wrapper funkcijos (kad galima butu jas susieti su tkinter)
def test_all_networks_wrap():
    console.delete(1.0, tk.END)
    console.insert(tk.END, testing_mnist_text)
    console.delete(1.0, tk.END)
    console.insert(tk.END, result_text + newline)
    console.insert(tk.END, original_cnn_text + newline)
    console.insert(tk.END, mask_r_cnn_text + newline)
    console.insert(tk.END, vgg16_cnn_text + newline)
    console.insert(tk.END, mgi_cnn_text + newline)


def train_original_cnn_wrap():
    or_cnn = original_cnn.OriginalCnn()
    or_cnn.train()


def train_mask_r_cnn_wrap():
    print('todo')


def train_mgi_cnn_wrap():
    print('todo')


def train_vgg16_cnn_wrap():
    print('todo')


# tk aplikacijos inicializavimas
root = tk.Tk()

# lango pozicijos (naudojama lango tempimui)
last_click_x = 0
last_click_y = 0


def change_on_hovering(event):
    global close_button
    close_button['bg'] = 'red'


def return_to_normal_state(event):
    global close_button
    close_button['bg'] = '#373837'


def save_last_click_pos(event):
    global last_click_x, last_click_y
    last_click_x = event.x
    last_click_y = event.y


def dragging(event):
    x = event.x - last_click_x + root.winfo_x()
    y = event.y - last_click_y + root.winfo_y()
    root.geometry("+%s+%s" % (x, y))


# operacines sistemos sugeneruotos lango titulines juostos paslepimas ir naujos juostos sukurimas
root.overrideredirect(True)
root.geometry('550x450+200+200')
root.configure(background='#373837')    # gray background

title_bar = tk.Frame(root, bg='#373837', relief='raised', bd=2, highlightthickness=0)
close_button = tk.Button(title_bar, text='X', command=lambda: sys.exit(), bg='#373837', highlightthickness=0, padx=5)
window = tk.Canvas(root, bg='black', bd=0, highlightthickness=0)

# paveikslo dydzio pakeitimas
photo = ImageTk.PhotoImage(Image.open(path).resize((250, 250), Image.ANTIALIAS))

# GUI elementu surisimas su langu
title_bar.pack(expand=0, fill='x')
close_button.pack(side='right')
window.pack(expand=1, fill='both', padx='10', pady='10')
title_bar.bind('<Button-1>', save_last_click_pos)
title_bar.bind('<B1-Motion>', dragging)
close_button.bind('<Enter>', change_on_hovering)
close_button.bind('<Leave>', return_to_normal_state)

# pavadinimo pridejimas titulinei juostai
title = tk.Label(title_bar, text=title_text, bg='#373837', fg='white', font='none 10')
title.pack()

# atsitiktines nuotraukos rodymas
tk.Label(window, image=photo, bg="black").grid(row=1, column=1, rowspan=3, sticky='W')
tk.Label(window, text=pic_comment_text, bg='black', fg='white', font='none 10')\
    .grid(row=4, column=1)

# konsoles pridejimas langui
console = tk.Text(window, bg='#373837', height='5', width='60', state='normal') #.grid(row=5, column=1, columnspan=4)
console.grid(row=5, column=1, columnspan=4)


#tk.Label(window, text='test_label', bg='black', fg='white', font='none 10').grid(row=1, column=2, sticky='W')
#tk.Button(window, text='test', command=test_all_networks_wrap, bg='yellow', highlightthickness=0, padx=5)\
#    .grid(row=1, column=2)
# original conv:
tk.Button(window, text='or_cnn', command=train_original_cnn_wrap, bg='yellow', highlightthickness=0, padx=5)\
    .grid(row=1, column=2)
# mask r cnn:
tk.Button(window, text='mask_r', command=train_mask_r_cnn_wrap, bg='yellow', highlightthickness=0, padx=5)\
    .grid(row=1, column=3)
# mgi cnn:
tk.Button(window, text='mgi_cnn', command=train_mgi_cnn_wrap, bg='yellow', highlightthickness=0, padx=5)\
    .grid(row=2, column=2)
# vgg16 cnn:
tk.Button(window, text='vgg16_cnn', command=train_vgg16_cnn_wrap, bg='yellow', highlightthickness=0, padx=5)\
    .grid(row=2, column=3)

# pseudo-parasciu pridejimas tinkleliui
window.grid_columnconfigure(0, minsize=20)
window.grid_rowconfigure(0, minsize=20)

root.mainloop()

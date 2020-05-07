import tkinter as tk
import os
from pathlib import Path
from PIL import Image
from PIL import ImageTk
import sys

# temp
path = os.path.join(Path(Path(os.getcwd()).parent).parent, 'data', 'pics', 'candidate_0_subset_7_class_0.tiff')
#

# text strings - LT:
title_text = 'Plaučių nuotraukų klasifikavimas'
pic_comment_text = 'Rodoma atsitiktinė nuotrauka'
found_class_text = 'Nustatyta klasė: '
real_class_text = 'Tikroji klasė: '
positive_text = 'teigiama.'
negative_text = 'neigiama.'
open_button_text = 'Rasti nuotrauką diske'
random_button_text = 'Atsitiktinė nuotrauka'

# init tk app
root = tk.Tk()

# root position for window dragging
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


# hide default title bar and add a new one
root.overrideredirect(True)
root.geometry('550x450+200+200')
root.configure(background='#373837')    # gray background

title_bar = tk.Frame(root, bg='#373837', relief='raised', bd=2, highlightthickness=0)
close_button = tk.Button(title_bar, text='X', command=lambda: sys.exit(), bg='#373837', highlightthickness=0, padx=5)
window = tk.Canvas(root, bg='black', bd=0, highlightthickness=0)

# resize image to be bigger on canvas
photo = ImageTk.PhotoImage(Image.open(path).resize((250, 250), Image.ANTIALIAS))

# pack widgets & bind controls
title_bar.pack(expand=0, fill='x')
close_button.pack(side='right')
window.pack(expand=1, fill='both', padx='10', pady='10')
title_bar.bind('<Button-1>', save_last_click_pos)
title_bar.bind('<B1-Motion>', dragging)
close_button.bind('<Enter>', change_on_hovering)
close_button.bind('<Leave>', return_to_normal_state)

# add title to title bar
title = tk.Label(title_bar, text=title_text, bg='#373837', fg='white', font='none 10')
title.pack()

# add random pic to window
tk.Label(window, image=photo, bg="black").grid(row=1, column=1, rowspan=3, columnspan=2)
tk.Label(window, text=pic_comment_text, bg='black', fg='white', font='none 10')\
    .grid(row=4, column=1, columnspan=2)

# add console to the window
console = tk.Text(window, bg='#373837', height='5', width='60').grid(row=5, column=1)

# add pseudo-margins to grid
col_count, row_count = window.grid_size()
for i in range(0, col_count):
    window.grid_columnconfigure(i, minsize=20)
    window.grid_rowconfigure(i, minsize=20)

root.mainloop()

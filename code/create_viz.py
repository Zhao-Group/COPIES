
#Modules
import numpy as np
import pandas as pd
import math, sys
from collections import defaultdict
from bokeh.plotting import figure, output_file, show, save
#from bokeh.io import output_notebook
#from bokeh.resources import INLINE
from bokeh.models import ColumnDataSource, HoverTool, BasicTickFormatter

#Datas
df = pd.read_csv('data/s288c/output.csv', dtype={"Chromosome": "string"})
df["Chromosome"] = df["Chromosome"].fillna("nan")

#Processing
chr_names = np.unique(list(df['Chromosome']))
acc_ids = np.unique(list(df['Accession']))

#If chromosome names are available correctly, use those instead of accession IDs.
len_list = []
if len(chr_names) == len(acc_ids) and not 'nan' in chr_names:
    for i in range(len(chr_names)):
        len_list.append(df.loc[df['Chromosome'] == chr_names[i], 'Chromosome Length'].iloc[0])
        
    chr_df = pd.DataFrame(list(zip(chr_names, len_list)), columns=['ID','Chromosome Length'])
    label_to_use = 'Chromosome'
    
else:
    for i in range(len(acc_ids)):
        len_list.append(df.loc[df['Accession'] == acc_ids[i], 'Chromosome Length'].iloc[0])
        
    chr_df = pd.DataFrame(list(zip(acc_ids, len_list)), columns=['ID','Chromosome Length'])
    label_to_use = 'Accession'

y = list(chr_df['ID'])
y_pos = np.arange(len(y)) + 0.5
y_dict = {y_pos[i]: y[i] for i in range(len(y_pos))}


p = figure(title = "Neutral Integration Sites", sizing_mode="stretch_both", y_range = y)
  
#plotting the graph
p.hbar(y_pos, right = len_list, height = 0.6, alpha = 0.2)

guide_spec = defaultdict(list)
for i in range(len(y)):
    curr_df = df.loc[df[label_to_use] == y[i]][['ID','Guide Sequence','Location','Left Gene','Right Gene']].reset_index(drop=True)
    guide_ids = list(curr_df['ID'])
    site_loc = list(curr_df['Location'])
    guide_seq = list(curr_df['Guide Sequence'])
    left_gene = list(curr_df['Left Gene'])
    right_gene = list(curr_df['Right Gene'])
    for j in range(len(site_loc)):
        c_y = y_pos[i]
        c_x = site_loc[j]
        c_guide_ids = guide_ids[j]
        c_guide = guide_seq[j]
        c_left_gene = left_gene[j]
        c_right_gene = right_gene[j]

        c_ymin = c_y - 0.25
        c_ymax = c_y + 0.25
        
        guide_spec["X"].append([c_x,c_x])
        guide_spec["Y"].append([c_ymin,c_ymax])
        guide_spec['ID'].append(c_guide_ids)
        guide_spec['Guide'].append(c_guide)
        guide_spec['Genes'].append(c_left_gene + ',' + c_right_gene)
        
source = ColumnDataSource(guide_spec)

loc_line = p.multi_line(xs='X', ys='Y', line_color='red', source=source)
loc_hover = HoverTool(renderers=[loc_line], tooltips=[("ID","@ID"),("Guide", "@Guide"),("Surrounding Genes", "@Genes")])
p.add_tools(loc_hover)
        
p.grid.visible = False
p.xaxis.axis_label = 'Length'
p.yaxis.axis_label = 'ID'
p.xaxis.axis_label_text_font_style = 'normal'
p.yaxis.axis_label_text_font_style = 'normal'
p.title.align = 'center'
p.toolbar.logo = None
p.xaxis.formatter = BasicTickFormatter(use_scientific=False)

output_file('data/s288c/copies_visualization.html')
save(p)

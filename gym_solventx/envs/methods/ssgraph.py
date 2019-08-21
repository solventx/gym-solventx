import time, os, sys
import numpy               as np
from   io                  import StringIO

from graphviz import Digraph

def create_graph(obj=None, render=False, filename='ss_graph'):
  '''
    Creates graph from obj variable

    @params
    render:   if true creates graph and renders in default pdf viewer.
              if previously rendered file has not been closed, render does nothing
    
    filename: name of file to save graph

  '''
  if obj: 
    graph = Digraph('graphs/' + filename, node_attr={'shape': 'plaintext'})
    graph.attr(rankdir='TB', size='8,5')

    purity = obj.strip_pur[0]
    recovery = obj.strip_recov[0]
    rec_color = get_color(recovery)

    aq_concs     = []
    org_concs    = []
    output_concs = []
    nd_indices   = (4, 8) #aq, org
    cols         = list(map(int, obj.variables[-3:]))
    sum_stages   = [] #sum of stages up to column index
    for i in range(len(cols)):
      sum_stages.append(sum(cols[:i+1]))

    output_stages = [] #stage indices for input feeds
    for i, val in enumerate(sum_stages):
      output_stages.append(val+i)
    
    org_dup_stages = [0] #duplicate organic stage conc indices//org feed into next col
    for i, val in enumerate(cols):
      org_dup_stages.append(val+org_dup_stages[i]+1)
    org_dup_stages.pop()

    stages      = obj.y.reshape(obj.y.shape[0]//10, -1).tolist() #10 is size of aq and org arrays
    for i, stage in enumerate(stages):
      aq  = stage[nd_indices[0]]
      org = stage[nd_indices[1]]

      aq_concs.append(aq)
      org_concs.append(org)
    
    output_concs += [aq_concs.pop(index) for index in output_stages[::-1]][::-1] #remove output feeds
    [org_concs.pop(index) for index in org_dup_stages[::-1]] #remove dup org concs

    rev_aq_conc = []
    rev_org_conc = []
    for col_num, stages in enumerate(sum_stages):
      if col_num == 0:
        rev_aq_conc += aq_concs[0:stages][::-1]
        rev_org_conc += org_concs[0:stages][::-1]
      else:
        rev_aq_conc += aq_concs[sum_stages[col_num-1]:stages][::-1]
        rev_org_conc += org_concs[sum_stages[col_num-1]:stages][::-1]

    aq_concs = rev_aq_conc
    org_concs = rev_org_conc
    
    all_concs = aq_concs + org_concs + output_concs
    minVal, maxVal = min(all_concs), max(all_concs)

    norm_aq = []
    norm_org = []
    norm_out = []
    #normalize data
    for i, val in enumerate(aq_concs):
      norm_aq.append(normalize([minVal, val, maxVal], 0, 1)[1])
    for i, val in enumerate(org_concs):
      norm_org.append(normalize([minVal, val, maxVal], 0, 1)[1])
    for i, val in enumerate(output_concs):
      norm_out.append(normalize([minVal, val, maxVal], 0, 1)[1])

    for i, col in enumerate(cols): #create table
      output_conc = output_concs.pop(0)
      norm_output_conc = norm_out.pop(0)
      with graph.subgraph() as dot:
        dot.attr(rank='same')
        rows = ''
        for j in range(col): #create rows
          aq  = aq_concs.pop(0)
          org = org_concs.pop(0)
          norm_aq_conc = norm_aq.pop(0)
          norm_org_conc = norm_org.pop(0)

          if j == 0:
            rows+=f'<TR>\
              {add_stage(i, org, norm_org_conc, "Org", "begin")}\
              {add_stage(i, aq, norm_aq_conc, "Aq", "begin")}\
            </TR>'
          elif j == col-1:
            rows+=f'<TR>\
              {add_stage(i, org, norm_org_conc, "Org", "end")}\
              {add_stage(i, aq, norm_aq_conc, "Aq", "end")}\
            </TR>'
          else:
            rows+=f'<TR>\
              {add_stage(i, org, norm_org_conc, "Org")}\
              {add_stage(i, aq, norm_aq_conc, "Aq")}\
            </TR>'

        #Center Graph
        dot.node(f'struct{i}', f'''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
          {rows}
        </TABLE>>''', fillcolor='lightblue')

        new_line = '\n'
        #Labels
        dot.node(f'col_{i}', f'''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
          f'<TR>
              <TD PORT="col_name{i}" BGCOLOR="{get_color(norm_output_conc)}">{get_exit_row(i, output_conc, recovery, purity)}</TD>
            </TR>'
            

          f'<TR>
            <TD PORT="col_name_input{i}">Input Feed</TD>
          </TR>'
        </TABLE>>''')

    structs = ['struct0', 'struct1', 'struct2'] #extract, scrub, strip
    #org
    graph.edges([
      ('struct0:end0_org', 'struct1:begin1_org'), 
      ('struct1:end1_org', 'struct2:begin2_org'), 
      ('struct2:end2_org', 'struct0:begin0_org')
    ])

    #neatness
    invis_edges = [
      ('struct2:begin2_aq', 'struct1:end1_aq'), 
      ('struct1:begin1_aq', 'struct0:end0_aq')
    ]
    for line in invis_edges:
      graph.edge(line[0], line[1], style='invis')
    
    #aq
    graph.edges([
      ('struct2:begin2_aq', 'col_2:col_name2'), 
      ('struct1:begin1_aq', 'col_1:col_name1'), 
      ('struct0:begin0_aq', 'col_0:col_name0')
    ])

    #input flows
    graph.edges([
      ('col_0:col_name_input0', 'struct0:end0_aq'), 
      ('col_1:col_name_input1', 'struct1:end1_aq'), 
      ('col_2:col_name_input2', 'struct2:end2_aq')
    ])

    #supress output to console
    actualstdout = sys.stdout
    sys.stdout   = StringIO()
    
    directory = os.getcwd() + '/output/graphs'
    graph.save(filename=filename, directory=directory)
    graph.render(renderer=None, format='pdf', cleanup=True, quiet=True) #save graph
    if render:
      try:
        graph.view(cleanup=True) #show graph
      except: #if file has not been closed
        pass

    sys.stdout = actualstdout

#rgb = (r, g, b)
def rgb_to_hex(rgb):
  if len(rgb) == 3:
    valid = True
    for val in rgb:
      if not(val != None and type(val) == int and 0<=val<=255):
        valid = False
        break
        
    if valid:
      as_hex = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
    else: #invalid values
      as_hex = '#000000'
    
    return as_hex

  #normalizes a set of data between a range
def normalize(data, rangeMin, rangeMax):
  if(rangeMin>rangeMax):
    raise 'Invalid Ranges'
  newVals = []
  maxVal=max(data)
  minVal=min(data)
  for val in data:
    if maxVal-minVal == 0:
      newVals.append(rangeMin)
    else: 
      newVals.append((rangeMax-rangeMin)*(val-minVal)/(maxVal-minVal)+rangeMin)
  return newVals

def add_stage(col_num, conc, norm_conc, type_cell, stage_type=None):
  if stage_type:
    string = f'<TD PORT="{stage_type}{col_num}_{type_cell.lower()}" BGCOLOR="{get_color(norm_conc)}">{type_cell}: %.2E</TD>' % conc
  else:
    string = f'<TD BGCOLOR="{get_color(norm_conc)}">{type_cell}: %.2E</TD>' % conc

  return string

def get_color(conc):
  return f'{rgb_to_hex((255, int(conc*255), 0))}'

def get_exit_row(i, conc, recovery, purity):
  if i == 2:
    return f'Strip<BR/>Conc: %.2E <BR/>Purity: %.3f<BR/>Recovery: %.3f' % (conc, purity, recovery)
  else:
    return f'{["Extraction:", "Scrub:"][i]}<BR/>Conc: %.2E' % conc
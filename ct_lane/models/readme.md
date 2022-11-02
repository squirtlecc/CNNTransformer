### desc
model deal with data in net.detector function. in this function give a pic ,return a dict,include pred and loss(caculate loss in this function)
and main model running step by step: 
origin_img 
-> backbone : get multi-scale feature from img
-> necks: fusion feature from backbone
-> cores: deep processing some feature (from backbone or neck)
-> decoder: resize the freature to output shape (some decoder using fusion to concat necks feature or backbone feature)
-> losses: compose result and gt_mask to get loss for backpropagation

> this all step not necessarilly do all, but must in order.


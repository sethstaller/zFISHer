'''
Stores paremeterss of zFISHer analysis set
'''


seg_nuc_count = None

nuc_polygons = None




# TEMP DATA FOR PUNCTAPICK INTEGRATION
keypoints_array = [0,0]



# ROI picking export for calculations and excel writing
kpchan_kpnuc_xbundle = []
kpchan_ROIradius_xbundle = []
kpchan_coloc_xbundle = []
kpchan_analysistoggle_xbundle = []
arrows_xbundle = [-1,-1,-1,-1,-1,-1,-1,-1]

f1_full_exp_arr = [] #file_id,channel,chanrad,chan_e,kpID,kpOvalID,f_kp_x,f_kp_y,nucInd,polyID,polyTextID
f2_full_exp_arr = []

kpchan_kpnucxyz_xbundle = [] #file_id,channel,chanrad,chan_e,kpID,kpOvalID,kp_x,kp_y,nucInd,polyID,polyTextID, max_slice_index, max_intensity

yes_arr = []
no_arr = []
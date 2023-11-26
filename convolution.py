import torch
import MDAnalysis as mda
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch

class FrameConvolution:
    def __init__(self,universe, frame_number):
        self.universe = universe
        self.frame_number = frame_number
        csv_path = f"frame{str(self.frame_number)}.csv"
        try:
            self.df = pd.read_csv(csv_path, index_col=False)
        except FileNotFoundError:
            self.df = self.gen_frame_df()
        self.x_dim = self.universe.trajectory[0].dimensions[0]
        self.y_dim = self.universe.trajectory[0].dimensions[1]
        self.z_dim = self.universe.trajectory[0].dimensions[2] 
        
    def gen_frame_df(self):
        atom_radius_mapping = {'C': .914, 'H': .5, 'N': .92, "O": .73}
        data_dict = {'AtomID': [], "AtomName":[], 'AtomType': [], "ResType": [], "ResID": [], 'X': [], 'Y': [], 'Z': []}
        for atom in self.universe.atoms:
            #print(atom)
            data_dict['AtomID'].append(atom.id)
            data_dict["AtomName"].append(atom.name)
            data_dict["AtomType"].append(atom.type)
            data_dict["ResType"].append(atom.resname)
            data_dict["ResID"].append(atom.resid)
        for j in range(len(self.universe.atoms)):
            data_dict['X'].append(self.universe.trajectory[self.frame_number][j][0])
            data_dict['Y'].append(self.universe.trajectory[self.frame_number][j][1])
            data_dict['Z'].append(self.universe.trajectory[self.frame_number][j][2])
        df = pd.DataFrame(data_dict)
        df['Radius'] = df['AtomType'].map(atom_radius_mapping)
        csv_path = f"frame{str(self.frame_number)}.csv"
        print(csv_path)
        df.to_csv(csv_path,index=False)
        return df
        
    def get_dimension(self, travel_axis):
        if travel_axis == "X":
            travel_axis_dim = self.x_dim
        if travel_axis == "Y":
            travel_axis_dim = self.y_dim
        if travel_axis == "Z":
            travel_axis_dim = self.z_dim
        return travel_axis_dim
        
    def filter_df(self, travel_axis, travel_axis_val):
        #Were filtering out all the rows whose z(or rather, whose travel_axis) value
        #is not within the largest radius of all atoms. This is so that we can reduce
        #subesequent computation times.
        max_r = max(set(self.df.Radius)) 
        travel_axis_low = travel_axis_val-max_r
        travel_axis_high = travel_axis_val+max_r
        filtered_df = self.df[(self.df[travel_axis] >= travel_axis_low) & (self.df[travel_axis] <= travel_axis_high)]
        return filtered_df
    
    def calc_conc(self, travel_axis, travel_axis_val, travel_axis_thickness):
        # Initialize variables for total area and areas of each residue
        area_unk = 0
        area_sol = 0
        #This makes a df that has all atoms to far away to project into the slice out. This is NOT self.df!!!
        #Don't get confused.
        df = self.filter_df(travel_axis, travel_axis_val)
        # Iterate through each atom in the DataFrame
        for index, atom in df.iterrows():
            # Calculate the radius for the slice at Z(or rather, the travel axis
            # The atom projects into the slice
            r_proj_squared = atom['Radius']**2 - (atom[travel_axis] - travel_axis_val)**2
            if r_proj_squared <= 0:
                #print("invalid projection")
                #Move on if we get a non-positive value. Some atoms are within the MAX atom radius of the slice location
                #but their radius isn't large enough to project into it.
                continue
            radius = np.sqrt(atom['Radius']**2 - (atom[travel_axis] - travel_axis_val)**2)
            
            # Calculate the area of the circle using A = π * r^2
            #This aspect of the code won't translate to other systems
            #TODO modify to implement a general case
            if radius > 0:
                area = np.pi * radius**2
                # Add the area to the corresponding residue
                if atom['ResType'] == 'UNK':
                    area_unk += area
                elif atom['ResType'] == 'SOL':
                    area_sol += area
        
        #Objects aren't subscriptable unfortunately so I need these stupid if then clauses to extract the
        #orthoganol axis dimensions.
        if travel_axis == "X":
            orth_dim_1 = self.y_dim
            orth_dim_2 = self.z_dim
        if travel_axis == "Y":
            orth_dim_1 = self.x_dim
            orth_dim_2 = self.z_dim
        if travel_axis == "Z":
            orth_dim_1 = self.x_dim
            orth_dim_2 = self.y_dim
        
        total_area = orth_dim_1*orth_dim_2
        total_vol = total_area * travel_axis_thickness #i.e, delta z

        #This aspect of the code won't translate to other systems
        #TODO modify to implement a general case
        nic_vol = area_unk * travel_axis_thickness
        sol_vol = area_sol * travel_axis_thickness
        # Print the calculated areas
        vol_d = {"Total":total_vol, "Nicotine":nic_vol,"Solvent":sol_vol}
        #print(f"Total Area: {total_area}")
        #print(f"Area of UNK: {area_unk}")
        #print(f"Area of SOL: {area_sol}")
        return vol_d
    
    def plot_slice(df, z_slice,x_dim,y_dim):
        fig, ax = plt.subplots()
        # Initialize variables for total area and areas of each residue
        total_area = 0
        area_unk = 0
        area_sol = 0
        
        # Iterate through each atom in the DataFrame
        for index, atom in df.iterrows():
            # Calculate the radius for the slice at Z
            r_proj_squared = atom['Radius']**2 - (atom['Z'] - z_slice)**2
            if r_proj_squared < 0:
                #print("invalid projection")
                continue
            radius = np.sqrt(atom['Radius']**2 - (atom['Z'] - z_slice)**2)
            
            # Map different face colors based on Res
            #face_color = 'r' if atom['ResType'] == 'UNK' else 'b'
            if atom.AtomType == "C":
                face_color = "k"
            if atom.AtomType == "H":
                face_color = "w"
            if atom.AtomType == "O":
                face_color = "r"
            if atom.AtomType == "N":
                face_color = "b"
            
            # Plot the circle with different face color for each ResID
            circle = plt.Circle((atom['X'], atom['Y']), radius, edgecolor='k', facecolor=face_color, alpha=0.5)
            ax.add_patch(circle)
        
            # Calculate the area of the circle using A = π * r^2
            if radius > 0:
                area = np.pi * radius**2
                # Add the area to the corresponding residue
                if atom['ResType'] == 'UNK':
                    area_unk += area
                elif atom['ResType'] == 'SOL':
                    area_sol += area
    
        # Set plot limits
        ax.set_xlim(0,x_dim)
        ax.set_ylim(0,y_dim)
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Show the plot
        plt.show()
        
    def create_conc_df(self, travel_axis, bins):
        data_dict = {travel_axis: [], "NicConc":[], 'SolConc': [] }

        travel_axis_dim = self.get_dimension(travel_axis)
            
        travel_axis_thickness = travel_axis_dim / bins #i.e, delta Z 
        
        for i in range(bins):
            travel_axis_val = i * travel_axis_thickness
            #filtered_df = filter_df(d_travel,z)
            vol_d = self.calc_conc(travel_axis, travel_axis_val, travel_axis_thickness)
            nic_conc = vol_d["Nicotine"]/vol_d["Total"]
            sol_conc = vol_d["Solvent"]/vol_d["Total"]
            data_dict[travel_axis].append(travel_axis_val)
            data_dict["NicConc"].append(nic_conc)
            data_dict["SolConc"].append(sol_conc)
        conc_df = pd.DataFrame(data_dict)
        return conc_df
    
    def auto_corr(self, conc_tensor, travel_axis_dim, d):
        mean_conc = torch.mean(conc_tensor)
        #print(f"DEBUG: conc tensor: {conc_tensor}")
        dist_per_bin = travel_axis_dim / len(conc_tensor)  
    
        # Calculate the ratio between the reference index 'd' and the original z-value per index
        z_ratio = int(1/dist_per_bin) #if Z is the axis of travel
        
        # Calculate the new indices after re-referencing
        #print(d * z_ratio)
        new_indices = (torch.arange(len(conc_tensor)) - d * z_ratio) % len(conc_tensor)
        #print(f"DEBUG Indices: {new_indices}")
        # Use the new indices to create the re-referenced tensor
        re_referenced_tensor = conc_tensor[new_indices]
        delta_conc_z = conc_tensor - mean_conc
        delta_conc_zminusd = re_referenced_tensor - mean_conc
        c_aa = torch.dot(delta_conc_z, delta_conc_zminusd)
        return float(c_aa)

    def plot_histogram(self,travel_axis, bins):
        caa_d = {"d":[],"c_aa": []}
        travel_axis_dim = self.get_dimension(travel_axis)
        conc_df = self.create_conc_df(travel_axis, bins)
        conc_tensor = torch.tensor(np.array(conc_df.NicConc))
        travel_midpoint = travel_axis_dim/2
        for d in range(int(-1*travel_midpoint),int(travel_midpoint+1)):
            #print(d)
            c_aa =self.auto_corr(conc_tensor, travel_axis_dim, d)
            caa_d["d"].append(d)
            caa_d["c_aa"].append(c_aa)
        pd.DataFrame(caa_d).plot(kind="scatter", x="d",y="c_aa")
        #return pd.DataFrame(caa_d)
        return None

    def create_autocorr_df(self,travel_axis,bins):
        caa_d = {"d":[],"c_aa": []}
        travel_axis_dim = self.get_dimension(travel_axis)
        conc_df = self.create_conc_df(travel_axis, bins)
        conc_tensor = torch.tensor(np.array(conc_df.NicConc))
        travel_midpoint = travel_axis_dim/2
        for d in range(int(-1*travel_midpoint),int(travel_midpoint+1)):
            #print(d)
            c_aa =self.auto_corr(conc_tensor, travel_axis_dim, d)
            caa_d["d"].append(d)
            caa_d["c_aa"].append(c_aa)
        return pd.DataFrame(caa_d)
        
def main():
    pass
if __name__ == "__main__":
    main()
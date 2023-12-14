#--------------------------------------------------------------------------------------------------------------------
#Imports-------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
import torch
import MDAnalysis as mda
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

#--------------------------------------------------------------------------------------------------------------------
#Class Definitions---------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
class UniverseConvolution:
    def __init__(self, conf_path, traj_path):
        self.universe = mda.Universe(conf_path,traj_path)
    
    def create_shifted_conc_df(self):
        pass
    





class FrameConvolution:
    #----------------------------------------------------------------------------------------------------------------
    #Special methods-------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------
    def __init__(self,universe, frame_number):
        self.universe = universe
        self.frame_number = frame_number
        csv_path = f"frame{str(self.frame_number)}.csv"
        try:
            self.df = pd.read_csv(csv_path, index_col=False)
        except FileNotFoundError:
            self.df = self.gen_frame_df()
        self.molecules = set(self.df.ResType)
        self.x_dim = self.universe.trajectory[0].dimensions[0]
        self.y_dim = self.universe.trajectory[0].dimensions[1]
        self.z_dim = self.universe.trajectory[0].dimensions[2]
        #self.animation_fig, self.animation_ax = plt.subplots()
        
    #----------------------------------------------------------------------------------------------------------------
    #Frame DF Generation---------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------     
    def gen_frame_df(self):
        atom_radius_mapping = {'C': .914, 'H': .5, 'N': .92, "O": .73, "A":1 , "B":1 }
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
        
    #----------------------------------------------------------------------------------------------------------------
    #General Helper Methods------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------------------------    
    #Analysis Methods------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------
    def create_conc_df(self, travel_axis = "Z", bins = 1000):
        #Initialize the data dictionary to store the concentration with the residue concentrations
        data_dict = {travel_axis: []}
        for molecule in self.molecules:
            data_dict[f"{molecule}Conc"] = []
        #data_dict = {travel_axis: [], "NicConc":[], 'SolConc': [] }
        
        
        travel_axis_dim = self.get_dimension(travel_axis) 
        travel_axis_thickness = travel_axis_dim / bins #i.e, delta Z 
        for i in range(bins):
            travel_axis_val = i * travel_axis_thickness
            #filtered_df = filter_df(d_travel,z)
            
            
            #vol_d in format: vol_d = {"Total":total_vol, "Nicotine":nic_vol,"Solvent":sol_vol}
            vol_d = self.calc_conc(travel_axis, travel_axis_val, travel_axis_thickness)
            
            data_dict[travel_axis].append(travel_axis_val)
            for molecule in self.molecules:
                mol_conc = vol_d[molecule]/vol_d["Total"]
                #sol_conc = vol_d["Solvent"]/vol_d["Total"]
                data_dict[f"{molecule}Conc"].append(mol_conc)
                
        conc_df = pd.DataFrame(data_dict)
        return conc_df    
    
    def plot_caa_scatter(self,travel_axis = "Z", molecule_conc="NicConc", bins=1000):
        df = self.create_autocorr_df(travel_axis, molecule_conc, bins)
        df.plot(kind="scatter", x="d",y="c_aa")
        return None
    #The autocorrelator function is the convlolutional method we're applying   
    def create_autocorr_df(self,travel_axis = "Z",molecule_conc = "NicConc", bins = 1000):
        caa_d = {"d":[],"c_aa": []}
        travel_axis_dim = self.get_dimension(travel_axis)
        conc_df = self.create_conc_df(travel_axis, bins)
        conc_tensor = torch.tensor(np.array(conc_df[molecule_conc]))
        travel_midpoint = travel_axis_dim/2
        for d in range(int(-1*travel_midpoint),int(travel_midpoint+1)):
            #print(d)
            c_aa =self.auto_corr(conc_tensor, travel_axis_dim, d)
            caa_d["d"].append(d)
            caa_d["c_aa"].append(c_aa)
        return pd.DataFrame(caa_d)
        
    def plot_conc_scatter(self, travel_axis="Z", molecule_conc="NicConc",bins=1000):
        df = self.create_conc_df(travel_axis, bins)
        df.plot(kind="scatter",x=travel_axis, y=molecule_conc)
    
    def create_pulse_df(self,travel_axis="Z", molecule_conc="NicConc", bins=1000):
        df = self.create_autocorr_df(travel_axis, molecule_conc,bins)
        d = {"d":[],"pulse":[]}
        for row in df.T:
            d["d"].append(df.iloc[row].d)
            if df.iloc[row].c_aa > 0:
                d["pulse"].append(1)
            else:
                d["pulse"].append(0)
        return pd.DataFrame(d)
    
    def plot_pulse_scatter(self,travel_axis = "Z", molecule_conc="NicConc", bins=1000):
        df = self.create_pulse_df(travel_axis, molecule_conc, bins)
        df.plot(kind="scatter", x="d",y="pulse")
        return None
    
    def create_Xc_df(self,travel_axis = "Z",molecule_conc = "NicConc", bins = 1000):
        pulse_df = self.create_pulse_df(travel_axis, molecule_conc)
        conc_df = self.create_conc_df(travel_axis=travel_axis, bins=len(pulse_df))
        pulse_tensor = torch.tensor(np.array(pulse_df.pulse), dtype=torch.double)
        conc_tensor = torch.tensor(np.array(conc_df[molecule_conc]))
        
        Xc_d = {"n":[],"Xc": []}
        travel_axis_dim = self.get_dimension(travel_axis)
        travel_midpoint = travel_axis_dim/2
        for n in range(len(pulse_df)):
            #print(d)
            X_c =self.X_c(n, conc_tensor, pulse_tensor)
            Xc_d["n"].append(n)
            Xc_d["Xc"].append(X_c)
        return pd.DataFrame(Xc_d)
    
    def plot_Xc_scatter(self, travel_axis="Z", molecule_conc="NicConc",bins=1000):
        df = self.create_Xc_df(travel_axis, molecule_conc, bins)
        df.plot(kind="scatter", x="n",y="Xc")
        return None
    
    """    
    def create_shifted_conc_df(self,travel_axis = "Z",molecule_conc = "NicConc", bins = 1000):
        conc_df = self.create_conc_df(travel_axis, bins)
        Xc_df = self.create_Xc_df(travel_axis,molecule_conc, bins)
        n_max = int(Xc_df[Xc_df.Xc == Xc_df.Xc.max()].n)
        print(f"DEBUG n_max:{n_max}")
        conc_df["Z"] = conc_df["Z"].apply(lambda x: x + n_max)
        return conc_df
    """
    
    def create_shifted_conc_df(self, travel_axis="Z", molecule_conc="NicConc", bins=1000):
        conc_df = self.create_conc_df(travel_axis, bins)
        Xc_df = self.create_Xc_df(travel_axis, molecule_conc, bins)
        n_max = int(Xc_df[Xc_df.Xc == Xc_df.Xc.max()].n)
        print(f"DEBUG n_max: {n_max}")
        n_max = -n_max
        max_z = conc_df[travel_axis].max()
        conc_df["Z"] = conc_df["Z"].apply(lambda x: (x + n_max) % (max_z + 1))

        return conc_df    
    
    
    
    
    def plot_shifted_conc_scatter(self, travel_axis="Z", molecule_conc="NicConc",bins=1000):
        df = self.create_shifted_conc_df(travel_axis, molecule_conc, bins)
        df.plot(kind="scatter", x=travel_axis,y=molecule_conc)
        return None
        
    #Analysis Helper Methods-----------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------
    def calc_conc(self, travel_axis, travel_axis_val, travel_axis_thickness):
        # Initialize variables for total area and areas of each residue
        area_d = {}
        #This makes a df that has all atoms that are too far away to project into the slice filtered out. 
        #This is NOT self.df!!!
        #Don't get confused.
        df = self.filter_df(travel_axis, travel_axis_val)
        
        for residue in set(self.df.ResType):
            area_d[residue]=0
        
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
                area_d[atom['ResType']] += area
                
               #if atom['ResType'] == 'UNK':
                #    area_unk += area
                #elif atom['ResType'] == 'SOL':
                #    area_sol += area
        
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
        vol_d = {"Total":total_vol}
        for res in area_d:
            vol_d[res] = area_d[res] * travel_axis_thickness
        #nic_vol = area_unk * travel_axis_thickness
        #sol_vol = area_sol * travel_axis_thickness
        # Print the calculated areas
        #vol_d = {"Total":total_vol, "Nicotine":nic_vol,"Solvent":sol_vol}
        #print(f"Total Area: {total_area}")
        #print(f"Area of UNK: {area_unk}")
        #print(f"Area of SOL: {area_sol}")
        return vol_d
    
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
        
    def X_c(self,n, conc_tensor, pulse_tensor):
        shifted_tensor = pulse_tensor.roll(n)
        dot_prod = torch.dot(conc_tensor, shifted_tensor)
        return float(dot_prod)
        
    #----------------------------------------------------------------------------------------------------------------    
    #Visualization Methods-------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------   

    #TODO Finish this such that I can actually make an animation
    def create_animation(self, travel_axis, bins):
        # Set up the animation
        travel_axis_dim = self.get_dimension(travel_axis)
        travel_axis_step = travel_axis_dim / bins  # i.e., delta Z

        animation = FuncAnimation(
            self.animation_fig,
            self.update_animation,
            fargs=(travel_axis, travel_axis_step),
            frames=bins,
            interval=100
        )

        # If you want to save the animation to a file, you can use the following line
        # animation.save('slice_animation.gif', writer='imagemagick')

        # Show the animation
        plt.show()
        
    #Visualization Helper Methods------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------
    def plot_slice(self, travel_axis, travel_axis_val):
            fig, ax = plt.subplots()
            # Initialize variables for total area and areas of each residue
            area_d = {}
            df = self.filter_df(travel_axis, travel_axis_val)
        
            for residue in set(self.df.ResType):
                area_d[residue]=0
            
            #define the orthoganal axes based off the travel axis
            if travel_axis == "X":
                orth_dim_1 = self.y_dim
                orth_axis_1 = "Y"
                orth_dim_2 = self.z_dim
                orth_axis_2 = "Z"
            if travel_axis == "Y":
                orth_dim_1 = self.x_dim
                orth_axis_1 = "X"
                orth_dim_2 = self.z_dim
                orth_axis_2 = "Z"
            if travel_axis == "Z":
                orth_dim_1 = self.x_dim
                orth_axis_1 = "X"
                orth_dim_2 = self.y_dim
                orth_axis_2 = "Y"
            # Iterate through each atom in the DataFrame
            for index, atom in df.iterrows():
                # Calculate the radius for the slice at Z
                r_proj_squared = atom['Radius']**2 - (atom[travel_axis] - travel_axis_val)**2
                if r_proj_squared < 0:
                    #print("invalid projection")
                    continue
                radius = np.sqrt(atom['Radius']**2 - (atom[travel_axis] - travel_axis_val)**2)
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
                if atom.AtomType == "A":
                    face_color = "k"
                if atom.AtomType == "B":
                    face_color = "w"
                # Plot the circle with different face color for each ResID
                circle = plt.Circle((atom[orth_axis_1], atom[orth_axis_2]), radius, edgecolor='k', facecolor=face_color, alpha=0.5)
                ax.add_patch(circle)
                
            # Set plot limits
            ax.set_xlim(0,orth_dim_1)
            ax.set_ylim(0,orth_dim_2)
            
            # Set labels
            ax.set_xlabel(orth_axis_1)
            ax.set_ylabel(orth_axis_2)
            
            # Show the plot
            plt.show()
    
    def update_animation(self, frame, travel_axis, travel_axis_step):
        plt.clf()  # Clear the previous frame
        travel_axis_val = frame * travel_axis_step
        circle = self.plot_slice(travel_axis, travel_axis_val)
        self.animation_fig.canvas.draw()
        
#--------------------------------------------------------------------------------------------------------------------
#Main (most likely not used)-----------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------     
def main():
    pass
if __name__ == "__main__":
    main()
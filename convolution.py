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
import os
import concurrent.futures
from functools import partial
#--------------------------------------------------------------------------------------------------------------------
#Global Constants---------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
avogadro_number = 6.022e23
angstrom3_to_liters = 1e-27
#--------------------------------------------------------------------------------------------------------------------
#Functions-----------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
def process_frame(frame, universe, travel_axis, molecule, bins, method="COM"):
    try:
        frameConv = FrameConvolution(universe, frame)
        df = frameConv.create_shifted_conc_df(travel_axis=travel_axis, molecule=molecule, bins=bins,method=method)
        return df
    except TypeError:
        print(f"Error encountered. Skipping frame: {frame}")
        return None

def normalize_df(df, norm_param="c_aa"):
    # Create a copy of the DataFrame to avoid modifying the original
    normalized_df = df.copy()

    # Find the maximum value in the specified column
    max_value = df[norm_param].max()

    # Divide the specified column by the maximum value
    normalized_df[norm_param] = df[norm_param] / max_value

    return normalized_df
#--------------------------------------------------------------------------------------------------------------------
#Class Definitions---------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
class UniverseConvolution:
    def __init__(self, conf_path, traj_path):
        print("NOTE: All units are in MDAnalysis units, see appropriate documentation. All distance units in angstroms")
        self.universe = mda.Universe(conf_path, traj_path)
        self.size = len(self.universe.trajectory)  # number of frames
    
    def create_ave_df(self, travel_axis="Z", molecule="UNK", bins=100, method="COM", frame_start=0, frame_end=False):
        if frame_end is False:
            frame_end = self.size

        df_l = []

        process_frame_partial = partial(process_frame, universe=self.universe, travel_axis=travel_axis,
                                        molecule=molecule, bins=bins, method=method)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_frame_partial, range(frame_start, frame_end)))

        for result in results:
            if result is not None:
                df_l.append(result)

        concatenated_df = pd.concat(df_l, axis=0)
        grouped_df = concatenated_df.groupby(concatenated_df.index)
        average_df = grouped_df.mean()
        return average_df

    def plot_ave_conc_scatter(self, travel_axis="Z", molecule="UNK", bins=100, method="COM", frame_start=0, frame_end=False):
        if frame_end is False:
            frame_end = self.size
 
        df = self.create_ave_df(travel_axis=travel_axis, molecule=molecule, bins=bins, method=method,
                                frame_start=frame_start, frame_end=frame_end)

        if method == "Area":
            df.plot(kind="scatter", x=travel_axis, y=f"{molecule}Conc")
        elif method == "COM":
            df.plot(kind="scatter", x=travel_axis, y=f"{molecule}Fraction")
        return None
    
class FrameConvolution:
    #***Special methods***-------------------------------------------------------------------------------------------
    def __init__(self,universe, frame_number):
        self.universe = universe
        self.frame_number = frame_number
        folder_name = "frames"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        csv_path = f"frames/frame{str(self.frame_number)}.csv"
        try:
            self.df = pd.read_csv(csv_path, index_col=False)
        except FileNotFoundError:
            self.df = self.gen_frame_df()
        self.molecules = set(self.df.ResType)
        self.x_dim = self.universe.trajectory[0].dimensions[0]
        self.y_dim = self.universe.trajectory[0].dimensions[1]
        self.z_dim = self.universe.trajectory[0].dimensions[2]
        #calculate cell parameters
        cell_conc_dict = {}
        self.cell_volume = self.x_dim * self.y_dim * self.z_dim
        for species in self.molecules:
            cell_conc_dict[species] = []
        for species in self.molecules:
            species_count = self.df[self.df["ResType"]==species].ResType.count()
            species_conc = (species_count / self.cell_volume) / avogadro_number / angstrom3_to_liters
            cell_conc_dict[species].append(species_conc)
        cell_conc_df = pd.DataFrame(cell_conc_dict)
        self.cell_conc_df = cell_conc_df
        
    #***Frame DF Generation***---------------------------------------------------------------------------------------
    def gen_frame_df(self):
        #---Create the initial dataframe:---
        atom_radius_mapping = {'C': .914, 'H': .5, 'N': .92, "O": .73, "A":1 , "B":1 }
        data_dict = {'AtomID': [], "AtomName":[], 'AtomType': [], "ResType": [], "ResID": [], 'X': [], 'Y': [], 'Z': [], 'Mass':[]
                    ,'X_COM': [],'Y_COM': [],'Z_COM': []}
        for j, atom in enumerate(self.universe.atoms):
            #print(atom)
            data_dict['AtomID'].append(atom.id)
            data_dict["AtomName"].append(atom.name)
            data_dict["AtomType"].append(atom.type)
            data_dict["ResType"].append(atom.resname)
            data_dict["ResID"].append(atom.resid)
            data_dict["Mass"].append(atom.mass)
            data_dict['X'].append(self.universe.trajectory[self.frame_number][j][0])
            data_dict['Y'].append(self.universe.trajectory[self.frame_number][j][1])
            data_dict['Z'].append(self.universe.trajectory[self.frame_number][j][2])
            #placeholders to keep the lists the same length
            data_dict["X_COM"].append(0)
            data_dict["Y_COM"].append(0)
            data_dict["Z_COM"].append(0)
        df = pd.DataFrame(data_dict)
        #df['Radius'] = df['AtomType'].map(atom_radius_mapping)

        #---calculate center of masses:---
        #Reset the list structures for COM
        data_dict["X_COM"] = []
        data_dict["Y_COM"] = []
        data_dict["Z_COM"] = []
        
        resid_set = set(df.ResID)
        for resid in resid_set:
            mol_df = df[df["ResID"] == resid]
            com_ten = self.calc_com(mol_df)
            for i in range(len(mol_df)):
                data_dict["X_COM"].append(float(com_ten[0]))
                data_dict["Y_COM"].append(float(com_ten[1]))
                data_dict["Z_COM"].append(float(com_ten[2]))
        #---Recreate the data_dict with the values:---
        df = pd.DataFrame(data_dict)   
        #map the radii
        df['Radius'] = df['AtomType'].map(atom_radius_mapping)
        #---Save to file:---
        csv_path = f"frames/frame{str(self.frame_number)}.csv"
        df.to_csv(csv_path,index=False)
        return df
    def calc_com(self, mol_df):
        mass_ten = torch.tensor(np.array(mol_df.Mass),dtype=torch.float64)
        #Center of mass is location for a monoatomic atom
        if len(mass_ten) == 1:
            x_COM = float(mol_df.X)
            y_COM = float(mol_df.Y)
            z_COM = float(mol_df.Z)
            com_tensor = torch.tensor((float(x_COM), float(y_COM),float(z_COM)))
            return com_tensor
        x_ten = torch.tensor(np.array(mol_df.X),dtype=torch.float64)
        y_ten = torch.tensor(np.array(mol_df.Y),dtype=torch.float64)
        z_ten = torch.tensor(np.array(mol_df.Z),dtype=torch.float64)
        mass_sum = mass_ten.sum()
        x_dot_prod = torch.dot(x_ten, mass_ten)
        y_dot_prod = torch.dot(y_ten, mass_ten)
        z_dot_prod = torch.dot(z_ten, mass_ten)
        x_COM = x_dot_prod / mass_sum
        y_COM = y_dot_prod / mass_sum
        z_COM = z_dot_prod / mass_sum
        return torch.tensor((float(x_COM), float(y_COM),float(z_COM)))
        
    #***General Helper Methods***------------------------------------------------------------------------------------
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

    def create_bins_df(self, bins=100):
        data_dict = {'X_start': [], "X_end":[], 'Y_start': [], "Y_end":[],'Z_start': [], "Z_end":[] }
        
        bin_x_step = self.x_dim/bins
        bin_y_step = self.y_dim/bins
        bin_z_step = self.z_dim/bins
        
        for i in range(bins):
            bin_start_x = i * bin_x_step
            data_dict["X_start"].append(bin_start_x)
            data_dict["X_end"].append(bin_start_x + bin_x_step)
            bin_start_y = i * bin_y_step
            data_dict["Y_start"].append(bin_start_y)
            data_dict["Y_end"].append(bin_start_y + bin_y_step)
            bin_start_z = i * bin_z_step
            data_dict["Z_start"].append(bin_start_z)
            data_dict["Z_end"].append(bin_start_z + bin_z_step)
        return pd.DataFrame(data_dict)

    def create_mol_conc_df(self,travel_axis = "Z", bins=100):
        bins_df = self.create_bins_df(bins=bins)
        df = self.df
        conc_dict = {}
        #calculate dimensions of the bin
        if travel_axis == "X":
            orth_dim_1 = self.y_dim
            orth_dim_2 = self.z_dim
        if travel_axis == "Y":
            orth_dim_1 = self.x_dim
            orth_dim_2 = self.z_dim
        if travel_axis == "Z":
            orth_dim_1 = self.x_dim
            orth_dim_2 = self.y_dim
        travel_axis_width = bins_df.iloc[0][f"{travel_axis}_end"] - bins_df.iloc[0][f"{travel_axis}_start"]
        bin_volume = orth_dim_1 * orth_dim_2 * travel_axis_width 
        
        #initialize list for each species
        conc_dict[travel_axis] = []
        for species in self.molecules:
            conc_dict[f'{species}Conc'] = []
            conc_dict[f'{species}Fraction'] = []
        #place into each bin
        for bin in bins_df.iterrows():
            bin = bin[1]
            #get travel axis midpoint
            travel_axis_mid = (bin[f'{travel_axis}_start'] + bin[f'{travel_axis}_end']) / 2
            conc_dict[travel_axis].append(travel_axis_mid)
            #filter into bins
            filtered_df = df[
                (df[f'{travel_axis}_COM'] >= bin[f'{travel_axis}_start']) & (df[f'{travel_axis}_COM'] <= bin[f'{travel_axis}_end']) 
            ]
            #count each species in the bin
            for species in self.molecules:
                no_molecules = filtered_df[filtered_df["ResType"]==species].ResType.count()
                mol_conc = (no_molecules/bin_volume) / avogadro_number / angstrom3_to_liters
                mol_fraction = no_molecules/filtered_df.ResType.count()
                conc_dict[f'{species}Conc'].append(mol_conc)
                conc_dict[f'{species}Fraction'].append(mol_fraction)
        conc_df = pd.DataFrame(conc_dict)
        return conc_df
            
    #***Analysis Methods***------------------------------------------------------------------------------------------
    def create_conc_df(self, travel_axis = "Z", bins = 100):
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
    
    def plot_caa_scatter(self,travel_axis = "Z", molecule="UNK", bins=100, method="COM", normalize=True):
        df = self.create_autocorr_df(travel_axis, molecule, bins, method, normalize=normalize)
        df.plot(kind="scatter", x="d",y="c_aa")
        return None
        
    #The autocorrelator function is the convlolutional method we're applying   
    def create_autocorr_df(self,travel_axis = "Z",molecule = "UNK", bins = 100, method="COM", normalize=True):
        caa_d = {"d":[],"c_aa": []}
        travel_axis_dim = self.get_dimension(travel_axis)
        if method == "Area":
            conc_df = self.create_conc_df(travel_axis, bins)
            conc_tensor = torch.tensor(np.array(conc_df[f'{molecule}Conc']))
        elif method == "COM":
            conc_df = self.create_mol_conc_df(travel_axis, bins)
            conc_tensor = torch.tensor(np.array(conc_df[f'{molecule}Fraction']))
        travel_midpoint = travel_axis_dim/2
        for d in range(int(-1*travel_midpoint),int(travel_midpoint+1)):
            #print(d)
            c_aa =self.auto_corr(conc_tensor, travel_axis_dim, d, molecule=molecule, method=method)
            caa_d["d"].append(d)
            caa_d["c_aa"].append(c_aa)
        df = pd.DataFrame(caa_d)
        if normalize:
            df = normalize_df(df)
        return df
        
    def plot_conc_scatter(self, travel_axis="Z", molecule="UNK",bins=100, method="COM"):
        if method == "Area":
            df = self.create_conc_df(travel_axis, bins)
            df.plot(kind="scatter",x=travel_axis, y=f'{molecule}Conc')
        elif method == "COM":
            df = self.create_mol_conc_df(travel_axis, bins)
            df.plot(kind="scatter",x=travel_axis, y=f'{molecule}Fraction')
    
    def create_pulse_df(self,travel_axis="Z", molecule="UNK", bins=100, method="COM", normalize=True):
        df = self.create_autocorr_df(travel_axis=travel_axis, molecule=molecule,bins=bins, method=method, normalize=normalize)
        d = {"d":[],"pulse":[]}
        for row in df.T:
            d["d"].append(df.iloc[row].d)
            if df.iloc[row].c_aa > 0:
                d["pulse"].append(1)
            else:
                d["pulse"].append(0)
        return pd.DataFrame(d)
    
    def plot_pulse_scatter(self,travel_axis = "Z", molecule="UNK", bins=100, method="COM"):
        df = self.create_pulse_df(travel_axis=travel_axis, molecule=molecule, bins=bins, method=method)
        df.plot(kind="scatter", x="d",y="pulse")
        return None
    
    def create_Xc_df(self,travel_axis = "Z",molecule = "UNK", bins = 100, method="COM"):
        pulse_df = self.create_pulse_df(travel_axis=travel_axis, molecule=molecule, bins=bins, method=method)
        if method == "Area":
            conc_df = self.create_conc_df(travel_axis=travel_axis, bins=len(pulse_df))
            conc_tensor = torch.tensor(np.array(conc_df[f"{molecule}Conc"]))
        elif method == "COM":
            conc_df = self.create_mol_conc_df(travel_axis=travel_axis, bins=len(pulse_df))
            conc_tensor = torch.tensor(np.array(conc_df[f"{molecule}Fraction"]))
        pulse_tensor = torch.tensor(np.array(pulse_df.pulse), dtype=torch.double)
        
        Xc_d = {"n":[],"Xc": []}
        travel_axis_dim = self.get_dimension(travel_axis)
        travel_midpoint = travel_axis_dim/2
        for n in range(len(pulse_df)):
            #print(d)
            X_c =self.X_c(n, conc_tensor, pulse_tensor)
            Xc_d["n"].append(n)
            Xc_d["Xc"].append(X_c)
        return pd.DataFrame(Xc_d)
    
    def plot_Xc_scatter(self, travel_axis="Z", molecule="UNK",bins=100, method="COM"):
        df = self.create_Xc_df(travel_axis=travel_axis, molecule=molecule, bins=bins, method=method)
        df.plot(kind="scatter", x="n",y="Xc")
        return None
        
    def create_shifted_conc_df(self, travel_axis="Z", molecule="UNK", bins=100, method="COM"):
        if method=="Area":
            conc_df = self.create_conc_df(travel_axis=travel_axis, bins=bins)
        elif method == "COM":
            conc_df = self.create_mol_conc_df(travel_axis=travel_axis, bins=bins)
        Xc_df = self.create_Xc_df(travel_axis=travel_axis, molecule=molecule, bins=bins, method=method)
        #print(f"DEBUG: Xc_df:\n {Xc_df}")
        n_max = int(Xc_df[Xc_df.Xc == Xc_df.Xc.max()].n)
        #print(f"DEBUG n_max: {n_max}")
        n_max = -n_max
        max_z = conc_df[travel_axis].max()
        conc_df["Z"] = conc_df["Z"].apply(lambda x: (x + n_max) % (max_z + 1))

        return conc_df    
    
    def plot_shifted_conc_scatter(self, travel_axis="Z", molecule="UNK",bins=100, method="COM"):
        df = self.create_shifted_conc_df(travel_axis=travel_axis, molecule=molecule, bins=bins, method=method)
        if method == "Area":
            df.plot(kind="scatter", x=travel_axis,y=f'{molecule}Conc')
        elif method == "COM":
            df.plot(kind="scatter", x=travel_axis,y=f'{molecule}Fraction')
        return None
        
    #***Analysis Helper Methods***-----------------------------------------------------------------------------------
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
        return vol_d
    
    def auto_corr(self, conc_tensor, travel_axis_dim, d, molecule="UNK", method="COM"):
        if method == 'Area':
            mean_conc = torch.mean(conc_tensor)
        elif method == "COM":
            #mean_conc = self.cell_conc_df[molecule][0]
            mean_conc = torch.mean(conc_tensor)
        #print(f"DEBUG: conc tensor: {conc_tensor}")
        #print(f"Mean {molecule} concentration: {mean_conc}")
        dist_per_bin = travel_axis_dim / len(conc_tensor)  
    
        # Calculate the ratio between the reference index 'd' and the original z-value per index

        #THIS IS WHAT POSSBILY CHANGED
        z_ratio = max(1, int(1 / dist_per_bin)) 
        #z_ratio = int(1/dist_per_bin) 
        
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
            
    #***Visualization Methods***-------------------------------------------------------------------------------------
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
        
    #***Visualization Helper Methods***------------------------------------------------------------------------------
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
        
print(f"Using cuda: {torch.cuda.is_available()}")
#--------------------------------------------------------------------------------------------------------------------
#Main (most likely not used)-----------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------     
def main():
    pass
if __name__ == "__main__":
    main()
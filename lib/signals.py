import numpy as np
from scipy import interpolate

class SignalModel:

    def __init__(self,theory_data_dir):
        '''
        Initialize signal model object.
        '''
        self.loaded = False
        self.theory_data_dir = theory_data_dir

    def load_force_funcs(self):
        pass

    def get_force_at_pos(self,position,params):
        print('No force functions loaded!')

    def get_params_and_inds(self):
        pass


class GravFuncs(SignalModel):
    '''
    Child class of SignalModel for the Yukawa-modified gravity model.
    '''

    def load_force_funcs(self,lambda_range=[1e-6,1e-4],num_lambdas=None):
        '''
        Loads data from the output of /data/grav_sim_data/process_data.py
        which processes the raw simulation output from the farmshare code.
        '''

        # load modified gravity curves from simulation output
        Gdata = np.load(self.theory_data_dir + 'Gravdata.npy')
        yukdata = np.load(self.theory_data_dir + 'yukdata.npy')
        lambdas = np.load(self.theory_data_dir + 'lambdas.npy')
        xpos = np.load(self.theory_data_dir + 'xpos.npy')
        ypos = np.load(self.theory_data_dir + 'ypos.npy')
        zpos = np.load(self.theory_data_dir + 'zpos.npy')
        rbead = np.load(self.theory_data_dir + 'rbead_rhobead.npy')

        # get only the subset of lambda values requested
        lambda_inds = np.arange(0,len(lambdas))
        lambda_inds = lambda_inds[(lambdas>lambda_range[0]) & (lambdas<lambda_range[1])]
        lambda_sub = lambdas[(lambdas>lambda_range[0]) & (lambdas<lambda_range[1])]
        if num_lambdas:
            if num_lambdas<len(lambda_sub):
                lambda_mask = np.round(np.linspace(0,len(lambda_sub)-1,num_lambdas)).astype(int)
                lambda_sub = lambda_sub[lambda_mask]
                lambda_inds = lambda_inds[lambda_mask]

        # find limits to avoid out of range erros in interpolation
        xlim = (np.min(xpos), np.max(xpos))
        ylim = (np.min(ypos), np.max(ypos))
        zlim = (np.min(zpos), np.max(zpos))

        # build interpolating functions for regular gravity
        grav_funcs = [0,0,0]
        # loop through components of the force
        for component_ind in [0,1,2]:
            grav_funcs[component_ind] = interpolate.RegularGridInterpolator((xpos, ypos, zpos), Gdata[:,:,:,component_ind])

        # build interpolating functions for Yukawa-modified gravity
        yuk_funcs = [[],[],[]]
        # loop through components of the force
        for component_ind in [0,1,2]:
            for _,lamb_ind in enumerate(lambda_inds):
                lamb_func = interpolate.RegularGridInterpolator((xpos, ypos, zpos), yukdata[lamb_ind,:,:,:,component_ind])
                yuk_funcs[component_ind].append(lamb_func)
        lims = [xlim, ylim, zlim]
    
        self.grav_funcs = grav_funcs
        self.yuk_funcs = yuk_funcs
        self.lambdas = lambda_sub
        self.xyz_lims = lims
        self.rad_bead = rbead[0]
        self.rho_bead = rbead[1]
        self.loaded = True


    def get_params_and_inds(self):
        '''
        Returns a tuple of arrays of the parameters and their corresponding indices,
        to be used for passing to get_force_at_pos
        '''
        return self.lambdas,np.array(range(len(self.lambdas)))
    

    def get_force_at_pos(self, positions, param_inds):
        '''
        Return the force at a given position. Argument 'positions' is an array of dimensions
        Nx3 where N is the number of positions where the force should be calculated. Argument
        'param_inds' is a list of length M where M is the number of free parameters in the
        model. For the Yukawa-modified gravity model, there is only one additional parameter,
        lambda.
        '''

        # index of the parameter to choose the correct interpolating function
        lamb_ind = param_inds[0]

        # get the x,y, and z components of the force and return them
        x_force = self.yuk_funcs[0][lamb_ind](positions)
        y_force = self.yuk_funcs[1][lamb_ind](positions)
        z_force = self.yuk_funcs[2][lamb_ind](positions)

        force_array = np.array((x_force,y_force,z_force)).T

        return force_array
    

class EDMFuncs(SignalModel):
    '''
    Child class of SignalModel for the electric dipole moment background model.
    '''

    def load_force_funcs(self,dipole_vec=[100,0,0]):
        '''
        Load the electric dipole moment background model, which has been parsed from
        the COMSOL simulation output and saved as numpy arrays.
        '''

        # load electric dipole moment background data from simulation output
        grad_e = np.load(self.theory_data_dir + 'grad_efield.npy')
        xpos = np.load(self.theory_data_dir + 'xpos.npy')
        ypos = np.load(self.theory_data_dir + 'ypos.npy')
        zpos = np.load(self.theory_data_dir + 'zpos.npy')

        # dipole moment vector in units of e-microns
        dipole_vec = np.array(dipole_vec)

        # dot product of the dipole moment vector with the electric field gradient
        q_e = 1.602176634e-19
        force = np.einsum('i,ijklm->jklm',dipole_vec,grad_e)*q_e*1e-6 # force in N

        # build interpolating functions for the force on the dipole
        edm_funcs = [0,0,0]
        # loop through components of the force
        for component_ind in [0,1,2]:
            edm_funcs[component_ind] = interpolate.RegularGridInterpolator((xpos, ypos, zpos), force[component_ind,:,:,:])

        # save the interpolating functions as class attributes
        self.edm_funcs = edm_funcs
        self.loaded = True


    def get_force_at_pos(self, positions):
        '''
        Return the force at a given position. Argument 'positions' is an array of dimensions
        Nx3 where N is the number of positions where the force should be calculated.
        '''

        # get the x,y, and z components of the force and return them
        x_force = self.edm_funcs[0](positions)
        y_force = self.edm_funcs[1](positions)
        z_force = self.edm_funcs[2](positions)

        force_array = np.array((x_force,y_force,z_force)).T

        return force_array
import torch
import numpy as np
import pickle
import torch.nn as nn
import smplx
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

SMPLH_PARENTS = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                              16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                              35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50])


kwargs_disable_member_var = {
    'create_body_pose': False,
    'create_betas': False,
    'create_global_orient': False,
    'create_transl': False,
    'create_left_hand_pose': False,
    'create_right_hand_pose': False,
    'create_expression': False,
    'create_jaw_pose': False,
    'create_leye_pose': False,
    'create_reye_pose': False,
}

class BodyModelSMPLX(nn.Module):
    """Support Batch inference"""

    def __init__(self, model_path, **kwargs):
        super().__init__()
        logging.debug(f"Initializing BodyModelSMPLX with model_path: {model_path} and kwargs: {kwargs}")

        # Check if model_path exists and log its contents
        if not os.path.exists(model_path):
            logging.error(f"Model path does not exist: {model_path}")
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        else:
            logging.debug(f"Model path exists: {model_path}")
            logging.debug(f"Contents of model_path: {os.listdir(model_path)}")

        # enable flexible batchsize, handle missing variable at forward()
        kwargs.update(kwargs_disable_member_var)
        logging.debug("Creating SMPLX model with the provided parameters.")
        self.bm = smplx.create(model_path=model_path, **kwargs)
        logging.debug("SMPLX model created successfully.")

        self.hand_pose_dim = self.bm.num_pca_comps if self.bm.use_pca else 3 * self.bm.NUM_HAND_JOINTS
        logging.debug(f"Hand pose dimension set to: {self.hand_pose_dim}")

    def forward(self,
                betas=None,
                global_orient=None,
                transl=None,
                body_pose=None,
                left_hand_pose=None,
                right_hand_pose=None,
                expression=None,
                jaw_pose=None,
                leye_pose=None,
                reye_pose=None,
                **kwargs):

        logging.debug("Forward pass of BodyModelSMPLX called.")
        device, dtype = self.bm.shapedirs.device, self.bm.shapedirs.dtype
        logging.debug(f"Device: {device}, Dtype: {dtype}")

        model_vars = [betas, global_orient, body_pose, transl,
                      expression, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))

   

        # Log default values being set
        if global_orient is None:
      
            global_orient = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if body_pose is None:
  
            body_pose = torch.zeros(3 * self.bm.NUM_BODY_JOINTS, device=device,
                                    dtype=dtype)[None].expand(batch_size, -1).contiguous()
        if left_hand_pose is None:
            
            left_hand_pose = torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[
                None].expand(batch_size, -1).contiguous()
        if right_hand_pose is None:
           
            right_hand_pose = torch.zeros(self.hand_pose_dim, device=device, dtype=dtype)[
                None].expand(batch_size, -1).contiguous()
        if jaw_pose is None:
           
            jaw_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if leye_pose is None:
          
            leye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if reye_pose is None:
     
            reye_pose = torch.zeros([batch_size, 3], dtype=dtype, device=device)
        if expression is None:
           
            expression = torch.zeros([batch_size, self.bm.num_expression_coeffs], dtype=dtype, device=device)
        if betas is None:
      
            betas = torch.zeros([batch_size, self.bm.num_betas], dtype=dtype, device=device)
        if transl is None:
     
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        bm_out = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            **kwargs
        )

        logging.debug("Forward pass completed successfully.")
        return bm_out
    

def make_smplx(type='humanise', **kwargs,):
    logging.debug(f"make_smplx called with type: {type} and kwargs: {kwargs}")
    if type == 'humanise':
        model = BodyModelSMPLX(
            model_path='data/smpl_models',
            model_type='smplx',
            gender='neutral',
            ext='npz',
            num_betas=10,
            use_pca=False,
        )
        logging.debug("Humanise SMPLX model created successfully.")
    else:
        logging.error(f"Model type {type} not implemented.")
        raise NotImplementedError

    return model

def load_smpl_faces(npz_path="data/smpl_models/smplx/SMPLX_FEMALE.pkl"):
    import numpy as np
    
    print(f"NumPy version: {np.__version__}")
    print(f"NumPy location: {np.__file__}")
    
    smpl_model = pickle.load(open(npz_path, 'rb'), encoding='latin1')
    faces = smpl_model['f'].astype(np.int64)
    
    # Multiple attempts to ensure proper numpy array
    faces = np.ascontiguousarray(faces)
    faces = np.array(faces, dtype=np.int64)
    
    print(f"Final faces type: {type(faces)}")
    print(f"Final faces module: {type(faces).__module__}")
    print(f"Is numpy ndarray: {isinstance(faces, np.ndarray)}")
    
    logging.debug(f"Loaded SMPL faces with type: {type(faces)}, dtype: {faces.dtype}, flags: {faces.flags}")
    return faces
"""Observation-only history and deterministic rasterization for the 2D game.

No environment object, critic state, entity truth IDs or snapshot is accepted.
"""
from dataclasses import asdict, dataclass
import numpy as np
import torch

from envs import flatten_obs
from envs.config import TYPE_PARAMS, TYPE_NAMES


@dataclass(frozen=True)
class BeliefSpec:
    agents: int
    targets: int
    threats: int
    base_dim: int
    world_size: float
    length_scale: float
    radius_scale: float
    target_speed: float
    dt: float
    max_steps: int
    grid: int = 32

    @property
    def dim(self):
        return self.base_dim+self.targets*6+self.grid*self.grid


def belief_spec(env_cfg, obs_dim, grid=32):
    if grid < 4 or grid > 80:
        raise ValueError('belief grid must be between 4 and 80')
    return BeliefSpec(env_cfg.n_uavs,env_cfg.n_targets,env_cfg.n_threats,obs_dim,
        env_cfg.world_size,env_cfg.length_scale,env_cfg.threat_radius_scale,
        env_cfg.target_speed,env_cfg.dt,env_cfg.max_steps,grid)


def world_positions(own, entities, scale):
    # ego forward = dx*sin(yaw)+dy*cos(yaw); right = dx*cos(yaw)-dy*sin(yaw)
    fwd,right=entities[...,0]*scale,entities[...,1]*scale
    dx=fwd*own[...,5,None]+right*own[...,6,None]
    dy=fwd*own[...,6,None]-right*own[...,5,None]
    return np.stack((own[...,3,None]+dx,own[...,4,None]+dy),axis=-1)


class ObservationHistory:
    def __init__(self,spec,envs):
        self.spec=spec
        self.targets=np.zeros((envs,spec.agents,spec.targets,6),np.float32)
        self.coverage=np.full((envs,spec.agents,spec.grid,spec.grid),-1,np.float32)
        centers=(np.arange(spec.grid)+.5)/spec.grid
        x,y=np.meshgrid(centers,centers,indexing='xy')
        self.cells=np.stack((x,y),-1)

    def reset(self,mask):
        self.targets[mask]=0
        self.coverage[mask]=-1

    def _distance_tables(self,own,positions):
        # An unseen slot cannot change during association: a written slot is
        # immediately added to seen and excluded from all subsequent matches.
        # Negative evidence also reads only unseen slots. Thus both tables use
        # the same pre-association state as the original scalar distance calls.
        matches=np.linalg.norm(positions[..., :,None,:]-self.targets[...,None,:,:2],axis=-1)
        observer=np.linalg.norm(self.targets[...,:2]-own[...,None,3:5],axis=-1)
        return matches,observer

    def encode(self,obs,advance=False,env_mask=None,reset_mask=None):
        s=self.spec; E,N=obs['active'].shape
        if reset_mask is not None:self.reset(reset_mask)
        active=obs['active'].copy()
        if env_mask is not None:active &= np.asarray(env_mask)[:,None]
        age_step=s.dt/(s.max_steps*s.dt)
        if advance:
            known=self.targets[...,5]>0
            self.targets[...,4]+=age_step*(active[...,None]&known)
            cov_known=self.coverage>=0
            self.coverage=np.where(cov_known&active[...,None,None],
                np.minimum(1,self.coverage+age_step),self.coverage)
        own=obs['self']
        positions=world_positions(own,obs['targets'],s.length_scale/s.world_size)
        radii=np.array([TYPE_PARAMS[k]['recon_range'] for k in TYPE_NAMES])
        ranges=(own[...,:3]*radii).sum(-1)*own[...,8]/s.world_size
        # Two coordinates need no generic last-axis reduction or [...,2] temporary.
        dx=self.cells[None,None,...,0]-own[...,None,None,3]
        dy=self.cells[None,None,...,1]-own[...,None,None,4]
        distance=np.sqrt(dx*dx+dy*dy)
        covered=(distance<ranges[...,None,None])&active[...,None,None]
        self.coverage[covered]=0
        match_distances,observer_distances=self._distance_tables(own,positions)
        for e,n in np.argwhere(active):
            slots=self.targets[e,n]; seen=set()
            for j in np.flatnonzero(obs['targets_mask'][e,n]):
                pos=positions[e,n,j]
                available=[k for k in range(s.targets) if k not in seen and slots[k,5]>0]
                matches=[]
                for k in available:
                    age_seconds=slots[k,4]*s.max_steps*s.dt
                    gate=(2+s.target_speed*age_seconds)/s.world_size
                    d=float(match_distances[e,n,j,k])
                    if d<=gate:matches.append((d,k))
                if matches:k=min(matches)[1]
                else:
                    free=[k for k in range(s.targets) if k not in seen and slots[k,5]==0]
                    remaining=[k for k in range(s.targets) if k not in seen]
                    if not remaining:break
                    k=free[0] if free else max(remaining,key=lambda k:(slots[k,4],-k))
                rel_sin,rel_cos=obs['targets'][e,n,j,3:5]
                world_sin=rel_sin*own[e,n,6]+rel_cos*own[e,n,5]
                world_cos=rel_cos*own[e,n,6]-rel_sin*own[e,n,5]
                slots[k]=(*pos,world_sin,world_cos,0,1)
                seen.add(k)
            # Remove only when the complete bounded possible region is within
            # the observer's own current sensing disk and no detection matched.
            for k in range(s.targets):
                if slots[k,5] and k not in seen:
                    radius=s.target_speed*slots[k,4]*s.max_steps*s.dt/s.world_size
                    if observer_distances[e,n,k]+radius < ranges[e,n]:
                        slots[k]=0
        return np.concatenate((flatten_obs(obs),self.targets.reshape(E,N,-1),
                               self.coverage.reshape(E,N,-1)),-1).astype(np.float32)

    def state_dict(self):
        return dict(spec=asdict(self.spec),targets=torch.from_numpy(self.targets.copy()),
                    coverage=torch.from_numpy(self.coverage.copy()))

    def load_state_dict(self,payload):
        if payload['spec']!=asdict(self.spec):raise ValueError('belief schema differs')
        for name in ('targets','coverage'):
            value=payload[name].cpu().numpy()
            if value.shape!=getattr(self,name).shape:raise ValueError('belief batch differs')
            setattr(self,name,value.copy())


def rasterize(inputs,spec):
    """[batch,agent,compact] -> [batch,agent,9,y,x], no privileged inputs."""
    s=spec; shape=inputs.shape[:2]; dtype=inputs.dtype; device=inputs.device
    centers=(torch.arange(s.grid,device=device,dtype=dtype)+.5)/s.grid
    yy,xx=torch.meshgrid(centers,centers,indexing='ij')
    slots=inputs[...,s.base_dim:s.base_dim+s.targets*6].reshape(*shape,s.targets,6)
    dx=xx-slots[...,0,None,None]; dy=yy-slots[...,1,None,None]
    age=slots[...,4]; known=slots[...,5]
    sigma=(.5/s.grid+s.target_speed*age*s.max_steps*s.dt/s.world_size)
    mass=torch.exp(-(dx.square()+dy.square())/(2*sigma[...,None,None].square()))
    mass=mass/mass.sum((-1,-2),keepdim=True).clamp_min(1e-12)
    confidence=torch.exp(-3*age)*known
    mass=mass*confidence[...,None,None]
    target=mass.sum(-3)
    fresh=(mass*torch.exp(-3*age)[...,None,None]).sum(-3)
    uncertainty=(mass*(sigma*s.grid)[...,None,None]).sum(-3)
    hs=(mass*slots[...,2,None,None]).sum(-3)
    hc=(mass*slots[...,3,None,None]).sum(-3)
    own=inputs[...,:9]
    def positions(features):
        f,r=features[...,0]*s.length_scale/s.world_size,features[...,1]*s.length_scale/s.world_size
        return (own[...,3,None]+f*own[...,5,None]+r*own[...,6,None],
                own[...,4,None]+f*own[...,6,None]-r*own[...,5,None])
    cursor=9+(s.agents-1)*8
    mates=inputs[...,9:cursor].reshape(*shape,s.agents-1,8)
    mates_mask=inputs[...,cursor:cursor+s.agents-1]
    cursor+=s.agents-1+s.targets*5+s.targets
    threats=inputs[...,cursor:cursor+s.threats*4].reshape(*shape,s.threats,4)
    masks=inputs[...,cursor+s.threats*4:cursor+s.threats*5]
    tx,ty=positions(threats)
    radius=threats[...,3]*s.radius_scale/s.world_size
    threat=((xx-tx[...,None,None]).square()+(yy-ty[...,None,None]).square()
            <radius[...,None,None].square()).to(dtype)*masks[...,None,None]
    threat=threat.amax(-3)
    # Mate position fields follow the three one-hot type entries.
    mx,my=positions(mates[...,3:])
    friend=torch.exp(-((xx-mx[...,None,None]).square()+(yy-my[...,None,None]).square())
                     /(2*(.5/s.grid)**2))*mates_mask[...,None,None]
    friend=friend.amax(-3)
    self_marker=torch.exp(-((xx-own[...,3,None,None]).square()+(yy-own[...,4,None,None]).square())
                         /(2*(.5/s.grid)**2))
    coverage=inputs[...,s.base_dim+s.targets*6:].reshape(*shape,s.grid,s.grid)
    coverage_known=(coverage>=0).to(dtype)
    coverage_fresh=torch.exp(-3*coverage.clamp_min(0))*coverage_known
    return torch.stack((target,fresh,uncertainty,hs,hc,threat,
                        torch.maximum(friend,self_marker),coverage_known,coverage_fresh),-3)

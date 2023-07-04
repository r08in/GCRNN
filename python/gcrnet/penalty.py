import torch 
import math
def prox_plus(z):
        """Projection onto non-negative numbers
        """
        below = z < 0
        z[below] = 0
        return z

def SoftThresh(z, lam):
    '''
    z:vector of parameters
    lam: tuning parameter of penalty term
    '''
    return torch.sign(z) * prox_plus(torch.abs(z)-lam)

def GroupSoftThresh(z, lam):
    '''
    z:vector of parameters
    lam: tuning parameter of penalty term
    '''
    return z * prox_plus(1-lam/torch.norm(z, p=2))

def GMCP_op(z, lam, gamma=3):
    '''
    z:vector of parameters
    lam: tuning parameter of penalty term
    gamma: extra tuning parameter. Default is 3.
    '''
    z_norm = torch.norm(z, p=2)
    if (z_norm <= gamma*lam):
        return gamma/(gamma-1) * GroupSoftThresh(z, lam)
    else:
        return z

def GSCAD_op(z, lam, gamma=3.7):
    '''
    z:vector of parameters
    lam: tuning parameter of penalty term
    gamma: extra tuning parameter. Default is 3.7.
    '''
    z_norm = torch.norm(z, p=2)
    if (z_norm <= 2*lam):
        return  GroupSoftThresh(z, lam)
    elif 2*lam < z_norm and z_norm <= gamma * lam:
        return (gamma-1)/(gamma-2) * GroupSoftThresh(z, (gamma*lam)/(gamma-1))
    else:
        return z

def MCP(u, lam, gamma=3):
    '''
    calculate the value of MCP penalty
    u: a non-negative scalar
    lam: tuning parameter of penalty term
    gamma: extra tuning parameter. Default is 3.
    '''
    if u <= gamma * lam:
        return lam * u - u**2/(2*gamma)
    else:
        return gamma * lam**2/2

def MCP_D1(u, lam, gamma=3):
    '''
    calculate the first derivative of MCP penalty
    u: a non-negative scalar
    lam: tuning parameter of penalty term
    gamma: extra tuning parameter. Default is 3.
    '''
    if (u <= gamma * lam):
        return (lam - u/gamma)
    else:
        return 0

def SCAD(u, lam, gamma=3.7):
    '''
    calculate the value of SCAD penalty
    u: a non-negative scalar
    lam: tuning parameter of penalty term
    gamma: extra tuning parameter. Default is 3.7.
    '''
    if(u <= lam):
        return lam * u
    elif (lam < u and u <= gamma * lam):
        return (gamma * lam  * u - 0.5*(u**2+lam**2)) / (gamma-1)
    else:
        return lam**2 * (gamma**2-1)/(2*(gamma-1))

def SCAD_D1(u, lam, gamma=3.7):
    '''
    calculate the first derivative of SCAD penalty
    u: a non-negative scalar
    lam: tuning parameter of penalty term
    gamma: extra tuning parameter. Default is 3.7.
    '''
    if(u <= lam):
        return lam
    elif (lam < u and u <= gamma * lam):
        return (gamma * lam - u) / (gamma-1)
    else:
        return 0


def ConcaveSoftThresh(z, lam, outer_penalty, inner_penalty):
    if inner_penalty == "l2":
        lam = lam * math.sqrt(z.size(0))
        if outer_penalty == "MCP":
            return GMCP_op(z, lam)
        elif outer_penalty == "SCAD":
            return GSCAD_op(z, lam)
        elif outer_penalty == "LASSO":
            return GroupSoftThresh(z, lam)
        else:
            raise ValueError('Unknown outer penalty function: {}.'.format(outer_penalty))
    elif inner_penalty == "l1":
        raise NotImplementedError("This method has not been implemented yet")



def get_penalty_val(z, lam, outer_penalty, inner_penalty):
    if inner_penalty == "l2":
        z_norm = torch.norm(z, p=2)
        lam = lam * math.sqrt(z.size(0))
    elif inner_penalty == "l1":
        z_norm = torch.norm(z, p=1)
        lam = lam * z.size(0)
    if outer_penalty == "MCP":
        return MCP(z_norm, lam)
    elif outer_penalty == "SCAD":
        return SCAD(z_norm, lam)
    elif outer_penalty == "LASSO":
        return z_norm * lam
    else:
        raise ValueError('Unknown outer penalty function: {}.'.format(outer_penalty))
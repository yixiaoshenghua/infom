from agents.crl_infonce_pytorch import CRLInfoNCEAgentPytorch
from agents.dino_rebrac_pytorch import DINOReBRACAgentPytorch
from agents.fb_repr import ForwardBackwardRepresentationAgent # Assuming this is PyTorch
from agents.hilp import HILPAgent # Assuming JAX, or to be converted
from agents.infom import InFOMAgent # Assuming JAX, or to be converted
from agents.iql import IQLAgent # Assuming JAX, or to be converted
from agents.mbpo_rebrac import MBPOReBRACAgent # Assuming JAX, or to be converted
from agents.rebrac import ReBRACAgent # Assuming JAX, or to be converted
from agents.td_infonce import TDInfoNCEAgent # Assuming JAX, or to be converted

agents = dict(
    crl_infonce=CRLInfoNCEAgentPytorch,
    dino_rebrac=DINOReBRACAgentPytorch,
    fb_repr=ForwardBackwardRepresentationAgent, # Keep if PyTorch
    hilp=HILPAgent,
    infom=InFOMAgent,
    iql=IQLAgent,
    mbpo_rebrac=MBPOReBRACAgent,
    rebrac=ReBRACAgent,
    td_infonce=TDInfoNCEAgent,
)

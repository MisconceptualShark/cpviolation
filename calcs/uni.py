from __future__ import division
import numpy as np

Vud,Vud_err = [0.974390,[0.000014,-0.000058]]
Vus,Vus_err = [0.224834,[0.000252,-0.000059]]
Vub,Vub_err = [0.003683,[0.000075,-0.000061]]
Vcd,Vcd_err = [0.224701,[0.000254,-0.000058]]
Vcs,Vcs_err = [0.973539,[0.000038,-0.000060]]
Vcb,Vcb_err = [0.04162,[0.00026,-0.00080]]
Vtd,Vtd_err = [0.008545,[0.000075,-0.000157]]
Vts,Vts_err = [0.04090,[0.00026,-0.00076]]
Vtb,Vtb_err = [0.999127,[0.000032,-0.000012]]

print "|Vub'| = ",np.sqrt(1-(Vud**2 + Vus**2 + Vub**2))
print "|Vcb'| = ",np.sqrt(abs(1-(Vcd**2 + Vcs**2 + Vcb**2)))

print "|Vtb'| = ",np.sqrt(abs(1-(Vub**2 + Vcb**2 + Vtb**2)))

print "Vtd^2 + Vt'd^2 = ",(1-(Vud**2 + Vcd**2))
print "Vts^2 + Vt's^2 = ",(1-(Vus**2 + Vcs**2))


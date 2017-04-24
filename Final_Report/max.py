import numpy as np
import matplotlib.pyplot as plt

vals10 = [0.71377459749552774, 0.71198568872987478, 0.7155635062611807, 0.71019677996422181, 0.70483005366726292, 
		0.71019677996422181, 0.69767441860465118, 0.71019677996422181, 0.70304114490161007, 0.71735241502683367,
		0.7155635062611807, 0.70840787119856885, 0.70483005366726292, 0.70483005366726292, 0.72271914132379245,
		0.71914132379248663, 0.71735241502683367, 0.70840787119856885, 0.7155635062611807, 0.71914132379248663,
		0.7155635062611807, 0.69588550983899822, 0.69588550983899822, 0.70840787119856885, 0.69588550983899822, 
		0.7298747763864043, 0.7155635062611807, 0.70125223613595711, 0.7155635062611807, 0.71914132379248663,
		0.70661896243291589, 0.71019677996422181, 0.71019677996422181, 0.72093023255813948, 0.71735241502683367,
		0.70661896243291589, 0.70483005366726292, 0.70840787119856885, 0.71019677996422181, 0.71735241502683367,
		0.70661896243291589, 0.7155635062611807, 0.7155635062611807, 0.71019677996422181, 0.70483005366726292,
		0.72450805008944541, 0.71198568872987478, 0.70661896243291589, 0.7155635062611807, 0.69588550983899822]

vals100 = [0.72093023255813948, 0.71914132379248663, 0.70661896243291589, 0.70661896243291589, 0.7155635062611807,
			0.72093023255813948, 0.71019677996422181, 0.7155635062611807, 0.71735241502683367, 0.71019677996422181,
			0.71198568872987478, 0.72093023255813948, 0.72271914132379245, 0.72093023255813948, 0.71019677996422181, 
			0.71377459749552774, 0.7155635062611807, 0.71019677996422181, 0.73166368515205726, 0.70661896243291589,
			0.72093023255813948, 0.70840787119856885, 0.70483005366726292, 0.71019677996422181, 0.72629695885509837,
			0.7155635062611807, 0.71377459749552774, 0.73524150268336319, 0.71019677996422181, 0.71735241502683367,
			0.71019677996422181, 0.71019677996422181, 0.70483005366726292, 0.71019677996422181, 0.7155635062611807, 
			0.70483005366726292, 0.70661896243291589, 0.72093023255813948, 0.71735241502683367, 0.7155635062611807,
			0.71019677996422181, 0.7155635062611807, 0.70840787119856885, 0.70840787119856885, 0.73166368515205726, 
			0.72271914132379245, 0.72093023255813948, 0.72093023255813948, 0.7155635062611807, 0.70304114490161007]

vals500 = [0.76386404293381038, 0.73166368515205726, 0.73524150268336319, 0.72629695885509837, 0.71735241502683367, 
			0.73345259391771023, 0.72450805008944541, 0.73166368515205726, 0.71377459749552774, 0.7155635062611807,
			0.74239713774597493, 0.71377459749552774, 0.73345259391771023, 0.74060822898032197, 0.7298747763864043, 
			0.7155635062611807, 0.73703041144901615, 0.7155635062611807, 0.73345259391771023, 0.71735241502683367,
			0.72629695885509837, 0.71735241502683367, 0.71019677996422181, 0.72450805008944541, 0.72271914132379245, 
			0.71198568872987478, 0.7155635062611807, 0.7298747763864043, 0.74239713774597493, 0.74060822898032197,
			0.73166368515205726, 0.73881932021466901, 0.71377459749552774, 0.71914132379248663, 0.71914132379248663,
			0.71377459749552774, 0.72093023255813948, 0.7441860465116279, 0.7155635062611807, 0.72629695885509837,
			0.70840787119856885, 0.7441860465116279, 0.73345259391771023, 0.72093023255813948, 0.74597495527728086,
			0.73345259391771023, 0.72450805008944541, 0.71914132379248663, 0.73345259391771023, 0.72271914132379245]

vals1000 = [0.70661896243291589, 0.74597495527728086, 0.71019677996422181, 0.70483005366726292, 0.7155635062611807,
			0.76207513416815742, 0.71019677996422181, 0.73524150268336319, 0.72629695885509837, 0.7441860465116279,
			0.73524150268336319, 0.71588550983899822, 0.71377459749552774, 0.71735241502683367, 0.70483005366726292,
			0.72767441860465118, 0.76028622540250446, 0.72093023255813948, 0.71019677996422181, 0.73703041144901615,
			0.71377459749552774, 0.71198568872987478, 0.74776386404293382, 0.72450805008944541, 0.74776386404293382, 
			0.73524150268336319, 0.70840787119856885, 0.75776886404293382, 0.72629695885509837, 0.73524150268336319,
			0.72450805008944541, 0.71377459749552774, 0.73345259391771023, 0.75491949910554557, 0.71198568872987478,
			0.73703041144901615, 0.71914132379248663, 0.71735241502683367, 0.74239713774597493, 0.74239713774597493,
			0.7155635062611807, 0.71735241502683367, 0.72450805008944541, 0.71735241502683367, 0.72450805008944541, 
			0.76028622540250446, 0.72271914132379245, 0.73703041144901615, 0.71019677996422181, 0.73881932021466901]




# vals2 = [0.735241502683,0.728085867621, 0.738819320215, 0.731663685152, 0.738819320215,
#  0.745974955277, 0.710196779964, 0.744186046512, 0.729874776386,  0.715563506261]

plt.plot(list(range(len(vals10))), vals10, 'g', label ="Max learning iterations: 10")
# plt.plot(list(range(len(vals100))), vals100, 'r', label ="Max learning iterations: 100")
plt.plot(list(range(len(vals500))), vals500, 'b', label ="Max learning iterations: 500")
# plt.plot(list(range(len(vals1000))), vals1000, 'y', label ="Max learning iterations: 1000")
# plt.plot(list(range(len(vals))), vals500, 'g', label ="Max learning iterations: 10")
# plt.plot(list(range(len(vals))), vals2, 'r', label ="Select Best Two")
plt.legend(loc = "best")

plt.show()
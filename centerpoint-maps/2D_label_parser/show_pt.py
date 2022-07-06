import torch
ref = "'../data/output_3d/e93e98b63d3b40209056d129dc53ceee.pt"
file = "../output_3d_1/0e8782aa721545caabc7073d32fb1fb1.pt"
file_2 = "../output_3d/1e43672fe86540b78402473300ca4c8f.pt"
file_in = "/home/foj-sy/late_fusion/data/clocs_data/input_data/000010.pt"
file_in_2 = "data/nuScenes/test/0d12459cd86a4e6a87b48b779d9e3469.pt"

a = torch.load(file)
b = torch.load(file_2)
c = torch.load(file_in)
d = torch.load(file_in_2)
print(a)
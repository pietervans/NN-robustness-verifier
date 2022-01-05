from verifier import *
import os


def main():

    # List the networks
    networks = ['net0_fc1', 'net1_fc1', 'net0_fc2', 'net1_fc2', 'net0_fc3', 'net1_fc3', 'net0_fc4', 'net1_fc4', 'net0_fc5', 'net1_fc5']

    # List the specs
    test_files = [['example_img0_0.01800.txt', 'example_img1_0.05000.txt'], ['example_img0_0.07500.txt', 'example_img1_0.07200.txt'], 
    ['example_img0_0.09500.txt', 'example_img1_0.08300.txt'], ['example_img0_0.05200.txt', 'example_img1_0.07200.txt'], 
    ['example_img0_0.07500.txt', 'example_img1_0.08100.txt'], ['example_img0_0.06100.txt', 'example_img1_0.00230.txt'],
    ['example_img0_0.03300.txt', 'example_img1_0.01800.txt'], ['example_img0_0.05200.txt', 'example_img1_0.01300.txt'],
    ['example_img0_0.02100.txt', 'example_img1_0.01900.txt'], ['example_img0_0.08400.txt', 'example_img1_0.07800.txt']] 

    # Collect verifications 
    verifications = []

    # Test all networks on their corresponding test cases
    for net_file, net_specs in zip(networks, test_files):
        print("Verifying network", net_file)
        for net_spec in net_specs:  # for each of the two test files per network

            # Fix the net_spec argument
            spec = "../test_cases/" + net_file + "/" + net_spec

            with open(spec, 'r') as f:
                lines = [line[:-1] for line in f.readlines()]
                true_label = int(lines[0])
                pixel_values = [float(line) for line in lines[1:]]
                eps = float(spec[:-4].split('/')[-1].split('_')[-1])
                f.close()

            if net_file.endswith('fc1'):
                net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
            elif net_file.endswith('fc2'):
                net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
            elif net_file.endswith('fc3'):
                net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
            elif net_file.endswith('fc4'):
                net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
            elif net_file.endswith('fc5'):
                net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
            else:
                assert False

            net.load_state_dict(torch.load('../mnist_nets/%s.pt' % net_file, map_location=torch.device(DEVICE)))

            inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
            outs = net(inputs)
            pred_label = outs.max(dim=1)[1].item()
            assert pred_label == true_label

            # Verify the network
            verified = False
            if analyze(net, inputs, eps, true_label):
                verified = True
                output = "verified"
                print(output)
            else:
                output = "not verified"
                print(output)
            verifications.append(output)

    # Summarize result in txt file
    to_text = False
    if to_text:
        output_file = "test_summarize.txt"
        with open(output_file, 'w') as g:
            for i in range(len(networks)):
                line1 = networks[i] + "," + test_files[i][0] + "," + verifications[i] + "\n"  # first of the network tests
                line2 = networks[i] + "," + test_files[i][1] + "," + verifications[i] + "\n"  # second
                g.write(line1)
                g.write(line2)
            
            g.close()


if __name__ == '__main__':
    main()
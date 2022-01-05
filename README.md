# Course Project for the Reliable AI course at the ETH

This software can provably certify the robustness of neural networks using the DeepPoly framework ([paper](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf)).


## Running the verifier

The verifier can be run from the `code` directory using the command:

```bash
$ python verifier.py --net {net} --spec ../test_cases/{net}/img{test_idx}_{eps}.txt
```

In this command, `{net}` is equal to one of the following values (each representing one of the networks we want to verify): `net0_fc1, net1_fc1, net0_fc2, net1_fc2, net0_fc3, net1_fc3, net0_fc4, net1_fc4, net0_fc5, net1_fc5`.
`test_idx` is an integer representing index of the test case, while `eps` is the maximal perturbation that the verifier should certify in this test case.

E.g. you can run:

```bash
$ python verifier.py --net net0_fc1 --spec ../test_cases/net0_fc1/example_img0_0.01800.txt
```

This command corresponds to attempting to verify that the given image is classified correctly with any perturbation bounded by $\varepsilon=0.01800$ (L-infinity norm). If the verifier finishes successfully (i.e. outputs `verified`), it is guaranteed that the neural network is robust on the given image (this is called soundness).

To evaluate the verifier on all the images in the `test_cases` folder, you can run:

```bash
chmod +x evaluate
./evaluate
```

Analagously, the `evaluate_prelim` bash script runs the verifier on all the images in the `prelim_test_cases` folder. The testcases in this folder are harder to verify.

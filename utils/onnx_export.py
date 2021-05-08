import numpy as np
import onnx
import onnxruntime
import torch

def test_export_psp_net():
    # net = MyPSPNet("mobilenet_v2", encoder_weights="imagenet", activation="sigmoid", psp_use_batchnorm=False)
    # net.eval()

    net = torch.load("../best_models/unet.pt", map_location="cpu")
    net.eval()

    with torch.no_grad():
        # Input to the model
        x = torch.randn(1, 1, 512, 512)
        torch_out = net(x)

        # Export the model
        torch.onnx.export(net,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          "../best_models/unet.onnx",  # where to save the model (can be a file or file-like object)
                          # keep_initializers_as_inputs=True,
                          opset_version=10,
                          export_params=True)  # store the trained parameter weights inside the model file
    # opset_version=12,  # the ONNX version to export the model to
    # do_constant_folding=True,  # whether to execute constant folding for optimization
    # input_names=['input'],  # the model's input names
    # output_names=['output'],  # the model's output names
    # dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
    #              'output': {0: 'batch_size'}})
    # operator_export_type generally not working, only caffee

    onnx_model = onnx.load("../best_models/unet.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("../best_models/unet.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    test_export_psp_net()

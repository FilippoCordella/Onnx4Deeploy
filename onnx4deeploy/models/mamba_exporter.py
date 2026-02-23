# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: MIT

"""Mamba Model Exporter - ONNX with Custom Operators."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import onnx 
from onnx import helper, TensorProto, shape_inference

from ..core.base_exporter import BaseONNXExporter
from .pytorch_models.mamba import Mamba
from ..core.optimization_passes import RemoveIdentityPass, ShapeInferencePass

class MambaExporter(BaseONNXExporter):
    """
    ONNX exporter for Mamba model (Selective State Space Model).

    Exports clean ONNX graphs using custom SelectiveSSM operator:
    - High-level operators only (LayerNorm, Linear, Conv1d, Silu, SelectiveSSM)
    - ~10-15 nodes per layer (vs 50-80 with standard export)
    - No fragmented graphs (no excessive Const/Shape/Gather nodes)
    """

    def __init__(self, save_path: str = None, config_file: str = "config.yaml"):
        """
        Initialize Mamba exporter.

        Args:
            save_path: Optional custom path to save ONNX files
            config_file: Path to configuration YAML file
        """
        super().__init__(save_path, config_file)

    def load_config(self) -> Dict[str, Any]:
        """
        Load Mamba configuration.

        Returns:
            Dictionary containing Mamba configuration parameters
        """
        config = {
            "batch_size": 1,
            "d_model": 256,  # Model dimension
            "n_layers": 4,  # Number of Mamba layers
            "d_state": 16,  # SSM state dimension
            "d_conv": 4,  # Convolution kernel size
            "expand_factor": 2,  # SSM expansion factor
            "max_seq_len": 512,  # Maximum sequence length
            "vocab_size": None,  # Vocabulary size (None for continuous input)
            "num_classes": 10,  # Number of output classes (for classification)
            "dropout": 0.0,  # No dropout for inference
            "use_embedding": False,  # Use embedding layer for token inputs
            "opset_version": 17,  # ONNX opset version
            # Training configuration
            "training_strategy": "full",  # Options: "full", "last_layer", "custom"
            "custom_trainable_params": [],
        }
        return config

    def create_model(self) -> torch.nn.Module:
        """
        Create Mamba PyTorch model with ONNX export.

        Returns:
            Mamba model ready for export with custom operators
        """
        model = Mamba(
            d_model=self.config["d_model"],
            n_layers=self.config["n_layers"],
            d_state=self.config["d_state"],
            d_conv=self.config["d_conv"],
            expand_factor=self.config["expand_factor"],
            vocab_size=self.config["vocab_size"],
            num_classes=self.config["num_classes"],
            max_seq_len=self.config["max_seq_len"],
            dropout=self.config["dropout"],
            use_embedding=self.config["use_embedding"],
        )

        # Print model info
        num_params = model.get_num_params()
        print("\n📊 Mamba Model Configuration:")
        print("   Model: Mamba (Custom operators)")
        print(f"   Model dimension: {self.config['d_model']}")
        print(f"   Number of layers: {self.config['n_layers']}")
        print(f"   SSM state dimension: {self.config['d_state']}")
        print(f"   Total parameters: {num_params:,}")

        print("\n✨ ONNX Export Strategy:")
        print("   Using custom SelectiveSSM operator")
        print("   Generating ONNX graph with high-level operators")

        print("\n🎯 Expected ONNX operators (per layer):")
        print("   • LayerNorm - Normalization")
        print("   • Linear (projections) - Input/output projections")
        print("   • Conv1d (temporal) - Temporal convolution")
        print("   • Silu - Activation function")
        print("   • ai.mamba::SelectiveSSM - Custom SSM operator")
        print("   • Add (residual) - Residual connection")
        print("   Total: ~10-15 nodes/layer (vs 50-80 with fragmented export)")

        return model

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the input tensor shape for Mamba.

        Returns:
            Tuple representing input shape:
            - If use_embedding: (batch_size, seq_len) - token indices
            - Otherwise: (batch_size, seq_len, d_model) - continuous input
        """
        batch_size = self.config["batch_size"]
        seq_len = self.config["max_seq_len"]

        if self.config["use_embedding"]:
            return (batch_size, seq_len)
        else:
            d_model = self.config["d_model"]
            return (batch_size, seq_len, d_model)

    def get_trainable_params(self, all_param_names: List[str]) -> List[str]:
        """
        Get list of trainable parameter names for Mamba.

        Supports multiple training strategies:
        - "full": Train all parameters (default)
        - "last_layer": Only train the final classification/output layer
        - "custom": Use custom_trainable_params from config

        Args:
            all_param_names: List of all parameter names in the model

        Returns:
            List of parameter names that should be trainable
        """
        strategy = self.config.get("training_strategy", "full")

        if strategy == "full":
            trainable_params = all_param_names
        elif strategy == "last_layer":
            trainable_params = [
                name
                for name in all_param_names
                if "classifier" in name or "lm_head" in name or "norm_f" in name
            ]
        elif strategy == "custom":
            trainable_params = self.config.get("custom_trainable_params", [])
        else:
            print(f"⚠️  Unknown training strategy '{strategy}', using 'full'")
            trainable_params = all_param_names

        requires_grad = [name for name in all_param_names if name in trainable_params]

        print(f"\n🎯 Training Strategy: '{strategy}'")
        print(f"   Total params: {len(all_param_names)}")
        print(f"   Trainable: {len(requires_grad)}")
        print(f"   Frozen: {len(all_param_names) - len(requires_grad)}")

        return requires_grad

    def _get_config_string(self) -> str:
        """Get configuration string for folder naming."""
        d_model = self.config["d_model"]
        n_layers = self.config["n_layers"]
        seq_len = self.config["max_seq_len"]
        num_classes = self.config["num_classes"]
        return f"_mamba_{d_model}_{n_layers}_{seq_len}_{num_classes}"

    def get_SSM_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get all tensor shapes inside SelectiveSSM operator."""
        B = self.config["batch_size"]
        L = self.config["max_seq_len"]
        D = self.config["d_model"] * self.config["expand_factor"]  # d_inner
        N = self.config["d_state"]
        
        return {
            # Inputs
            "x": (B, L, D),
            "A_log": (D, N),
            "B": (B, L, N),
            "C": (B, L, N),
            "D": (D,),
            "dt": (B, L, D),
            "res": (B, L, D),
            
            # Intermediates
            "A": (D, N),                    # Exp(A_log)
            "x_expanded": (B, L, D, 1),     # Reshape for broadcast
            "A_expanded": (1, 1, D, N),     # Reshape for broadcast
            "Ax": (B, L, D, N),             # x * A
            "B_expanded": (B, L, 1, N),     # Reshape for broadcast
            "state": (B, L, D, N),          # Ax + B
            "C_expanded": (B, L, 1, N),     # Reshape for broadcast
            "state_C": (B, L, D, N),        # state * C
            "y_sum": (B, L, D),             # ReduceSum over N
            "D_expanded": (1, 1, D),        # Reshape for broadcast
            "Dx_dt": (B, L, D),             # D * x * dt
            "y_skip": (B, L, D),            # y_sum + Dx_dt
            "res_act": (B, L, D),           # SiLU(res)
            
            # Output
            "y": (B, L, D),
        }

    def _inject_ssm_value_info(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Inject ValueInfoProto entries for SelectiveSSM inputs/outputs.

        This uses `get_SSM_shapes()` to map expected shapes to the actual
        input/output names of SelectiveSSM nodes in the graph. It adds
        conservative type/shape hints so shape inference can propagate
        through custom-operator boundaries.
        """
        graph = onnx_model.graph

        #collect names of all existing ValueInfoProto to avoid duplicates
        existing = {v.name for v in list(graph.input) + list(graph.value_info) + list(graph.output)}

        # Map from tensor name to element type (TensorProto data type) -> faster access
        known_elem = {}
        for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
            tensor_type = value_info.type.tensor_type if value_info.type else None
            if tensor_type and tensor_type.elem_type:
                known_elem[value_info.name] = tensor_type.elem_type
        
        #Also initializers constant tensors might be useful
        for initializer in graph.initializer:
            known_elem.setdefault(initializer.name, initializer.data_type)

        def _find_elem_type(name: str):
            return known_elem.get(name, TensorProto.FLOAT) #Default must be changed

        shapes = self.get_SSM_shapes()
        input_keys = ["x", "A_log", "B", "C", "D", "dt", "res"]

        for node in graph.node:
            if node.op_type == "SelectiveSSM" and node.domain == "ai.mamba":
    
                for inp_name, key in zip(node.input, input_keys):
                    shape = shapes.get(key)  #Expected shape for this input
                    if shape is None or inp_name in existing:
                        continue

                    #Create ValueInfoProto with conservative element type and expected shape
                    value_info = helper.make_tensor_value_info(inp_name, _find_elem_type(inp_name), list(shape))
                    
                    #Append value info to graph
                    graph.value_info.append(value_info)

                    #Update existing names and known element types
                    existing.add(inp_name)
                    known_elem[inp_name] = _find_elem_type(inp_name)
                
                #one outhput with expected shape
                if node.output and "y" in shapes:
                    out_name = node.output[0]
                    if out_name not in existing:
                        elem_type = _find_elem_type(node.input[0]) if node.input else TensorProto.FLOAT

                        #Create ValueInfoProto with conservative element type and expected shape for output
                        value_info = helper.make_tensor_value_info(out_name, elem_type, list(shapes["y"]))
                        graph.value_info.append(value_info)

                        existing.add(out_name)
                        known_elem[out_name] = elem_type

        return onnx_model

    #for memory-aware t
    def _inject_ssm_intermidiate_value_info(self, onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Inject ValueInfoProto entries for SelectiveSSM intermediate tensors.

        This is an optional step that adds ValueInfoProto entries for
        intermediate tensors inside the SelectiveSSM operator. This can
        help with debugging and visualization, but is not strictly necessary
        for inference optimization.
        """
        """This is a placeholder for future implementation if we want to add more detailed value 
        info for intermediate tensors inside the SelectiveSSM operator. 
        IDEA: intermediate tensors as metadata inside .json"""
        
        return onnx_model

    def _export_to_onnx(
        self, model: torch.nn.Module, input_tensor: torch.Tensor, opset_version: int = 17
    ):
        """
        Export Mamba model to ONNX with custom SelectiveSSM operator.

        Args:
            model: Mamba PyTorch model
            input_tensor: Sample input tensor
            opset_version: ONNX opset version (default 17)

        Returns:
            ONNX model with custom operators
        """
        import io

        import onnx

        print("\n🔧 Preparing custom ONNX export...")
        print("   Custom operator: ai.mamba::SelectiveSSM")
        print("   Opset version: 17")
        print("   Domain: ai.mamba:1")

        # Export to ONNX with custom operator domain
        f = io.BytesIO()
        torch.onnx.export(
            model,
            input_tensor,
            f,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False,
            custom_opsets={"ai.mamba": 1},  # Custom operator domain
            dynamo=False,  # Disable Dynamo for cleaner export
        )

        onnx_model = onnx.load_model_from_string(f.getvalue())
        onnx_model = self._inject_ssm_value_info(onnx_model)

        # Print export summary
        print("\n✨ ONNX Export Complete:")
        print("   Expected operators per layer:")
        print("   • LayerNorm")
        print("   • Linear (MatMul+Add)")
        print("   • Conv1d")
        print("   • Silu")
        print("   • ai.mamba::SelectiveSSM (custom operator)")
        print("   • Add (residual)")
        print("   Nodes per layer: ~10-15 (vs 50-80 with fragmented export)")

        return onnx_model

    def run_inference_optimization(self, onnx_file: str, output_file: str):
        """
        Skip aggressive optimizations to preserve custom operators.

        For Mamba with custom operators, we skip graph optimizations
        that might unfold or remove the SelectiveSSM custom operator.
        """
        print("   ⏭️  Skipping graph optimizations (preserving custom operators)")
        RemoveIdentityPass().apply(onnx_file, output_file, {})
        ShapeInferencePass().apply(output_file, output_file, {})

        if onnx_file != output_file:
            import shutil

            shutil.copy(onnx_file, output_file)

    def save_test_data(self, model: torch.nn.Module, save_dir: str):
        """
        Save test input/output data for validation.

        Uses PyTorch model to generate reference output.

        Args:
            model: PyTorch model to run inference with
            save_dir: Directory to save test data
        """
        print("💾 Saving test input/output data...")

        input_shape = self.get_input_shape()

        if self.config["use_embedding"]:
            vocab_size = self.config["vocab_size"]
            test_input = np.random.randint(0, vocab_size, size=input_shape, dtype=np.int64)
        else:
            test_input = np.random.randn(*input_shape).astype(np.float32)

        # Get PyTorch output
        was_training = model.training
        model.eval()

        with torch.no_grad():
            if self.config["use_embedding"]:
                input_tensor = torch.from_numpy(test_input).long()
            else:
                input_tensor = torch.from_numpy(test_input).float()

            output_tensor = model(input_tensor)
            test_output = output_tensor.numpy()

        if was_training:
            model.train()

        # Save as .npz files
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        np.savez(save_path / "inputs.npz", input=test_input)
        np.savez(save_path / "outputs.npz", output=test_output)

        print("  ✅ Saved test data:")
        print(f"     Input:  {save_path / 'inputs.npz'} shape={test_input.shape}")
        print(f"     Output: {save_path / 'outputs.npz'} shape={test_output.shape}")
